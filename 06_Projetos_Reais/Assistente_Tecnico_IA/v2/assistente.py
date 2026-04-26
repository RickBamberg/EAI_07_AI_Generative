"""
assistente.py  — v2 (ChromaDB)
===============================
Núcleo do Assistente Técnico IA — integra RAG semântico com ChromaDB, ToolRunner e memória.

Diferenças em relação à v1 (FAISS + pkl):
    - Banco vetorial: ChromaDB persistente em data/chroma_db/  (antes: indice_rag.pkl)
    - Filtro por módulo: where={'modulo_prefixo': {'$eq': 'EAI_01'}}  (antes: índice temporário)
    - Query expansion com consciência de histórico: resolve "desse projeto", "nele", etc.
    - Corpus: 1.763 chunks de 33 arquivos (antes: 1.553 de 26)
    - _get_indice() retorna a collection ChromaDB (mesma assinatura, compatível com app.py)
    - _pronto e a interface pública (responder, limpar_historico) são idênticos à v1

Fluxo por pergunta:
    1. Query expansion com histórico recente (resolve referências anafóricas)
    2. Busca semântica no ChromaDB (com filtro opcional por módulo)
    3. Memória de curto prazo (últimas N perguntas globais, sem distinção de usuário)
    4. Retorna resposta + fontes para o Flask exibir no painel lateral

Pré-requisito:
    O banco ChromaDB deve ter sido gerado pelo 04_rag_avancado_chromadb.ipynb.
    Caminho esperado: EAI_07_AI_Generative/data/chroma_db/
"""

import os
import sys
import json
import threading
import time
import numpy as np
import chromadb
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ── Path resolution: sobe até encontrar o EAI_07 com shared/ ─────────────
_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE.parent, _HERE.parent.parent, _HERE.parent.parent.parent]:
    if (_candidate / 'shared' / 'llm_factory.py').exists():
        sys.path.insert(0, str(_candidate))
        print(f'[assistente] shared/ encontrado em: {_candidate}')
        break

load_dotenv()

from shared.tool_runner import ToolRunner

# ── Cliente LLM ───────────────────────────────────────────────────────────
_llm = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url='https://api.deepseek.com'
)
_LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-chat')

# ── Caminhos ──────────────────────────────────────────────────────────────
_PROJETO_BASE = Path(os.getenv('PROJETO_BASE', str(_HERE.parent.parent.parent)))
_CHROMA_PATH  = _HERE.parent.parent.parent / 'data' / 'chroma_db'   # gerado pelo notebook v2
_MEMORIA_PATH = _HERE / 'data' / 'historico_global.json'
_MEMORIA_PATH.parent.mkdir(parents=True, exist_ok=True)

_COLLECTION_NAME = 'agent_contexts'

# ── Embedding + ChromaDB (pré-carregados em background thread) ────────────
_modelo_emb  = None
_INDICE_LOCK = threading.Lock()
_pronto      = False   # True quando embedding + collection estão prontos
_INDICE      = None    # será a chromadb.Collection


def _precarregar():
    """Carrega embedding e abre o banco ChromaDB em thread background — não bloqueia o Flask."""
    global _modelo_emb, _INDICE, _pronto
    print('[assistente] Pré-carregando em background...')
    try:
        _modelo_emb = SentenceTransformer('all-MiniLM-L6-v2')
        print('[assistente] Embedding pronto.')
        with _INDICE_LOCK:
            _INDICE = _abrir_colecao()
        _pronto = True
        print('[assistente] ✓ Pronto para responder.')
    except Exception as e:
        print(f'[assistente] Erro no pré-carregamento: {e}')


def _get_modelo_emb():
    """Retorna o modelo de embedding (aguarda se a thread background ainda estiver carregando)."""
    tentativas = 0
    while _modelo_emb is None and tentativas < 60:   # espera até 30s
        time.sleep(0.5)
        tentativas += 1
    return _modelo_emb


# ═══════════════════════════════════════════════════════════════════════════
# 1. BANCO VETORIAL — ChromaDB
# ═══════════════════════════════════════════════════════════════════════════

def _abrir_colecao():
    """
    Abre a collection ChromaDB existente (gerada pelo 04_rag_avancado_chromadb.ipynb).
    Não reconstrói o índice — apenas conecta ao banco persistente em disco.
    """
    if not _CHROMA_PATH.exists():
        print(f'[assistente] Banco ChromaDB não encontrado: {_CHROMA_PATH}')
        print('[assistente] Execute o 04_rag_avancado_chromadb.ipynb para gerar o banco.')
        return None

    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))

    try:
        collection = client.get_collection(name=_COLLECTION_NAME)
        print(f'[assistente] ChromaDB aberto: {collection.count()} chunks')
        return collection
    except Exception as e:
        print(f'[assistente] Erro ao abrir collection "{_COLLECTION_NAME}": {e}')
        return None


def _get_indice():
    """
    Retorna a collection ChromaDB (aguarda se o background thread ainda estiver carregando).
    Assinatura idêntica à v1 — app.py não precisa mudar.
    """
    tentativas = 0
    while _INDICE is None and tentativas < 60:   # espera até 30s
        time.sleep(0.5)
        tentativas += 1
    return _INDICE


# Dispara o pré-carregamento assim que o módulo é importado pelo Flask
threading.Thread(target=_precarregar, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════
# 2. QUERY EXPANSION COM CONSCIÊNCIA DE HISTÓRICO
# ═══════════════════════════════════════════════════════════════════════════

def _expandir_query(pergunta: str, historico_recente: list = None) -> str:
    """
    Reformula a pergunta em termos técnicos para melhorar o recall da busca.

    Quando há histórico, o LLM decide se a pergunta é:
    - CONTINUAÇÃO ("desse projeto", "nele", "qual foi a acurácia")
      → usa o histórico para resolver a referência antes de expandir
    - INDEPENDENTE (menciona explicitamente novo módulo ou tema)
      → ignora o histórico, expande só com termos técnicos

    Isso evita que perguntas sobre EAI_06 contaminem o resultado com
    contexto do EAI_03 discutido anteriormente.
    """
    if historico_recente:
        ultimas = historico_recente[-4:]   # máximo 2 trocas (user+assistant x2)
        ctx_linhas = []
        for msg in ultimas:
            role = 'Usuario' if msg['role'] == 'user' else 'Assistente'
            ctx_linhas.append(f"{role}: {msg['content'][:300]}")
        ctx_bloco = (
            '\n\nHistorico recente (use APENAS se a pergunta for continuacao):\n'
            + '\n'.join(ctx_linhas) + '\n'
        )
    else:
        ctx_bloco = ''

    prompt = (
        'Voce e um especialista em IA. Reformule a pergunta abaixo em termos tecnicos '
        'mais precisos para melhorar uma busca semantica em documentacao tecnica de IA.\n\n'
        'REGRAS:\n'
        '1. Se a pergunta for INDEPENDENTE (menciona explicitamente um modulo, tecnologia '
        'ou tema novo), ignore o historico e expanda apenas com sinonimos e termos tecnicos.\n'
        '2. Se a pergunta for CONTINUACAO (usa pronomes ou referencias como "desse projeto", '
        '"nele", "qual foi a acuracia", sem nomear o assunto), use o historico para resolver '
        'a referencia e inclua os termos concretos na query.\n'
        '3. Responda APENAS com a query reformulada, sem explicacoes. Maximo de 2 linhas.'
        + ctx_bloco
        + f'\n\nPergunta original: {pergunta}\nQuery reformulada:'
    )

    try:
        response = _llm.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return pergunta   # fallback: usa pergunta original se a expansão falhar


# ═══════════════════════════════════════════════════════════════════════════
# 3. BUSCA RAG — ChromaDB
# ═══════════════════════════════════════════════════════════════════════════

def buscar_rag(
    query: str,
    top_k: int = 5,
    score_min: float = 0.3,
    filtro_modulo: str = None,
    historico_recente: list = None,
) -> list[dict]:
    """
    Busca semântica no ChromaDB com query expansion e filtro opcional por módulo.

    Args:
        query            : pergunta do usuário
        top_k            : número máximo de chunks retornados
        score_min        : score mínimo de similaridade (0–1)
        filtro_modulo    : prefixo do módulo, ex: 'EAI_01' (opcional)
        historico_recente: últimas mensagens para a query expansion

    Retorna lista de {contexto, score, modulo, titulo, arquivo}.
    """
    collection = _get_indice()
    if collection is None:
        return []

    # Query expansion com consciência de histórico
    query_expandida = _expandir_query(query, historico_recente)

    # Gera embedding da query expandida
    emb_q = _get_modelo_emb().encode(
        [query_expandida], normalize_embeddings=True
    ).tolist()

    # Monta kwargs — where só entra se filtro_modulo for informado
    kwargs = dict(
        query_embeddings=emb_q,
        n_results=top_k,
        include=['documents', 'metadatas', 'distances'],
    )
    if filtro_modulo:
        kwargs['where'] = {'modulo_prefixo': {'$eq': filtro_modulo}}

    try:
        raw = collection.query(**kwargs)
    except Exception as e:
        print(f'[assistente] Erro na busca ChromaDB: {e}')
        return []

    resultados = []
    for doc, meta, dist in zip(
        raw['documents'][0],
        raw['metadatas'][0],
        raw['distances'][0],
    ):
        # ChromaDB retorna distância cosine [0, 2] → converte para similaridade [0, 1]
        score = round(1.0 - (dist / 2.0), 3)
        if score < score_min:
            continue
        resultados.append({
            'contexto': doc,
            'score'   : score,
            'modulo'  : meta.get('modulo', ''),
            'titulo'  : meta.get('titulo', ''),
            'arquivo' : meta.get('arquivo', ''),
        })

    return resultados


# ═══════════════════════════════════════════════════════════════════════════
# 4. MEMÓRIA GLOBAL (últimas N interações, sem distinção de usuário)
# ═══════════════════════════════════════════════════════════════════════════

_MAX_HISTORICO = 20   # número de turns (user + assistant = 2 por turn)


def _carregar_historico() -> list:
    if _MEMORIA_PATH.exists():
        with open(_MEMORIA_PATH, encoding='utf-8') as f:
            return json.load(f)
    return []


def _salvar_historico(historico: list):
    with open(_MEMORIA_PATH, 'w', encoding='utf-8') as f:
        json.dump(historico[-_MAX_HISTORICO:], f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# 5. SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM = """Você é o Assistente Técnico do curso Especialista em IA de Carlos Henrique.

ESTRUTURA COMPLETA DO CURSO (você conhece isso — não dependa do contexto RAG):
1. EAI_01 — Fundamentos Matemáticos para IA (álgebra linear, regressão, gradiente)
2. EAI_02 — Machine Learning Clássico (KNN, SVM, Random Forest, XGBoost)
3. EAI_03 — Deep Learning (ANN, CNN, RNN, LSTM, Transfer Learning)
4. EAI_04 — NLP Clássico (TF-IDF, embeddings, classificação de texto)
5. EAI_05 — Visão Computacional (OpenCV, YOLO, detecção de objetos)
6. EAI_06 — NLP Moderno / Transformers (BERT, GPT, fine-tuning)
7. EAI_07 — IA Generativa (LLMs, RAG, agentes, function calling)
8. EAI_08 — MLOps (FastAPI, monitoramento, drift, deploy)

REGRAS:
- Perguntas sobre a lista, estrutura ou visão geral do curso: responda DIRETAMENTE
  com a estrutura acima. NÃO mencione "contexto recuperado" nesse caso.
- Perguntas técnicas específicas: use os trechos de contexto RAG quando fornecidos.
- Nunca diga que um módulo "não está no contexto" ao listar os módulos do curso.
- Seja preciso e técnico. Use markdown para código e listas.

INFORMAÇÕES TÉCNICAS DO EAI_07 que você já conhece:
- Provider LLM é controlado pelo arquivo .env na raiz do EAI_07
- Para trocar de provider: editar LLM_PROVIDER, LLM_MODEL e a API_KEY correspondente
  Exemplo DeepSeek:  LLM_PROVIDER=deepseek  | LLM_MODEL=deepseek-chat  | DEEPSEEK_API_KEY=sk-...
  Exemplo OpenAI:    LLM_PROVIDER=openai    | LLM_MODEL=gpt-4o-mini    | OPENAI_API_KEY=sk-...
  Exemplo Ollama:    LLM_PROVIDER=ollama    | LLM_MODEL=llama3.2       | (sem chave)
- O shared/llm_factory.py abstrai o provider — os notebooks não precisam mudar
- Provider atual do Carlos: DeepSeek (deepseek-chat)
"""


# ═══════════════════════════════════════════════════════════════════════════
# 6. FUNÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

# Palavras que indicam pergunta geral — não precisa de RAG
_PERGUNTAS_GERAIS = {
    'módulos', 'modulos', 'lista', 'estrutura', 'visão geral', 'visao geral',
    'quais módulos', 'quantos módulos', 'o que é o curso', 'sobre o curso',
}


def _precisa_rag(pergunta: str) -> bool:
    """Retorna False para perguntas gerais que o system prompt já responde."""
    p = pergunta.lower()
    if any(k in p for k in _PERGUNTAS_GERAIS):
        if not any(t in p for t in ['como', 'implementa', 'código', 'função', 'algoritmo']):
            return False
    return True


def responder(pergunta: str) -> dict:
    """
    Processa uma pergunta e retorna resposta + fontes + histórico atualizado.
    Interface pública idêntica à v1 — app.py não precisa mudar.

    Retorno:
        {
            'resposta'  : str,
            'fontes'    : [{modulo, titulo, score, trecho, arquivo}],
            'historico' : [{role, content, timestamp}],
            'timestamp' : str,
        }
    """
    # ── 1. Carrega histórico (usado na query expansion e no contexto do LLM) ──
    historico = _carregar_historico()

    # ── 2. Busca RAG com query expansion ──────────────────────────────────
    usar_rag = _precisa_rag(pergunta)
    resultados_rag = (
        buscar_rag(
            pergunta,
            top_k=5,
            score_min=0.45,
            historico_recente=historico[-4:],   # passa últimas 2 trocas para expansion
        )
        if usar_rag else []
    )

    # ── 3. Monta prompt ───────────────────────────────────────────────────
    if resultados_rag:
        contexto_texto = '\n\n---\n\n'.join(
            f'[{r["modulo"]} — {r["titulo"]}] (score: {r["score"]})\n{r["contexto"]}'
            for r in resultados_rag
        )
        prompt_com_contexto = (
            f'Contexto técnico dos módulos do curso:\n\n{contexto_texto}'
            f'\n\n---\n\nPergunta: {pergunta}'
        )
    else:
        prompt_com_contexto = pergunta

    # ── 4. Chamada ao LLM com histórico ───────────────────────────────────
    msgs_contexto = historico[-_MAX_HISTORICO:]
    messages = (
        [{'role': 'system', 'content': _SYSTEM}]
        + [{'role': m['role'], 'content': m['content']} for m in msgs_contexto]
        + [{'role': 'user', 'content': prompt_com_contexto}]
    )

    resp = _llm.chat.completions.create(
        model=_LLM_MODEL,
        messages=messages,
    )
    resposta = resp.choices[0].message.content

    # ── 5. Atualiza histórico ─────────────────────────────────────────────
    ts = datetime.now().strftime('%H:%M')
    historico.append({'role': 'user',      'content': pergunta, 'timestamp': ts})
    historico.append({'role': 'assistant', 'content': resposta,  'timestamp': ts})
    _salvar_historico(historico)

    # ── 6. Prepara fontes para o painel lateral ───────────────────────────
    fontes = [
        {
            'modulo'  : r['modulo'],
            'titulo'  : r['titulo'],
            'score'   : r['score'],
            'arquivo' : r['arquivo'],   # campo extra disponível no ChromaDB
            'trecho'  : r['contexto'][:300] + ('...' if len(r['contexto']) > 300 else ''),
        }
        for r in resultados_rag
    ]

    return {
        'resposta'  : resposta,
        'fontes'    : fontes,
        'historico' : historico[-10:],   # últimas 5 trocas para o front
        'timestamp' : ts,
    }


def limpar_historico():
    """Apaga o histórico global de conversas."""
    if _MEMORIA_PATH.exists():
        _MEMORIA_PATH.unlink()
    return {'status': 'ok', 'mensagem': 'Histórico apagado.'}

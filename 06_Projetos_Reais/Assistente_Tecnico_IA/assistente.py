"""
assistente.py
=============
Núcleo do Assistente Técnico IA — integra RAG semântico, ToolRunner e memória.

Fluxo por pergunta:
    1. Busca semântica no índice FAISS (AGENT_CONTEXT.md dos módulos EAI_01-EAI_08)
    2. ToolRunner com os chunks como contexto
    3. Memória de curto prazo (últimas N perguntas globais, sem distinção de usuário)
    4. Retorna resposta + fontes para o Flask exibir no painel lateral
"""

import os
import sys
import json
import pickle
import time
import faiss
import numpy as np
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ── Path resolution: sobe até encontrar o EAI_07 com shared/ ─────────────
_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE.parent, _HERE.parent.parent]:
    if (_candidate / 'shared' / 'llm_factory.py').exists():
        sys.path.insert(0, str(_candidate))
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
_PROJETO_BASE  = Path(os.getenv('PROJETO_BASE', str(_HERE.parent.parent.parent)))
_CACHE_RAG     = _HERE.parent.parent / 'data' / 'cache' / 'indice_rag.pkl'
_MEMORIA_PATH  = _HERE / 'data' / 'historico_global.json'
_MEMORIA_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Embedding + Índice (pré-carregados em background thread) ─────────────
import threading

_modelo_emb  = None
_INDICE_LOCK = threading.Lock()
_pronto      = False   # True quando embedding + índice estão carregados


def _precarregar():
    """Carrega embedding e índice em thread background — não bloqueia o Flask."""
    global _modelo_emb, _INDICE, _pronto
    print('[assistente] Pré-carregando em background...')
    try:
        _modelo_emb = SentenceTransformer('all-MiniLM-L6-v2')
        print('[assistente] Embedding pronto.')
        with _INDICE_LOCK:
            _INDICE = _carregar_indice()
        _pronto = True
        print('[assistente] ✓ Pronto para responder.')
    except Exception as e:
        print(f'[assistente] Erro no pré-carregamento: {e}')


def _get_modelo_emb():
    """Retorna o modelo de embedding (aguarda se a thread background ainda estiver carregando)."""
    import time
    tentativas = 0
    while _modelo_emb is None and tentativas < 60:   # espera até 30s
        time.sleep(0.5)
        tentativas += 1
    return _modelo_emb


# ═══════════════════════════════════════════════════════════════════════════
# 1. ÍNDICE RAG
# ═══════════════════════════════════════════════════════════════════════════

def _carregar_indice() -> dict | None:
    """Carrega o índice FAISS do cache do 03_RAG."""
    if not _CACHE_RAG.exists():
        print(f'[assistente] Cache RAG não encontrado: {_CACHE_RAG}')
        print('[assistente] Execute o 03_rag_basico.ipynb para gerar o índice.')
        return None
    with open(_CACHE_RAG, 'rb') as f:
        dados = pickle.load(f)
    indice = faiss.deserialize_index(dados['faiss_bytes'])
    print(f'[assistente] Índice RAG carregado: {indice.ntotal} chunks')
    return {'faiss': indice, 'chunks': dados['chunks'], 'embs': dados['embs']}


# Índice carregado de forma lazy — não bloqueia o startup do Flask
_INDICE = None

def _get_indice():
    """Retorna o índice FAISS (aguarda se a thread background ainda estiver carregando)."""
    import time
    tentativas = 0
    while _INDICE is None and tentativas < 60:   # espera até 30s
        time.sleep(0.5)
        tentativas += 1
    return _INDICE


# Dispara o pré-carregamento assim que o módulo é importado pelo Flask
threading.Thread(target=_precarregar, daemon=True).start()


def buscar_rag(query: str, top_k: int = 5, score_min: float = 0.3) -> list[dict]:
    """
    Busca semântica no índice FAISS.
    Retorna lista de {contexto, score, modulo, titulo}.
    """
    indice = _get_indice()
    if indice is None:
        return []
    emb_q = _get_modelo_emb().encode([query], normalize_embeddings=True).astype(np.float32)
    scores, pos = indice['faiss'].search(emb_q, top_k)
    resultados = []
    for j, idx in enumerate(pos[0]):
        score = float(scores[0][j])
        if score < score_min:
            continue
        chunk = indice['chunks'][idx]
        resultados.append({
            'contexto': chunk['chunk_contexto'],
            'score'   : round(score, 3),
            'modulo'  : chunk.get('modulo', ''),
            'titulo'  : chunk.get('titulo', ''),
        })
    return resultados


# ═══════════════════════════════════════════════════════════════════════════
# 2. MEMÓRIA GLOBAL (últimas N interações, sem distinção de usuário)
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
# 3. SYSTEM PROMPT
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
# 4. FUNÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

# Palavras que indicam pergunta geral — não precisa de RAG
_PERGUNTAS_GERAIS = {
    'módulos', 'modulos', 'lista', 'estrutura', 'visão geral', 'visao geral',
    'quais módulos', 'quantos módulos', 'o que é o curso', 'sobre o curso',
}

def _precisa_rag(pergunta: str) -> bool:
    """Retorna False para perguntas gerais que o system prompt já responde."""
    p = pergunta.lower()
    # Pergunta sobre lista/estrutura de módulos → não precisa de RAG
    if any(k in p for k in _PERGUNTAS_GERAIS):
        if not any(t in p for t in ['como', 'implementa', 'código', 'função', 'algoritmo']):
            return False
    return True


def responder(pergunta: str) -> dict:
    """
    Processa uma pergunta e retorna resposta + fontes + histórico atualizado.

    Retorno:
        {
            'resposta'  : str,
            'fontes'    : [{modulo, titulo, score, trecho}],
            'historico' : [{role, content, timestamp}],
            'timestamp' : str,
        }
    """
    # ── 1. Busca RAG (só para perguntas técnicas) ─────────────────────────
    usar_rag       = _precisa_rag(pergunta)
    resultados_rag = buscar_rag(pergunta, top_k=5, score_min=0.45) if usar_rag else []

    # Monta prompt: só injeta contexto RAG se houver e for relevante
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
        # Pergunta geral — LLM responde com o conhecimento do system prompt
        prompt_com_contexto = pergunta

    # ── 2. Histórico de conversa ──────────────────────────────────────────
    historico = _carregar_historico()
    msgs_contexto = historico[-_MAX_HISTORICO:]

    # ── 3. Chamada ao LLM ─────────────────────────────────────────────────
    messages = (
        [{'role': 'system', 'content': _SYSTEM}]
        + [{'role': m['role'], 'content': m['content']} for m in msgs_contexto]
        + [{'role': 'user', 'content': prompt_com_contexto}]
    )

    resp = _llm.chat.completions.create(
        model    = _LLM_MODEL,
        messages = messages,
    )
    resposta = resp.choices[0].message.content

    # ── 4. Atualiza histórico ─────────────────────────────────────────────
    ts = datetime.now().strftime('%H:%M')
    historico.append({'role': 'user',      'content': pergunta, 'timestamp': ts})
    historico.append({'role': 'assistant', 'content': resposta,  'timestamp': ts})
    _salvar_historico(historico)

    # ── 5. Prepara fontes para o painel lateral ───────────────────────────
    fontes = [
        {
            'modulo' : r['modulo'],
            'titulo' : r['titulo'],
            'score'  : r['score'],
            'trecho' : r['contexto'][:300] + ('...' if len(r['contexto']) > 300 else ''),
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

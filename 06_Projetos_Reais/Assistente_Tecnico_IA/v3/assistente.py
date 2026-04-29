"""
assistente.py  — v3 (LangChain 1.x / LCEL)
============================================
Núcleo do Assistente Técnico IA — integra RAG semântico com LangChain, ToolRunner e memória.

Diferenças em relação à v2 (ChromaDB puro):
    - Vectorstore  : Chroma (LangChain) em vez de chromadb.PersistentClient direto
    - Embeddings   : HuggingFaceEmbeddings em vez de SentenceTransformer.encode() manual
    - LLM          : ChatOpenAI (LangChain) em vez de openai.OpenAI()
    - Query Exp.   : PromptTemplate | ChatOpenAI | StrOutputParser  (LCEL chain)
    - LLM principal: ChatPromptTemplate | ChatOpenAI | StrOutputParser  (LCEL chain)
    - Banco        : mesmo chroma_db/ da v2 — nenhuma reindexação necessária
    - _get_modelo_emb() removido — embeddings gerados internamente pelo vectorstore
    - _pronto e interface pública (responder, limpar_historico) idênticos à v2

Fluxo por pergunta:
    1. Query expansion com histórico recente (LCEL: prompt | llm | parser)
    2. Busca semântica via vectorstore.similarity_search_with_score()
    3. Geração da resposta (LCEL: prompt | llm | parser) com contexto RAG + histórico
    4. Retorna resposta + fontes para o Flask exibir no painel lateral

Pré-requisito:
    O banco ChromaDB deve ter sido gerado pelo 04_rag_avancado_chromadb.ipynb
    ou pelo 04_rag_avancado_langchain.ipynb (banco compartilhado).
    Caminho esperado: EAI_07_AI_Generative/data/chroma_db/

Instalação das dependências LangChain (conda activate eai07):
    pip install langchain langchain-community langchain-chroma langchain-openai langchain-huggingface
"""

import os
import sys
import json
import threading
import time
from pathlib import Path
from datetime import datetime

# ── LangChain 1.x ────────────────────────────────────────────────────────────
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

# ── Path resolution: sobe até encontrar o EAI_07 com shared/ ─────────────────
_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE.parent, _HERE.parent.parent, _HERE.parent.parent.parent]:
    if (_candidate / 'shared' / 'llm_factory.py').exists():
        sys.path.insert(0, str(_candidate))
        print(f'[assistente] shared/ encontrado em: {_candidate}')
        break

load_dotenv()

from shared.tool_runner import ToolRunner

# ── Caminhos ──────────────────────────────────────────────────────────────────
_PROJETO_BASE    = Path(os.getenv('PROJETO_BASE', str(_HERE.parent.parent.parent)))
_CHROMA_PATH     = _HERE.parent / 'data' / 'chroma_db'
_MEMORIA_PATH  = _HERE.parent / 'data' / 'historico' / 'v3_historico_global.json'
_MEMORIA_PATH.parent.mkdir(parents=True, exist_ok=True)

_COLLECTION_NAME = 'agent_contexts'

# ── Componentes LangChain (pré-carregados em background thread) ───────────────
_embedding_function = None   # HuggingFaceEmbeddings
_vectorstore        = None   # Chroma (LangChain)
_llm                = None   # ChatOpenAI
_expansion_chain    = None   # PromptTemplate | llm | StrOutputParser
_rag_chain          = None   # ChatPromptTemplate | llm | StrOutputParser
_INDICE_LOCK        = threading.Lock()
_pronto             = False  # True quando todos os componentes estão prontos


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PRÉ-CARREGAMENTO EM BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════════

def _precarregar():
    """
    Inicializa embeddings, vectorstore e chains LangChain em thread background.
    Não bloqueia o Flask — _pronto sinaliza quando está pronto.
    """
    global _embedding_function, _vectorstore, _llm, _expansion_chain, _rag_chain, _pronto

    print('[assistente] Pré-carregando em background...')
    try:
        # ── Embeddings — mesmo modelo all-MiniLM-L6-v2 ──────────────────────
        # HuggingFaceEmbeddings encapsula SentenceTransformer internamente.
        # normalize_embeddings=True mantém compatibilidade com o banco existente.
        _embedding_function = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        print('[assistente] Embeddings prontos.')

        # ── Vectorstore — abre o banco ChromaDB existente ────────────────────
        # Mesmo banco da v2: chroma_db/ é compartilhado entre as versões.
        # Nenhuma reindexação necessária se o banco já estiver populado.
        with _INDICE_LOCK:
            _vectorstore = _abrir_vectorstore()

        # ── LLM — DeepSeek via interface OpenAI-compatível ───────────────────
        _llm = ChatOpenAI(
            model=os.getenv('LLM_MODEL', 'deepseek-chat'),
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url='https://api.deepseek.com',
            temperature=0.2,
            max_tokens=600,
        )

        # ── LCEL chains ───────────────────────────────────────────────────────
        _expansion_chain = _build_expansion_chain()
        _rag_chain       = _build_rag_chain()

        _pronto = True
        print('[assistente] ✓ Pronto para responder.')

    except Exception as e:
        print(f'[assistente] Erro no pré-carregamento: {e}')


def _get_vectorstore():
    """Retorna o vectorstore (aguarda até 30s se o background thread ainda estiver carregando)."""
    tentativas = 0
    while _vectorstore is None and tentativas < 60:
        time.sleep(0.5)
        tentativas += 1
    return _vectorstore


def _get_chains():
    """Retorna (expansion_chain, rag_chain) após o pré-carregamento."""
    tentativas = 0
    while (_expansion_chain is None or _rag_chain is None) and tentativas < 60:
        time.sleep(0.5)
        tentativas += 1
    return _expansion_chain, _rag_chain


# Dispara o pré-carregamento assim que o módulo é importado pelo Flask
threading.Thread(target=_precarregar, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BANCO VETORIAL — Chroma (LangChain)
# ═══════════════════════════════════════════════════════════════════════════════

def _abrir_vectorstore():
    """
    Abre o banco ChromaDB existente via LangChain.

    Diferença da v2:
        v2 → chromadb.PersistentClient(path=...).get_collection(name=...)
        v3 → Chroma(collection_name=..., embedding_function=..., persist_directory=...)

    O banco em disco é o mesmo — só a camada de acesso muda.
    """
    if not _CHROMA_PATH.exists():
        print(f'[assistente] Banco ChromaDB não encontrado: {_CHROMA_PATH}')
        print('[assistente] Execute o 04_rag_avancado_chromadb.ipynb ou '
              '04_rag_avancado_langchain.ipynb para gerar o banco.')
        return None

    try:
        vs = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=_embedding_function,
            persist_directory=str(_CHROMA_PATH),
            collection_metadata={'hnsw:space': 'cosine'},
        )
        total = vs._collection.count()
        print(f'[assistente] ChromaDB (LangChain) aberto: {total} chunks')
        return vs
    except Exception as e:
        print(f'[assistente] Erro ao abrir vectorstore "{_COLLECTION_NAME}": {e}')
        return None


# ── Compatibilidade com app.py (mesma assinatura da v2) ──────────────────────
def _get_indice():
    """
    Mantido para compatibilidade com app.py.
    Na v3 retorna o vectorstore LangChain em vez da collection ChromaDB.
    """
    return _get_vectorstore()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LCEL CHAINS — Query Expansion e RAG
# ═══════════════════════════════════════════════════════════════════════════════

_EXPANSION_TEMPLATE = PromptTemplate.from_template(
    """Voce e um especialista em IA. Reformule a pergunta abaixo em termos tecnicos \
mais precisos para melhorar uma busca semantica em documentacao tecnica de IA.

REGRAS:
1. Se a pergunta for INDEPENDENTE (menciona explicitamente um modulo, tecnologia \
ou tema novo), ignore o historico e expanda apenas com sinonimos e termos tecnicos.
2. Se a pergunta for CONTINUACAO (usa pronomes ou referencias como "desse projeto", \
"nele", "qual foi a acuracia", sem nomear o assunto), use o historico para resolver \
a referencia e inclua os termos concretos na query.
3. Responda APENAS com a query reformulada, sem explicacoes. Maximo de 2 linhas.{ctx_bloco}

Pergunta original: {pergunta}
Query reformulada:"""
)

def _build_expansion_chain():
    """Constrói a chain LCEL de query expansion: prompt | llm | parser."""
    # max_tokens pequeno — só precisa de uma linha reformulada
    llm_expansion = ChatOpenAI(
        model=os.getenv('LLM_MODEL', 'deepseek-chat'),
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url='https://api.deepseek.com',
        temperature=0.0,
        max_tokens=120,
    )
    return _EXPANSION_TEMPLATE | llm_expansion | StrOutputParser()


def _build_rag_chain():
    """
    Constrói a chain LCEL principal: ChatPromptTemplate | llm | parser.

    Aceita variáveis: system, historico (lista de mensagens), contexto, pergunta.
    O histórico é injetado como lista de dicts {role, content} e convertido
    em mensagens HumanMessage/AIMessage antes de invocar.
    """
    # Usamos um template simples — o histórico é montado manualmente em responder()
    # para manter compatibilidade com o formato {role, content, timestamp} do JSON em disco.
    prompt = ChatPromptTemplate.from_messages([
        ('system', '{system}'),
        MessagesPlaceholder(variable_name='historico'),
        ('human', '{prompt_usuario}'),
    ])
    return prompt | _llm | StrOutputParser()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. QUERY EXPANSION COM CONSCIÊNCIA DE HISTÓRICO
# ═══════════════════════════════════════════════════════════════════════════════

def _expandir_query(pergunta: str, historico_recente: list = None) -> str:
    """
    Reformula a pergunta em termos técnicos para melhorar o recall da busca.

    Diferença da v2:
        v2 → _llm.chat.completions.create(messages=[...])
        v3 → expansion_chain.invoke({'pergunta': ..., 'ctx_bloco': ...})

    A lógica CONTINUAÇÃO / INDEPENDENTE é preservada — só muda a chamada ao LLM.
    """
    if historico_recente:
        ultimas = historico_recente[-4:]
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

    chain, _ = _get_chains()
    if chain is None:
        return pergunta   # fallback: pré-carregamento ainda não concluiu

    try:
        return chain.invoke({'pergunta': pergunta, 'ctx_bloco': ctx_bloco})
    except Exception:
        return pergunta   # fallback: usa pergunta original se a expansão falhar


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BUSCA RAG — Chroma (LangChain)
# ═══════════════════════════════════════════════════════════════════════════════

def buscar_rag(
    query: str,
    top_k: int = 5,
    score_min: float = 0.3,
    filtro_modulo: str = None,
    historico_recente: list = None,
) -> list[dict]:
    """
    Busca semântica via LangChain com query expansion e filtro opcional por módulo.

    Diferença da v2:
        v2 → collection.query(query_embeddings=emb_q.tolist(), n_results=top_k, where=...)
        v3 → vectorstore.similarity_search_with_score(query, k=top_k, filter=...)
             O vectorstore gera o embedding internamente — não é mais necessário chamar
             _get_modelo_emb().encode() manualmente.

    O filtro usa a mesma sintaxe $eq do ChromaDB puro — o LangChain repassa direto.

    Retorna lista de {contexto, score, modulo, titulo, arquivo}.
    """
    vs = _get_vectorstore()
    if vs is None:
        return []

    # Query expansion com consciência de histórico
    query_expandida = _expandir_query(query, historico_recente)

    # Monta kwargs — filter só entra se filtro_modulo for informado
    kwargs = {'k': top_k}
    if filtro_modulo:
        kwargs['filter'] = {'modulo_prefixo': {'$eq': filtro_modulo}}

    try:
        # Retorna list[tuple[Document, float]]
        # Document.page_content = chunk_busca (texto de busca)
        # Document.metadata     = {'modulo', 'modulo_prefixo', 'titulo', 'arquivo', '_contexto'}
        # float                 = distância cosine ChromaDB [0, 2]
        pares = vs.similarity_search_with_score(query_expandida, **kwargs)
    except Exception as e:
        print(f'[assistente] Erro na busca LangChain: {e}')
        return []

    resultados = []
    for doc, dist in pares:
        # Converte distância cosine → similaridade [0, 1]
        score = round(1.0 - (dist / 2.0), 3)
        if score < score_min:
            continue

        # _contexto foi armazenado como metadado na indexação LangChain.
        # Fallback para page_content se o banco foi gerado pelo ChromaDB puro
        # (que armazena o chunk_busca como document, sem _contexto separado).
        contexto = doc.metadata.get('_contexto', doc.page_content)

        resultados.append({
            'contexto': contexto,
            'score'   : score,
            'modulo'  : doc.metadata.get('modulo', ''),
            'titulo'  : doc.metadata.get('titulo', ''),
            'arquivo' : doc.metadata.get('arquivo', ''),
        })

    return resultados


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MEMÓRIA GLOBAL (últimas N interações, sem distinção de usuário)
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_HISTORICO = 20   # número de mensagens (user + assistant = 2 por turn)


def _carregar_historico() -> list:
    if _MEMORIA_PATH.exists():
        with open(_MEMORIA_PATH, encoding='utf-8') as f:
            return json.load(f)
    return []


def _salvar_historico(historico: list):
    with open(_MEMORIA_PATH, 'w', encoding='utf-8') as f:
        json.dump(historico[-_MAX_HISTORICO:], f, ensure_ascii=False, indent=2)


def _historico_para_mensagens(historico: list) -> list:
    """
    Converte o histórico JSON {role, content, timestamp} em objetos LangChain
    HumanMessage / AIMessage para passar ao ChatPromptTemplate.

    Diferença da v2:
        v2 → [{'role': m['role'], 'content': m['content']} for m in historico]
        v3 → [HumanMessage(...) | AIMessage(...) for m in historico]
    """
    msgs = []
    for m in historico:
        if m['role'] == 'user':
            msgs.append(HumanMessage(content=m['content']))
        elif m['role'] == 'assistant':
            msgs.append(AIMessage(content=m['content']))
    return msgs


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 8. LÓGICA DE RAG CONDICIONAL
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FUNÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def responder(pergunta: str) -> dict:
    """
    Processa uma pergunta e retorna resposta + fontes + histórico atualizado.
    Interface pública idêntica à v2 — app.py não precisa mudar.

    Diferença interna da v2:
        v2 → _llm.chat.completions.create(messages=[{'role':..., 'content':...}])
        v3 → rag_chain.invoke({'system':..., 'historico': [HumanMessage|AIMessage], 'prompt_usuario':...})

    Retorno:
        {
            'resposta'  : str,
            'fontes'    : [{modulo, titulo, score, trecho, arquivo}],
            'historico' : [{role, content, timestamp}],
            'timestamp' : str,
        }
    """
    # ── 1. Carrega histórico ───────────────────────────────────────────────────
    historico = _carregar_historico()

    # ── 2. Busca RAG com query expansion ─────────────────────────────────────
    usar_rag = _precisa_rag(pergunta)
    resultados_rag = (
        buscar_rag(
            pergunta,
            top_k=5,
            score_min=0.45,
            historico_recente=historico[-4:],
        )
        if usar_rag else []
    )

    # ── 3. Monta prompt ───────────────────────────────────────────────────────
    if resultados_rag:
        contexto_texto = '\n\n---\n\n'.join(
            f'[{r["modulo"]} — {r["titulo"]}] (score: {r["score"]})\n{r["contexto"]}'
            for r in resultados_rag
        )
        prompt_usuario = (
            f'Contexto técnico dos módulos do curso:\n\n{contexto_texto}'
            f'\n\n---\n\nPergunta: {pergunta}'
        )
    else:
        prompt_usuario = pergunta

    # ── 4. Chamada ao LLM via LCEL chain ──────────────────────────────────────
    # Converte histórico JSON → objetos LangChain para o MessagesPlaceholder
    msgs_historico = _historico_para_mensagens(historico[-_MAX_HISTORICO:])

    _, rag_chain = _get_chains()

    if rag_chain is None:
        # Pré-carregamento ainda não concluiu (não deve ocorrer em uso normal)
        return {
            'resposta'  : 'Assistente ainda carregando, tente novamente em instantes.',
            'fontes'    : [],
            'historico' : [],
            'timestamp' : datetime.now().strftime('%H:%M'),
        }

    resposta = rag_chain.invoke({
        'system'        : _SYSTEM,
        'historico'     : msgs_historico,
        'prompt_usuario': prompt_usuario,
    })

    # ── 5. Atualiza histórico ─────────────────────────────────────────────────
    ts = datetime.now().strftime('%H:%M')
    historico.append({'role': 'user',      'content': pergunta, 'timestamp': ts})
    historico.append({'role': 'assistant', 'content': resposta,  'timestamp': ts})
    _salvar_historico(historico)

    # ── 6. Prepara fontes para o painel lateral ───────────────────────────────
    fontes = [
        {
            'modulo'  : r['modulo'],
            'titulo'  : r['titulo'],
            'score'   : r['score'],
            'arquivo' : r['arquivo'],
            'trecho'  : r['contexto'][:300] + ('...' if len(r['contexto']) > 300 else ''),
        }
        for r in resultados_rag
    ]

    return {
        'resposta'  : resposta,
        'fontes'    : fontes,
        'historico' : historico[-10:],
        'timestamp' : ts,
    }


def limpar_historico():
    """Apaga o histórico global de conversas."""
    if _MEMORIA_PATH.exists():
        _MEMORIA_PATH.unlink()
    return {'status': 'ok', 'mensagem': 'Histórico apagado.'}

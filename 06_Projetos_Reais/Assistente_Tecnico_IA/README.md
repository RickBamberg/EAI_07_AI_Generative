# Assistente Técnico IA

**Projeto de** EAI_07_AI_Generative / 06_Projetos_Reais  
**Ambiente:** `eai07` (Python 3.11)

Aplicação Flask com interface de chat que responde perguntas sobre os módulos
EAI_01 a EAI_08 usando RAG semântico sobre os `AGENT_CONTEXT.md` do curso.

---

## Versões disponíveis

O projeto possui três versões de backend RAG, todas com a mesma interface Flask e o mesmo `app.py`:

| Versão | Backend | Banco | Chunks | Query Expansion |
|---|---|---|---|---|
| `v1` | FAISS + pkl | `data/cache/indice_rag.pkl` | 1.553 | Não |
| `v2` | ChromaDB puro | `data/chroma_db/` | 1.763 | Sim |
| `v3` | LangChain 1.x | `data/chroma_db/` (mesmo da v2) | 1.763 | Sim |

> **v3 não requer reindexação** se a v2 já foi executada — o banco é compartilhado.

---

## Pré-requisitos

1. Ambiente `eai07` ativo
2. **Índice RAG gerado** para a versão escolhida:
   - v1 → execute `03_RAG/03_rag_basico.ipynb` → gera `data/cache/indice_rag.pkl`
   - v2 → execute `03_RAG/04_rag_avancado_chromadb.ipynb` → gera `data/chroma_db/`
   - v3 → mesmo banco da v2 (ou execute `03_RAG/04_rag_avancado_langchain.ipynb`)
3. Dependências instaladas:

```bash
conda activate eai07
pip install flask

# v2 — ChromaDB puro
pip install chromadb

# v3 — LangChain 1.x
pip install langchain langchain-community langchain-chroma langchain-openai langchain-huggingface
```

---

## Como executar

O `app.py` fica na raiz do projeto. Antes de iniciar, edite a linha de import
para apontar para a versão desejada:

```python
# app.py — escolha uma linha e comente as outras:
from v1.assistente import responder, limpar_historico, _get_indice, _pronto  # FAISS + pkl
from v2.assistente import responder, limpar_historico, _get_indice, _pronto  # ChromaDB
from v3.assistente import responder, limpar_historico, _get_indice, _pronto  # LangChain
```

```bash
cd EAI_07_AI_Generative/06_Projetos_Reais/Assistente_Tecnico_IA
python app.py
```

Acesse: **http://localhost:5000**

---

## Estrutura

```
Assistente_Tecnico_IA/
├── app.py                     ← Flask: rotas / /chat /limpar /status (idêntico nas 3 versões)
├── templates/
│   └── index.html             ← Interface de chat com painel de fontes (idêntica nas 3 versões)
├── data/
│   └── historico_global.json  ← Histórico persistido (criado automaticamente)
├── v1/
│   └── assistente.py          ← Núcleo v1: RAG com FAISS + pkl
├── v2/
│   └── assistente.py          ← Núcleo v2: RAG com ChromaDB puro
├── v3/
│   └── assistente.py          ← Núcleo v3: RAG com LangChain 1.x / LCEL
├── AGENT_CONTEXT.md
└── README.md
```

---

## Arquitetura

```
POST /chat
    │
    ▼
assistente.responder(pergunta)          ← mesma assinatura nas 3 versões
    │
    ├── _precisa_rag()                  ← perguntas gerais não usam RAG (v2 e v3)
    │
    ├── _expandir_query()               ← query expansion com histórico (v2 e v3)
    │     v2: openai.OpenAI().chat.completions.create()
    │     v3: PromptTemplate | ChatOpenAI | StrOutputParser  (LCEL)
    │
    ├── buscar_rag()
    │     v1: FAISS (indice_rag.pkl)    — 1.553 chunks, 26 arquivos
    │     v2: ChromaDB collection.query()  — 1.763 chunks, 33 arquivos
    │     v3: Chroma.similarity_search_with_score()  — mesmo banco da v2
    │
    ├── _carregar_historico()           ← últimas 20 msgs do historico_global.json
    │
    ├── LLM com system + histórico + contexto RAG
    │     v2: openai.OpenAI().chat.completions.create()
    │     v3: ChatPromptTemplate | ChatOpenAI | StrOutputParser  (LCEL)
    │
    └── _salvar_historico()
    │
    └── retorna {resposta, fontes, historico, timestamp}
```

---

## O que mudou da v2 para a v3

A v3 usa **LangChain 1.x** no lugar das chamadas diretas ao ChromaDB e OpenAI.
O banco em disco, a lógica de busca e a interface pública são idênticos.

| Componente | v2 (ChromaDB puro) | v3 (LangChain 1.x) |
|---|---|---|
| Vectorstore | `chromadb.PersistentClient` + `collection` | `Chroma(persist_directory=...)` |
| Embeddings | `SentenceTransformer.encode()` manual | `HuggingFaceEmbeddings` (interno) |
| LLM | `openai.OpenAI().chat.completions.create()` | `ChatOpenAI` + LCEL chains |
| Query Expansion | chamada direta ao LLM | `PromptTemplate \| llm \| StrOutputParser` |
| Busca | `collection.query(query_embeddings=...)` | `vs.similarity_search_with_score(query, ...)` |
| Histórico no LLM | lista de dicts `{role, content}` | `[HumanMessage \| AIMessage]` |
| `_get_modelo_emb()` | presente | removida (desnecessária) |
| Banco em disco | `data/chroma_db/` | **mesmo** `data/chroma_db/` |

> O `app.py` **não precisa mudar** além da linha de import — a interface pública é idêntica.

---

## LangChain 1.x — módulos corretos

A v3 usa LangChain na versão **1.x**, que removeu os módulos legados da versão 0.x.
Se você já conhece LangChain 0.x, atente para as mudanças:

| Removido (0.x) | Correto (1.x) |
|---|---|
| `langchain.memory.ConversationBufferWindowMemory` | `langchain_community.chat_message_histories.ChatMessageHistory` |
| `langchain.chains.ConversationalRetrievalChain` | LCEL: `prompt \| llm \| StrOutputParser()` |
| `langchain.prompts` | `langchain_core.prompts` |
| `langchain.schema` (mensagens) | `langchain_core.messages` |

---

## Interface

- **Chat** à esquerda — mensagens com renderização Markdown (código, listas, bold)
- **Painel lateral** à direita — chunks RAG recuperados com módulo, título e score
  - Clique em qualquer chunk para expandir e ver o trecho completo
- **Perguntas rápidas** na tela inicial para começar a conversa
- **Limpar histórico** no rodapé do painel apaga o `historico_global.json`
- **Polling de status** — o input fica desabilitado até `_pronto=True` (carregamento em background)

---

## Caminhos — nada precisa ser copiado

Cada `assistente.py` resolve os caminhos automaticamente subindo a hierarquia
até encontrar o `shared/llm_factory.py`:

```
EAI_07_AI_Generative/                    ← detectado automaticamente
├── shared/llm_factory.py
├── data/
│   ├── cache/indice_rag.pkl             ← v1
│   └── chroma_db/                       ← v2 e v3 (banco compartilhado)
└── 06_Projetos_Reais/
    └── Assistente_Tecnico_IA/
        ├── v1/assistente.py             ← _HERE (sobe 3 níveis)
        ├── v2/assistente.py             ← _HERE (sobe 3 níveis)
        └── v3/assistente.py             ← _HERE (sobe 3 níveis)
```

---

## Reindexação

**v1 — rebuild completo:**
1. Delete `data/cache/indice_rag.pkl`
2. Reexecute `03_RAG/03_rag_basico.ipynb` (~67s)
3. Reinicie o Flask

**v2 / v3 — atualização incremental** (sem reconstruir tudo):
```python
# No 04_rag_avancado_chromadb.ipynb ou 04_rag_avancado_langchain.ipynb
atualizar_modulo(PROJETO_BASE, 'EAI_09')   # reindexará apenas o EAI_09 (~5s)
```

**v2 / v3 — rebuild completo:**
```python
resetar_banco(confirmar=True)   # apaga tudo e reconstrói (~110s)
```

---

## Integração com o curso

| Componente | v1 (FAISS) | v2 (ChromaDB) | v3 (LangChain) |
|---|---|---|---|
| Banco vetorial | `data/cache/indice_rag.pkl` | `data/chroma_db/` | `data/chroma_db/` (mesmo) |
| Gerado por | `03_rag_basico.ipynb` | `04_rag_avancado_chromadb.ipynb` | `04_rag_avancado_langchain.ipynb` (ou v2) |
| Chunks | 1.553 de 26 arquivos | 1.763 de 33 arquivos | 1.763 de 33 arquivos |
| Embedding | `SentenceTransformer` direto | `SentenceTransformer` direto | `HuggingFaceEmbeddings` (LangChain) |
| Query expansion | — | Sim (openai direto) | Sim (LCEL chain) |
| LLM | `openai.OpenAI()` direto | `openai.OpenAI()` direto | `ChatOpenAI` (LangChain) |
| Reindexação | Rebuild completo | Upsert incremental | Upsert incremental |
| Histórico | `historico_global.json` | `historico_global.json` | `historico_global.json` |

---

## Rotas da API

| Rota | Método | Descrição |
|---|---|---|
| `/` | GET | Interface de chat |
| `/chat` | POST | `{pergunta: str}` → `{resposta, fontes, historico, timestamp}` |
| `/limpar` | POST | Apaga o histórico global |
| `/status` | GET | Status do servidor e contagem de chunks do índice |

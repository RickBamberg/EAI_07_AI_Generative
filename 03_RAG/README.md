# 03_RAG — Retrieval Augmented Generation

**Submódulo de** EAI_07_AI_Generative  
**Ambiente:** `eai07` (Python 3.11)

---

## O que você vai aprender

RAG é a técnica de conectar um LLM a uma base de conhecimento externa via busca semântica.
Em vez de depender apenas do que o modelo aprendeu no treinamento, o RAG recupera
informação relevante em tempo real e injeta no prompt antes de gerar a resposta.

Este submódulo cobre o RAG do básico ao especializado em código, em 5 notebooks progressivos
com 2 variantes do notebook avançado (ChromaDB puro e LangChain):

| Notebook | Foco |
|---|---|
| `03_rag_basico.ipynb` | Pipeline completo: embedding → índice → busca → geração |
| `04_rag_avancado.ipynb` | Cache FAISS+pkl, filtro por módulo, query expansion, reranking, histórico |
| `04_rag_avancado_chromadb.ipynb` | Mesmas técnicas do 04, com ChromaDB como banco vetorial persistente |
| `04_rag_avancado_langchain.ipynb` | Mesmas técnicas do ChromaDB, reescrito com LangChain 1.x (LCEL) |
| `05_rag_codigo_especializado.ipynb` | AST chunking, índice híbrido código+docs, BM25, busca híbrida |

---

## Pré-requisitos

- Módulo `01_Modelos_Pre_Treinados` concluído (conceito de embeddings)
- Ambiente `eai07` ativo com as dependências instaladas
- Arquivo `.env` configurado na raiz do EAI_07 com `DEEPSEEK_API_KEY` e `LLM_MODEL`

---

## Instalação

```bash
conda activate eai07

# Dependências base (todos os notebooks)
pip install sentence-transformers faiss-cpu chromadb openai python-dotenv

# Dependências adicionais — apenas para 04_rag_avancado_langchain.ipynb
pip install langchain langchain-community langchain-chroma langchain-openai langchain-huggingface
```

Verificar instalação:

```python
from sentence_transformers import SentenceTransformer
import faiss, chromadb
model = SentenceTransformer('all-MiniLM-L6-v2')
print('OK')
```

> **Nota:** O modelo `all-MiniLM-L6-v2` (~90 MB) é baixado automaticamente na primeira execução
> e fica em cache local. Conexão com internet necessária na primeira vez.

---

## Corpus

Os notebooks indexam os próprios arquivos `AGENT_CONTEXT.md` do curso como base de conhecimento:

- **33 arquivos** coletados automaticamente de EAI_00 a EAI_08 (incluindo subpastas)
- **1.553 chunks** no índice FAISS (`03` e `04_rag_avancado`)
- **1.763 chunks** no ChromaDB (`04_chromadb` e `04_langchain`) — varre mais subpastas
- O assistente final consegue responder perguntas sobre qualquer módulo do curso

---

## Cache e persistência

Os índices são salvos em disco para evitar reindexação a cada sessão:

```
EAI_07_AI_Generative/
└── data/
    ├── cache/
    │   ├── indice_rag.pkl         ← FAISS — notebooks 03 e 04_rag_avancado (5.4 MB)
    │   └── indice_codigo.pkl      ← FAISS — RAG especializado em código
    └── chroma_db/                 ← ChromaDB — compartilhado entre 04_chromadb e 04_langchain
        ├── chroma.sqlite3
        └── <uuid>/
```

**FAISS + pkl** (`04_rag_avancado.ipynb`):
- Primeira execução: ~67s | Execuções seguintes: <1s
- Para reindexar: deletar o arquivo `.pkl` correspondente

**ChromaDB** (`04_rag_avancado_chromadb.ipynb` e `04_rag_avancado_langchain.ipynb`):
- Primeira execução: ~110s (1.763 chunks de 33 arquivos) | Execuções seguintes: <1s
- O banco `chroma_db/` é **compartilhado** entre os dois notebooks — se um indexou, o outro abre direto
- Suporta `upsert` incremental — atualiza apenas o módulo alterado sem reconstruir tudo
- Para reindexar tudo: `resetar_banco(confirmar=True)` | Para um módulo: `atualizar_modulo(base, 'EAI_07')`

---

## Arquitetura RAG

```
Pergunta do usuário
        │
        ▼
  [Query Expansion]        ← LLM reformula em termos técnicos (notebooks 04 e 05)
        │
        ▼
  [Busca Semântica]        ← FAISS (04) / ChromaDB (04_chromadb) / Chroma LangChain (04_langchain)
        │
  [+ BM25 léxico]          ← apenas no notebook 05 (busca híbrida)
        │
        ▼
  [Reranking]              ← LLM reordena chunks por relevância real (notebooks 04 e 05)
        │
        ▼
  [Geração com contexto]   ← chunks recuperados + histórico → resposta final
```

---

## Comparativo entre os notebooks avançados

Os três notebooks `04_*` implementam as mesmas técnicas (query expansion, reranking, histórico)
com diferentes stacks. Use a tabela para escolher qual estudar primeiro:

| Aspecto | `04_rag_avancado` | `04_rag_avancado_chromadb` | `04_rag_avancado_langchain` |
|---|---|---|---|
| Banco vetorial | FAISS em memória + pkl | ChromaDB persistente | ChromaDB (mesmo banco) |
| Embeddings | `SentenceTransformer.encode()` | `SentenceTransformer.encode()` | `HuggingFaceEmbeddings` (LangChain) |
| LLM | `openai.OpenAI()` direto | `openai.OpenAI()` direto | `ChatOpenAI` (LangChain) |
| Filtro por módulo | Reconstrói índice FAISS temporário | `where={"modulo_prefixo": {"$eq": ...}}` | `filter={"modulo_prefixo": {"$eq": ...}}` |
| Histórico | Lista manual `self.historico` | Lista manual `self.historico` | `ChatMessageHistory` + `RunnableWithMessageHistory` |
| Pipeline RAG | Código manual | Código manual | LCEL: `prompt \| llm \| StrOutputParser()` |
| Upsert incremental | Não (reconstrói tudo) | Sim (`collection.upsert()`) | Sim (`vectorstore.add_texts()` com IDs) |
| Chunks indexados | 1.553 | 1.763 | 1.763 (banco compartilhado) |
| Dependências extras | — | `chromadb` | + `langchain*` (5 pacotes) |

---

## LangChain 1.x — o que mudou

O notebook `04_rag_avancado_langchain.ipynb` usa LangChain na versão **1.x**, que introduziu
a LCEL (LangChain Expression Language) e removeu os módulos legados da versão 0.x.

Se você já conhece LangChain 0.x, preste atenção nessas mudanças:

| Módulo removido (0.x) | Substituto correto (1.x) |
|---|---|
| `langchain.memory.ConversationBufferWindowMemory` | `langchain_community.chat_message_histories.ChatMessageHistory` |
| `langchain.chains.ConversationalRetrievalChain` | LCEL: `prompt \| llm \| StrOutputParser()` |
| `langchain.prompts` | `langchain_core.prompts` |
| `langchain.schema` (mensagens) | `langchain_core.messages` |

A LCEL usa o operador `|` para encadear componentes — a chain abaixo lê como
"formata o prompt, envia ao LLM e converte a resposta para string":

```python
chain = ChatPromptTemplate.from_messages([...]) | ChatOpenAI(...) | StrOutputParser()
resposta = chain.invoke({'pergunta': '...', 'contexto': '...'})
```

O histórico de conversa é gerenciado automaticamente pelo `RunnableWithMessageHistory`,
que injeta as mensagens anteriores no placeholder do prompt antes de cada chamada.

---

## Conceitos-chave

**Embedding semântico**
Transforma texto em vetor numérico (384 dimensões). Textos com significado similar
ficam próximos no espaço vetorial, permitindo busca por similaridade conceitual.

**FAISS IndexFlatIP**
Índice de busca vetorial por produto interno. Com embeddings normalizados,
equivale a cosine similarity. Busca exata (não aproximada), ideal para corpora < 100k chunks.

**ChromaDB**
Banco vetorial persistente com suporte nativo a metadados e filtros. Armazena embeddings,
documentos e metadados juntos em SQLite + índice HNSW. Permite `upsert` incremental e
filtros como `where={"modulo_prefixo": {"$eq": "EAI_01"}}` sem recriar o índice.

**chunk_busca vs chunk_contexto**
Separação intencional: `chunk_busca` é curto e enriquecido com sinônimos para melhorar
o recall do embedding. `chunk_contexto` é o texto completo enviado ao LLM para gerar a resposta.

**Query Expansion com consciência de histórico**
Antes de buscar, o LLM reformula a pergunta em termos técnicos. Com histórico de conversa,
resolve referências anafóricas ("desse projeto", "nele") quando a pergunta é continuação
da anterior, mas ignora o histórico quando a pergunta introduz um novo tema — evitando
contaminação entre módulos diferentes.

**Reranking**
Após a busca vetorial, o LLM reordena os chunks recuperados por relevância real para a
pergunta original. Resolve falsos positivos: chunks com score de embedding alto mas
contexto errado voltam para baixo na lista.

**BM25**
Algoritmo clássico de busca léxica (keyword matching). Complementa o embedding semântico
para consultas com nomes exatos de funções, variáveis ou termos técnicos específicos.

**Score híbrido**
`score = α × semântico + (1-α) × BM25`
α=0.7 para uso geral. Reduzir α quando a query tem nomes exatos; aumentar para perguntas conceituais.

**Chunking por AST**
Para código Python, usa a árvore sintática (`ast` module) para extrair funções e classes
completas como unidades de chunk — evita cortes no meio de uma função.

**LCEL (LangChain Expression Language)**
Sintaxe do LangChain 1.x para compor pipelines: `componente_a | componente_b | componente_c`.
Cada componente recebe a saída do anterior. Substitui as chains monolíticas do LangChain 0.x.

---

## Referências

- [FAISS — Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [all-MiniLM-L6-v2 no Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [ChromaDB — documentação oficial](https://docs.trychroma.com/)
- [LangChain 1.x — documentação](https://python.langchain.com/docs/introduction/)
- [LCEL — LangChain Expression Language](https://python.langchain.com/docs/concepts/lcel/)
- [BM25 — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Python ast module](https://docs.python.org/3/library/ast.html)

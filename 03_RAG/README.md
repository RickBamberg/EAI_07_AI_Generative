# 03_RAG — Retrieval Augmented Generation

**Submódulo de** EAI_07_AI_Generative  
**Ambiente:** `eai07` (Python 3.11)

---

## O que você vai aprender

RAG é a técnica de conectar um LLM a uma base de conhecimento externa via busca semântica.
Em vez de depender apenas do que o modelo aprendeu no treinamento, o RAG recupera
informação relevante em tempo real e injeta no prompt antes de gerar a resposta.

Este submódulo cobre o RAG do básico ao especializado em código, em 5 notebooks progressivos:

| Notebook | Foco |
|---|---|
| `03_rag_basico.ipynb` | Pipeline completo: embedding → índice → busca → geração |
| `04_rag_avancado.ipynb` | Cache, filtro por módulo, query expansion, reranking, histórico |
| `05_rag_codigo_especializado.ipynb` | AST chunking, índice híbrido código+docs, BM25, busca híbrida |

---

## Pré-requisitos

- Módulo `01_Modelos_Pre_Treinados` concluído (conceito de embeddings)
- Ambiente `eai07` ativo com as dependências instaladas
- Arquivo `.env` configurado na raiz do EAI_07 com `DEEPSEEK_API_KEY` e `DEEPSEEK_MODEL`

---

## Instalação

O ambiente `eai07` já deve estar criado. Se precisar instalar as dependências deste submódulo:

```bash
conda activate eai07
pip install sentence-transformers faiss-cpu
```

Verificar instalação:

```python
from sentence_transformers import SentenceTransformer
import faiss
model = SentenceTransformer('all-MiniLM-L6-v2')
print('OK')
```

> **Nota:** O modelo `all-MiniLM-L6-v2` (~90 MB) é baixado automaticamente na primeira execução
> e fica em cache local. Conexão com internet necessária na primeira vez.

---

## Corpus

Os notebooks indexam os próprios arquivos `AGENT_CONTEXT.md` do curso como base de conhecimento:

- **26 arquivos** coletados automaticamente de EAI_01 a EAI_08
- **1.553 chunks** gerados após chunking por seção
- O assistente final consegue responder perguntas sobre qualquer módulo do curso

---

## Cache

Os índices FAISS são salvos em disco para evitar reindexação a cada sessão:

```
EAI_07_AI_Generative/
└── data/
    └── cache/
        ├── indice_rag.pkl         ← RAG básico e avançado (5.4 MB)
        └── indice_codigo.pkl      ← RAG especializado em código
```

- **Primeira execução:** ~67 segundos para gerar embeddings
- **Execuções seguintes:** <1 segundo carregando do cache

Para forçar reindexação, basta deletar o arquivo `.pkl` correspondente.

---

## Arquitetura RAG

```
Pergunta do usuário
        │
        ▼
  [Query Expansion]        ← LLM reformula em termos técnicos (notebook 04)
        │
        ▼
  [Busca Híbrida]          ← semântico (FAISS) + léxico (BM25) com peso α
        │
        ▼
  [Reranking]              ← LLM reordena chunks por relevância real (notebook 04)
        │
        ▼
  [Geração com contexto]   ← chunks recuperados + histórico → resposta final
```

---

## Conceitos-chave

**Embedding semântico**  
Transforma texto em vetor numérico (384 dimensões). Textos com significado similar
ficam próximos no espaço vetorial, permitindo busca por similaridade conceitual.

**FAISS IndexFlatIP**  
Índice de busca vetorial por produto interno. Com embeddings normalizados,
equivale a cosine similarity. Busca exata (não aproximada), ideal para corpora < 100k chunks.

**chunk_busca vs chunk_contexto**  
Separação intencional: `chunk_busca` é curto e enriquecido com sinônimos para melhorar
o recall do embedding. `chunk_contexto` é o texto completo enviado ao LLM para gerar a resposta.

**BM25**  
Algoritmo clássico de busca léxica (keyword matching). Complementa o embedding semântico
para consultas com nomes exatos de funções, variáveis ou termos técnicos específicos.

**Score híbrido**  
`score = α × semântico + (1-α) × BM25`  
α=0.7 para uso geral. Reduzir α quando a query tem nomes exatos; aumentar para perguntas conceituais.

**Chunking por AST**  
Para código Python, usa a árvore sintática (`ast` module) para extrair funções e classes
completas como unidades de chunk — evita cortes no meio de uma função.

---

## Referências

- [FAISS — Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [all-MiniLM-L6-v2 no Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [BM25 — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Python ast module](https://docs.python.org/3/library/ast.html)

# AGENT_CONTEXT — 03_RAG
# Submódulo do EAI_07_AI_Generative
# Para uso por agentes de IA. Versão legível: README.md

## IDENTIFICAÇÃO
- Submódulo: 03_RAG
- Módulo pai: EAI_07_AI_Generative
- Ambiente: eai07 (Python 3.11, conda)
- Dependências base: sentence-transformers, faiss-cpu, chromadb, openai, python-dotenv, numpy
- Dependências LangChain: langchain, langchain-community, langchain-chroma, langchain-openai, langchain-huggingface

## VISÃO GERAL
Implementação progressiva de RAG (Retrieval Augmented Generation) em 5 notebooks (+2 variantes),
do básico ao especializado em código. O corpus indexado são os próprios AGENT_CONTEXT.md
do curso (EAI_00 a EAI_08), totalizando 1.553 chunks (FAISS) ou 1.763 chunks (ChromaDB)
de 33 arquivos distribuídos em subpastas.

## NOTEBOOKS

### 03_rag_basico.ipynb
Fundamentos do pipeline RAG completo.
- Embedding: all-MiniLM-L6-v2 (384 dimensões, sentence-transformers)
- Índice: faiss.IndexFlatIP com embeddings L2-normalizados (cosine similarity)
- Chunking: por seções `###` dos AGENT_CONTEXT.md
- Enriquecimento: dicionário de sinônimos técnicos adicionado ao chunk_busca
- Separação chunk_busca vs chunk_contexto: busca usa texto curto enriquecido; LLM recebe texto completo
- Provider: DeepSeek via openai-compatible client
- Resultado: 1.553 chunks indexados, ~30s para gerar embeddings

### 04_rag_avancado.ipynb
Técnicas avançadas sobre o índice do notebook 03.

#### Cache em disco
- Serialização FAISS: `faiss.serialize_index()` / `faiss.deserialize_index()`
- Cache completo em pickle: faiss_bytes + chunks + embs
- Resultado: recarregamento em <1s vs ~30s para reconstruir
- Arquivo: `../data/cache/indice_rag.pkl` (5.4 MB)

#### Filtro por módulo
- Cria índice FAISS temporário só com embeddings do módulo filtrado
- Preserva os índices originais para mapear de volta para os chunks
- Parâmetro: `filtro_modulo='EAI_01'` (prefixo)

#### Query Expansion
- LLM reformula a pergunta em termos técnicos antes de buscar
- Exemplo: "ajustar uma linha" → "regressão linear, mínimos quadrados, gradiente descendente"
- Melhora recall de 0.51 para 0.62 no teste de regressão linear
- Prompt: instrução para gerar só a query expandida, sem explicação

#### Reranking
- Após busca vetorial, LLM reordena chunks por relevância real
- Resolve falsos positivos do embedding (score alto mas contexto errado)
- Prompt: lista numerada de chunks, pede ranking como JSON

#### AssistenteRAG com histórico
- Mantém `self.historico` como lista de messages (role/content)
- Cada resposta usa os chunks recuperados + histórico completo
- System prompt: identidade do projeto + lista de módulos disponíveis

### 04_rag_avancado_chromadb.ipynb
Variante do 04_rag_avancado usando ChromaDB como banco vetorial persistente.
Mantém as mesmas técnicas (query expansion, reranking, histórico) mas substitui FAISS+pkl pelo ChromaDB.

#### Diferenças em relação ao 04_rag_avancado.ipynb
| Aspecto | FAISS + pkl | ChromaDB |
|---|---|---|
| Persistência | `pickle.dump` / `pickle.load` | `PersistentClient` automático |
| Filtro por módulo | Recria `IndexFlatIP` temporário | `where={"modulo_prefixo": {"$eq": "EAI_01"}}` |
| Atualizar 1 módulo | Reconstrói tudo | `collection.upsert()` parcial |
| IDs dos chunks | `modulo__chunk_NNNN` (colisão em subpastas) | `arquivo_slug__chunk_NNNN` (único por arquivo) |
| Espaço de distância | Inner Product (com normalize) | Cosine nativo (`hnsw:space: cosine`) |
| Score retornado | Similaridade direta | `1 - (distancia / 2)` |

#### Banco vetorial ChromaDB
- Cliente: `chromadb.PersistentClient(path='../data/chroma_db')`
- Collection: `agent_contexts` com `hnsw:space: cosine`
- `get_or_create_collection`: na primeira execução cria e indexa; nas seguintes apenas abre
- Diretório: `data/chroma_db/` (chroma.sqlite3 + índice HNSW binário em subdiretório uuid)

#### IDs únicos por arquivo (crítico para projetos com subpastas)
- Módulos com múltiplos AGENT_CONTEXT.md em subpastas causam colisão se o ID usar só o módulo
- EAI_02: 4 subpastas | EAI_03: 6 subpastas | EAI_04: 5 subpastas | EAI_07: 8 subpastas
- Solução: ID baseado no caminho relativo completo do arquivo convertido em slug
- Exemplo: `EAI_03_Deep_Learning__Projetos_Reais__Arte__AGENT_CONTEXT_md__chunk_0012`
- Resultado: 1.763 chunks sem colisões (vs 642 com IDs por módulo)

#### Metadados armazenados por chunk
```python
{
    'modulo'         : 'EAI_01_Fundamentos_Matemática_para_IA',  # nome completo do diretório EAI_XX
    'modulo_prefixo' : 'EAI_01',   # usado nos filtros where ($eq)
    'titulo'         : 'Regressão linear manual',
    'arquivo'        : 'EAI_01_Fundamentos_Matemática_para_IA/AGENT_CONTEXT.md',
}
```

#### Query Expansion com consciência de histórico
- `expandir_query(pergunta, historico=None)` — aceita as últimas trocas da conversa
- O LLM decide via prompt se o histórico é relevante (CONTINUAÇÃO) ou não (INDEPENDENTE)
- CONTINUAÇÃO: resolve pronomes como "desse projeto", "nele", "qual foi a acurácia"
- INDEPENDENTE: ignora histórico e expande só com termos técnicos (evita contaminação entre módulos)
- O `AssistenteRAG.responder()` passa automaticamente `self.historico[-4:]` para cada expansão

#### Operadores WHERE suportados pelo ChromaDB em metadados
- Suportados: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`
- NÃO suportado em metadados: `$contains` (existe apenas para `where_document`)

#### Utilitários extras
```python
inspecionar_colecao()                    # exibe chunks por módulo
atualizar_modulo(base, 'EAI_07')         # upsert incremental de um módulo
resetar_banco(confirmar=True)            # apaga collection e reconstrói do zero
```

#### Resultados obtidos (execução real)
- Banco: 1.763 chunks de 33 AGENT_CONTEXT.md (sem colisões)
- Indexação: ~110s | Recarga: <1s
- Busca "regressão linear" sem filtro: score 0.766, EAI_01 correto
- Busca com filtro EAI_01: apenas chunks do EAI_01, score 0.763
- Query expansion "ajustar uma linha" → EAI_07 (score 0.765, errado) vs expandida → EAI_01 (score 0.766, correto)
- Histórico multi-turn: acurácia do projeto de obras de arte (65%) respondida corretamente na segunda pergunta

### 04_rag_avancado_langchain.ipynb
Variante do 04_rag_avancado_chromadb reescrita com LangChain 1.x (LCEL).
O banco ChromaDB em disco é **o mesmo** — só a camada de código Python muda.
Objetivo didático: mostrar como o LangChain abstrai vectorstore, embeddings, LLM e histórico.

#### Versão do LangChain e breaking changes
- Versão utilizada: LangChain **1.x** (LCEL — LangChain Expression Language)
- LangChain 1.x removeu os módulos legados da versão 0.x:

| Removido (0.x) | Substituto correto (1.x) |
|---|---|
| `langchain.memory.ConversationBufferWindowMemory` | `langchain_community.chat_message_histories.ChatMessageHistory` |
| `langchain.chains.ConversationalRetrievalChain` | LCEL: `prompt \| llm \| StrOutputParser()` |
| `langchain.prompts` | `langchain_core.prompts` |
| `langchain.schema` (mensagens) | `langchain_core.messages` |

#### Imports corretos para LangChain 1.x
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
```

#### Equivalências entre ChromaDB puro e LangChain
| Componente | ChromaDB puro | LangChain 1.x |
|---|---|---|
| Vectorstore | `chromadb.PersistentClient` + `collection` | `Chroma(persist_directory=...)` |
| Embeddings | `SentenceTransformer.encode()` + normalize manual | `HuggingFaceEmbeddings(encode_kwargs={'normalize_embeddings': True})` |
| Busca | `collection.query(query_embeddings=...)` | `vectorstore.similarity_search_with_score(query, k=5, filter=...)` |
| Filtro módulo | `where={'modulo_prefixo': {'$eq': 'EAI_01'}}` | `filter={'modulo_prefixo': {'$eq': 'EAI_01'}}` (mesma sintaxe) |
| LLM | `openai.OpenAI().chat.completions.create()` | `ChatOpenAI.invoke()` |
| Pipeline | loop manual | LCEL: `prompt \| llm \| StrOutputParser()` |
| Histórico | lista `self.historico` manual | `ChatMessageHistory` + `RunnableWithMessageHistory` |

#### Como o banco é compartilhado entre os notebooks
- `Chroma(persist_directory='../data/chroma_db', collection_name='agent_contexts')` abre o mesmo banco
- O mesmo diretório `chroma_db/` funciona para `04_rag_avancado_chromadb.ipynb` e `04_rag_avancado_langchain.ipynb`
- Se o banco já estiver populado, `add_texts()` com IDs fixos funciona como upsert — não duplica

#### Indexação via LangChain (add_texts)
- `vectorstore.add_texts(texts, metadatas, ids)` substitui `collection.upsert()`
- LangChain gera os embeddings internamente via `embedding_function`
- O campo `_contexto` é armazenado como metadado extra (texto completo para o LLM)
- `doc.metadata.get('_contexto', doc.page_content)` recupera o contexto na busca
```python
vectorstore.add_texts(
    texts     = [c['chunk_busca'] for c in chunks],
    metadatas = [{'modulo': ..., 'modulo_prefixo': ..., 'titulo': ...,
                  'arquivo': ..., '_contexto': c['chunk_contexto']} for c in chunks],
    ids       = [f"{arquivo_slug}__chunk_{i:04d}" for i in range(len(chunks))],
)
```

#### Busca via LangChain
- `similarity_search_with_score()` retorna `list[tuple[Document, float]]`
- `Document.page_content` = texto de busca (chunk_busca)
- `Document.metadata` = metadados incluindo `_contexto`
- Distância retornada é cosine ChromaDB (0=idêntico) → `score = 1 - (dist / 2)`
- Filtro por módulo: `vectorstore.similarity_search_with_score(query, k=5, filter={'modulo_prefixo': {'$eq': 'EAI_01'}})`

#### LCEL — construção de chains
- Operador `|` encadeia componentes: `chain = prompt | llm | StrOutputParser()`
- `.invoke(dict)` executa a chain com as variáveis do template
- Substitui chamadas manuais a `llm.chat.completions.create()`
```python
# Query Expansion
expansion_chain = PromptTemplate.from_template("...{pergunta}...") | llm | StrOutputParser()
resultado = expansion_chain.invoke({'pergunta': pergunta, 'contexto_historico': ctx})

# Reranking
rerank_chain = PromptTemplate.from_template("...{query}...{chunks_texto}...") | llm | StrOutputParser()
ordem = rerank_chain.invoke({'query': query, 'chunks_texto': chunks_texto})
```

#### Histórico com RunnableWithMessageHistory
- `ChatMessageHistory` armazena mensagens (substitui lista manual)
- `RunnableWithMessageHistory` injeta o histórico automaticamente no prompt via `MessagesPlaceholder`
- Cada sessão tem um `session_id` — múltiplas sessões paralelas possível
- `input_messages_key='pergunta'` e `history_messages_key='historico'` mapeiam as variáveis do prompt
```python
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT + '\n\nContexto:\n{contexto}'),
    MessagesPlaceholder(variable_name='historico'),
    ('human', '{pergunta}'),
])
rag_chain = rag_prompt | llm | StrOutputParser()
chain_com_hist = RunnableWithMessageHistory(
    rag_chain,
    get_historico,           # função que retorna ChatMessageHistory dado session_id
    input_messages_key='pergunta',
    history_messages_key='historico',
)
resposta = chain_com_hist.invoke(
    {'pergunta': pergunta, 'contexto': contexto},
    config={'configurable': {'session_id': 'sessao_01'}},
)
```

#### Duas opções de pipeline incluídas no notebook
**Opção A — `responder_simples()` (função com session_id)**
- Interface funcional: `responder_simples(pergunta, session_id='default')`
- Inclui query expansion + reranking + histórico por session_id
- Histórico bruto por `session_id` em `_historicos_raw` para alimentar `expandir_query()`
- Usa `RunnableWithMessageHistory` com dicionário global `historicos`

**Opção B — `AssistenteRAG` (classe)**
- Mesma interface do notebook ChromaDB puro
- Histórico encapsulado: `self._chat_history` (ChatMessageHistory) + `self._historico_raw` (lista bruta)
- Parâmetros: `filtro_modulo`, `usar_expansion`, `usar_reranking`, `verbose`
- `limpar_historico()` reseta ambos os históricos

#### Causa raiz do erro "resultados errados" na Opção A (resolvido)
- **Problema**: versão inicial de `responder_simples()` usava a pergunta literal na busca, sem `expandir_query()`
- **Sintoma**: pergunta "Qual projeto de deep learning classificou obras de arte?" retornava chunks de MNIST
- **Solução**: `responder_simples()` passou a chamar `expandir_query()` e `rerankar()` antes de invocar a chain
- **Lição**: query expansion é essencial — busca vetorial literal falha em perguntas não-técnicas

#### Resultados obtidos (execução real)
- Banco: mesmo 1.763 chunks do 04_rag_avancado_chromadb (banco compartilhado, sem reindexação)
- Opção A corrigida: "Qual projeto de DL classificou obras de arte?" → resposta correta com MobileNetV2
- Opção B: resultados idênticos ao notebook ChromaDB puro
- Filtro EAI_01: query busca expandida para "mínimos quadrados / regressão linear" → score 0.773, chunks corretos

### 05_rag_codigo_especializado.ipynb
RAG especializado para código-fonte, com índice híbrido.

#### Chunking por AST
- Usa módulo `ast` nativo do Python (sem dependências extras)
- Extrai FunctionDef, AsyncFunctionDef, ClassDef como unidades completas
- Metadados por chunk: tipo_chunk, assinatura, docstring, arquivo, linhas (ini, fim)
- Fallback: arquivo inteiro como chunk de módulo se não há funções/classes
- Erros de sintaxe: capturados com try/except SyntaxError, arquivo ignorado

#### Índice Híbrido
- Mesmo FAISS recebe dois tipos: `tipo='codigo'` (AST de .py) e `tipo='doc'` (AGENT_CONTEXT.md)
- Corpus código: todos os .py do EAI_07_AI_Generative (shared/, notebooks)
- Corpus docs: todos os AGENT_CONTEXT.md do projeto (EAI_01 a EAI_08)
- Cache separado: `../data/cache/indice_codigo.pkl`
- Filtro pós-busca por `tipo` sem reindexar

#### BM25 (implementação from scratch)
- Sem dependências: usa apenas re, math, Counter
- Parâmetros: k1=1.5, b=0.75 (padrão da literatura)
- Tokenização: regex `[a-z\w]+` lowercase
- IDF: `log((N - df + 0.5) / (df + 0.5) + 1)`
- Construído sobre os chunk_busca (mesmo corpus do FAISS)

#### Busca Híbrida
- Score final: `α × score_semântico + (1-α) × score_bm25`
- Ambos os scores normalizados para [0, 1] antes de combinar
- α=0.7 recomendado para código (padrão)
- Guia de α: nomes exatos → 0.4-0.5 | conceitos → 0.8-1.0 | geral → 0.6-0.7

#### AssistenteCodigoRAG
- Herda padrão do AssistenteRAG: historico, responder(), resetar()
- Parâmetros: alfa (peso híbrido) e top_k configuráveis por instância
- filtro_tipo opcional: 'codigo' | 'doc' | None

## ARQUITETURA SHARED (usada por todos os notebooks)

### llm_factory.py
- Função: `chat(messages, system)` → str
- Função: `chat_stream(messages, system)` → generator
- Função: `get_provider_info()` → dict com provider, model, base_url
- Provider controlado por `.env`: DEEPSEEK_API_KEY, LLM_MODEL
- Client: openai.OpenAI com base_url='https://api.deepseek.com'

### Estrutura de chunk (padrão unificado nos notebooks)
```python
{
    'chunk_busca'   : str,   # texto curto enriquecido, usado para embedding/BM25
    'chunk_contexto': str,   # texto completo, enviado ao LLM como contexto
    'titulo'        : str,   # título da seção ou nome da função/classe
    'modulo'        : str,   # ex: 'EAI_01_Fundamentos_Matemática_para_IA'
    'modulo_prefixo': str,   # ex: 'EAI_01' — usado nos filtros ChromaDB / LangChain
    'tipo'          : str,   # 'doc' | 'codigo'
    'tipo_chunk'    : str,   # 'seção' | 'função' | 'classe' | 'módulo'
    'arquivo'       : str,   # caminho relativo ao PROJETO_BASE
    'assinatura'    : str,   # linha def/class (código) ou '' (doc)
    'docstring'     : str,   # docstring extraída (código) ou '' (doc)
    'linhas'        : tuple, # (linha_ini, linha_fim) no arquivo fonte
}
```

## DADOS E CACHE
- Corpus FAISS: 26 AGENT_CONTEXT.md → 1.553 chunks | `data/cache/indice_rag.pkl` (5.4 MB)
- Corpus ChromaDB / LangChain: 33 AGENT_CONTEXT.md → 1.763 chunks | `data/chroma_db/`
- Cache RAG código: `data/cache/indice_codigo.pkl`
- Tempo de reconstrução FAISS: ~67s | ChromaDB: ~110s | Recarga de cache: <1s
- Modelo embedding: all-MiniLM-L6-v2, 384 dim, normalizado

## PADRÕES DE CÓDIGO

### ChromaDB puro (04_rag_avancado_chromadb.ipynb)
```python
# Inicialização
modelo_emb = SentenceTransformer('all-MiniLM-L6-v2')
llm = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com')
chroma_client = chromadb.PersistentClient(path='../data/chroma_db')
collection = chroma_client.get_or_create_collection('agent_contexts', metadata={'hnsw:space': 'cosine'})

# Busca com filtro
resultados = collection.query(
    query_embeddings=emb_q.tolist(),
    n_results=top_k,
    where={'modulo_prefixo': {'$eq': 'EAI_01'}},
    include=['documents', 'metadatas', 'distances']
)
```

### LangChain 1.x (04_rag_avancado_langchain.ipynb)
```python
# Inicialização
embedding_function = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    encode_kwargs={'normalize_embeddings': True}
)
llm = ChatOpenAI(model='deepseek-chat', api_key=..., base_url='https://api.deepseek.com')
vectorstore = Chroma(
    collection_name='agent_contexts',
    embedding_function=embedding_function,
    persist_directory='../data/chroma_db',
    collection_metadata={'hnsw:space': 'cosine'},
)

# Busca com filtro
pares = vectorstore.similarity_search_with_score(
    query, k=top_k, filter={'modulo_prefixo': {'$eq': 'EAI_01'}}
)
for doc, dist in pares:
    score = 1.0 - (dist / 2.0)
    contexto = doc.metadata.get('_contexto', doc.page_content)

# LCEL chain
chain = PromptTemplate.from_template("...{var}...") | llm | StrOutputParser()
resultado = chain.invoke({'var': valor})

# Histórico automático
chain_hist = RunnableWithMessageHistory(chain, get_historico,
    input_messages_key='pergunta', history_messages_key='historico')
resposta = chain_hist.invoke({'pergunta': p, 'contexto': ctx},
    config={'configurable': {'session_id': 'sid'}})
```

### FAISS (03_rag_basico / 04_rag_avancado)
```python
# Busca semântica FAISS
emb_q = modelo_emb.encode([query], normalize_embeddings=True).astype(np.float32)
scores, pos = indice_faiss.search(emb_q, top_k)

# Busca híbrida (notebook 05)
resultados = buscar_hibrido(query, top_k=5, alfa=0.7, filtro_tipo='codigo')
```

## FAQ
Q: Por que IndexFlatIP e não IndexFlatL2?
A: Com embeddings normalizados, produto interno (IP) = cosine similarity. IndexFlatIP é mais rápido e o resultado é equivalente.

Q: Por que chunk_busca separado do chunk_contexto?
A: chunk_busca é curto e enriquecido para melhorar recall do embedding. chunk_contexto é completo para o LLM ter contexto suficiente para responder.

Q: Por que BM25 from scratch e não biblioteca?
A: Evita adicionar dependência (rank_bm25, elasticsearch) ao ambiente eai07. A implementação cobre o caso de uso completo com ~50 linhas.

Q: Qual α usar na busca híbrida?
A: 0.7 para uso geral em código. Reduzir para 0.4-0.5 quando a query tem nomes exatos de funções/variáveis. Aumentar para 0.9+ para perguntas conceituais.

Q: Os notebooks são independentes entre si?
A: Cada notebook copia as funções necessárias do anterior para ser executável isoladamente. O cache em disco é o único estado compartilhado entre sessões.

Q: Qual a diferença entre 04_rag_avancado_chromadb e 04_rag_avancado_langchain?
A: Mesmas técnicas e mesmo banco ChromaDB em disco. A diferença é só na camada de código: o notebook LangChain usa abstrações do framework (Chroma, HuggingFaceEmbeddings, ChatOpenAI, LCEL chains, RunnableWithMessageHistory) em vez de chamar ChromaDB e OpenAI diretamente. O objetivo é didático — mostrar como o LangChain unifica diferentes backends com a mesma interface.

Q: O banco chroma_db é compartilhado entre 04_chromadb e 04_langchain?
A: Sim. Ambos apontam para `../data/chroma_db` com `collection_name='agent_contexts'`. Se um já populou o banco, o outro abre sem reindexar. `add_texts()` com IDs fixos faz upsert, não duplica.

Q: Por que usar LangChain 1.x em vez de 0.x?
A: LangChain 0.x está descontinuado. Na versão 1.x, `langchain.memory` e `langchain.chains` foram removidos. Os substitutos são `ChatMessageHistory` + `RunnableWithMessageHistory` para histórico e LCEL (`prompt | llm | parser`) para pipelines. Todo código novo deve usar 1.x.

Q: Por que a Opção A do notebook LangChain dava resultados errados inicialmente?
A: A versão inicial de `responder_simples()` buscava com a pergunta literal, sem passar por `expandir_query()`. A busca vetorial falha em perguntas coloquiais ("qual projeto classificou obras de arte?") porque o embedding aproxima a frase de chunks genéricos de ML, não do chunk específico sobre o projeto de pinturas. Após adicionar expansion + reranking, os resultados ficaram idênticos à Opção B.

Q: Qual a diferença entre a Opção A e B dentro do notebook LangChain?
A: Estrutural apenas. Opção A é uma função `responder_simples(pergunta, session_id)` com histórico por session_id em dicionário global. Opção B é a classe `AssistenteRAG` com histórico encapsulado na instância. Ambas usam os mesmos componentes LCEL, query expansion e reranking.

Q: Qual a diferença entre 04_rag_avancado (FAISS) e 04_rag_avancado_chromadb?
A: Mesmas técnicas (query expansion, reranking, histórico), banco diferente. FAISS+pkl é mais simples e leve. ChromaDB oferece persistência automática, filtros nativos por metadado, upsert incremental e não sofre colisão de IDs em projetos com subpastas.

Q: Por que o ChromaDB tem 1.763 chunks e o FAISS tem 1.553?
A: O FAISS original foi indexado com 26 arquivos. O ChromaDB varre recursivamente todas as subpastas e encontra 33 arquivos. A diferença são os AGENT_CONTEXT.md em subpastas de EAI_07 adicionados depois da indexação original.

Q: O filtro $contains funciona em metadados no ChromaDB?
A: Não. `$contains` só funciona em `where_document` (texto do documento). Para metadados, usar `$eq` com o campo `modulo_prefixo` (ex: `{'modulo_prefixo': {'$eq': 'EAI_01'}}`). Isso vale tanto para ChromaDB puro quanto para o wrapper LangChain — a sintaxe é idêntica.

Q: Como a query expansion evita contaminar a busca ao trocar de módulo?
A: O prompt tem duas regras explícitas: CONTINUAÇÃO (usa histórico para resolver pronomes) e INDEPENDENTE (ignora histórico quando a pergunta menciona explicitamente novo módulo/tema). O LLM decide qual aplicar.

Q: Como limitar o histórico no AssistenteRAG LangChain para não crescer indefinidamente?
A: `self._chat_history.messages = self._chat_history.messages[-(max_historico * 2):]` após cada resposta. O `_historico_raw` também é acessado com slice `[-4:]` no expandir_query, nunca completo.

## TAGS DE BUSCA
RAG retrieval augmented generation busca semântica FAISS ChromaDB LangChain LCEL embeddings
sentence-transformers HuggingFaceEmbeddings ChatOpenAI all-MiniLM-L6-v2 IndexFlatIP cosine similarity
chunking AST BM25 busca híbrida query expansion reranking cache pickle filtro módulo histórico conversa
ChromaDB PersistentClient upsert incremental modulo_prefixo where metadados filter
RunnableWithMessageHistory ChatMessageHistory MessagesPlaceholder StrOutputParser
PromptTemplate ChatPromptTemplate LCEL pipe operator chain invoke session_id
langchain_core langchain_community langchain_chroma langchain_openai langchain_huggingface
breaking changes migração 0.x 1.x ConversationalRetrievalChain ConversationBufferWindowMemory removido
llm_factory chat_stream DeepSeek chunk_busca chunk_contexto enriquecimento sinônimos
função classe docstring assinatura código Python índice vetorial colisão IDs subpastas

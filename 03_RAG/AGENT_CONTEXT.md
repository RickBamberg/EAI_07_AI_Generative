# AGENT_CONTEXT — 03_RAG
# Submódulo do EAI_07_AI_Generative
# Para uso por agentes de IA. Versão legível: README.md

## IDENTIFICAÇÃO
- Submódulo: 03_RAG
- Módulo pai: EAI_07_AI_Generative
- Ambiente: eai07 (Python 3.11, conda)
- Dependências: sentence-transformers, faiss-cpu, openai, python-dotenv, numpy

## VISÃO GERAL
Implementação progressiva de RAG (Retrieval Augmented Generation) em 5 notebooks,
do básico ao especializado em código. O corpus indexado são os próprios AGENT_CONTEXT.md
do curso (EAI_01 a EAI_08), totalizando 1.553 chunks de 26 arquivos.

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

### Estrutura de chunk (padrão unificado nos 5 notebooks)
```python
{
    'chunk_busca'   : str,   # texto curto enriquecido, usado para embedding/BM25
    'chunk_contexto': str,   # texto completo, enviado ao LLM como contexto
    'titulo'        : str,   # título da seção ou nome da função/classe
    'modulo'        : str,   # ex: 'EAI_01_Fundamentos_Matemática_para_IA'
    'tipo'          : str,   # 'doc' | 'codigo'
    'tipo_chunk'    : str,   # 'seção' | 'função' | 'classe' | 'módulo'
    'arquivo'       : str,   # caminho relativo ao PROJETO_BASE
    'assinatura'    : str,   # linha def/class (código) ou '' (doc)
    'docstring'     : str,   # docstring extraída (código) ou '' (doc)
    'linhas'        : tuple, # (linha_ini, linha_fim) no arquivo fonte
}
```

## DADOS E CACHE
- Corpus principal: 26 AGENT_CONTEXT.md → 1.553 chunks
- Cache RAG básico: `data/cache/indice_rag.pkl` (5.4 MB)
- Cache RAG código: `data/cache/indice_codigo.pkl`
- Tempo de reconstrução: ~67s (embeddings) | <1s (cache)
- Modelo embedding: all-MiniLM-L6-v2, 384 dim, normalizado

## PADRÕES DE CÓDIGO
```python
# Inicialização padrão (todos os notebooks)
modelo_emb = SentenceTransformer('all-MiniLM-L6-v2')
llm = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com')
LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-chat')
PROJETO_BASE = os.path.abspath('../..')  # raiz do curso
EAI07_BASE   = os.path.abspath('..')    # raiz do EAI_07

# Busca semântica básica
emb_q = modelo_emb.encode([query], normalize_embeddings=True).astype(np.float32)
scores, pos = indice_faiss.search(emb_q, top_k)

# Busca híbrida
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

## TAGS DE BUSCA
RAG retrieval augmented generation busca semântica FAISS embeddings sentence-transformers
all-MiniLM-L6-v2 IndexFlatIP cosine similarity chunking AST BM25 busca híbrida
query expansion reranking cache pickle filtro módulo histórico conversa
llm_factory chat_stream DeepSeek chunk_busca chunk_contexto enriquecimento sinônimos
função classe docstring assinatura código Python índice vetorial

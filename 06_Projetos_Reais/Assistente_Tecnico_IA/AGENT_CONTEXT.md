# AGENT_CONTEXT — Assistente_Tecnico_IA
# Projeto do EAI_07_AI_Generative / 06_Projetos_Reais
# Para uso por agentes de IA. Versão legível: README.md

## IDENTIFICAÇÃO
- Projeto: Assistente_Tecnico_IA
- Módulo pai: EAI_07_AI_Generative / 06_Projetos_Reais
- Ambiente: eai07 (Python 3.11, conda) ou Docker (python:3.11-slim)
- Dependências base: flask, fastapi, uvicorn, openai, python-dotenv, sentence-transformers, numpy
- Dependências v1: faiss-cpu
- Dependências v2: chromadb
- Dependências v3: langchain, langchain-community, langchain-chroma, langchain-openai, langchain-huggingface

## VISÃO GERAL
Aplicação com interface de chat que responde perguntas sobre os módulos EAI_01 a EAI_08
usando RAG semântico sobre os AGENT_CONTEXT.md do curso.
Projeto integrador — usa o shared/llm_factory do EAI_07 e padrões de memória do 05_Agentes.

Dois servidores disponíveis com a mesma lógica de negócio:
- app.py: Flask (porta 5000) — interface HTML com chat e painel de fontes
- api.py: FastAPI (porta 8000) — REST JSON com Swagger em /docs

Três versões de backend RAG:
- v1/assistente.py: índice FAISS + pkl (1.553 chunks, 26 arquivos)
- v2/assistente.py: banco ChromaDB persistente (1.763 chunks, 33 arquivos, upsert incremental)
- v3/assistente.py: banco ChromaDB via LangChain 1.x / LCEL (mesmo banco da v2)

Suporte a Docker via Dockerfile + docker-compose.yml com profiles separados para Flask e FastAPI.

## ARQUIVOS

### app.py  (Flask — porta 5000)
Servidor Flask com interface HTML. Importa o assistente pela linha ativa.
Rotas:
- GET  /        → serve templates/index.html
- POST /chat    → recebe {pergunta: str}, retorna {resposta, fontes, historico, timestamp}
- POST /limpar  → apaga data/historico_global.json
- GET  /status  → retorna {status, pronto, rag, chunks}

### api.py  (FastAPI — porta 8000)
Servidor FastAPI com REST JSON e documentação automática. Importa o mesmo assistente que o Flask.
Importa exclusivamente da v3: `from v3.assistente import responder, limpar_historico, _get_indice, _pronto`
Rotas:
- GET  /        → redireciona para /docs (RedirectResponse)
- POST /chat    → valida com PerguntaRequest (Pydantic), retorna ChatResponse
- POST /limpar  → retorna LimparResponse
- GET  /status  → retorna StatusResponse
- GET  /docs    → Swagger UI gerado automaticamente pelo FastAPI
- GET  /redoc   → documentação ReDoc

Schemas Pydantic:
```python
class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., min_length=1)

class FonteSchema(BaseModel):
    modulo: str
    titulo: str
    score: float
    arquivo: str
    trecho: str

class ChatResponse(BaseModel):
    resposta: str
    fontes: list[FonteSchema]
    historico: list[dict]
    timestamp: str

class LimparResponse(BaseModel):
    status: str
    mensagem: str

class StatusResponse(BaseModel):
    status: str
    pronto: bool
    rag: str
    chunks: int
```

Rota /status no FastAPI:
```python
@app.get("/status", response_model=StatusResponse)
def status():
    indice = _get_indice() if _pronto else None
    return {
        "status": "ok",
        "pronto": _pronto,
        "rag"   : "carregado" if indice else "carregando...",
        "chunks": indice._collection.count() if indice else 0,
    }
```

Execução local:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# ou: python api.py  (entrypoint __main__ com uvicorn.run embutido)
```

Diferenças FastAPI vs Flask:
- Validação automática via Pydantic — payload inválido retorna 422 com detalhe do campo
- HTTPException padronizado para erros 500
- async/await nativo na rota /chat
- Documentação interativa em /docs sem esforço adicional
- Sem interface HTML — ideal para integração com outros sistemas ou frontend separado

### Dockerfile
Imagem única usada para Flask e FastAPI — o comando é sobrescrito pelo docker-compose.
```dockerfile
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential libglib2.0-0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]   # default Flask; sobrescrito pelo compose
```
- Base: `python:3.11-slim` (imagem mínima)
- `build-essential` e `libglib2.0-0`: necessários para compilar sentence-transformers e faiss
- `COPY requirements.txt` antes de `COPY . .`: aproveita cache de camadas do Docker
- `EXPOSE 5000`: declarativo — o mapeamento real é feito no docker-compose.yml
- O CMD padrão (`python app.py`) é sobrescrito pelo `command:` de cada serviço no compose

### docker-compose.yml
Dois serviços com profiles separados — apenas um sobe por vez:
```yaml
services:
  flask:
    build: .
    container_name: assistente_flask
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data      # histórico e banco ChromaDB persistidos no host
      - ./.env:/app/.env      # credenciais nunca incluídas na imagem
    command: python app.py
    profiles: ["flask"]

  fastapi:
    build: .
    container_name: assistente_fastapi
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    command: uvicorn api:app --host 0.0.0.0 --port 8000
    profiles: ["fastapi"]
```

Comandos Docker:
```bash
docker compose build                        # build da imagem (uma vez)
docker compose --profile flask up           # sobe Flask em localhost:5000
docker compose --profile fastapi up         # sobe FastAPI em localhost:8000/docs
docker compose --profile flask down         # para e remove o container
docker compose --profile fastapi down
```

Volumes montados:
- `./data:/app/data` → historico_global.json e chroma_db/ sobrevivem ao `docker compose down`
- `./.env:/app/.env` → credenciais injetadas em runtime, nunca na imagem

### v1/assistente.py  (FAISS + pkl)
Fluxo por pergunta:
1. _precisa_rag(pergunta) → decide se busca RAG ou responde direto do system
2. buscar_rag(query, top_k=5, score_min=0.45) → FAISS sobre indice_rag.pkl (1.553 chunks)
3. _carregar_historico() → últimas 20 msgs do historico_global.json
4. llm.chat.completions → DeepSeek com system + histórico + contexto RAG
5. _salvar_historico() → persiste turno
6. retorna {resposta, fontes, historico, timestamp}

### v2/assistente.py  (ChromaDB puro)
Fluxo por pergunta:
1. _precisa_rag(pergunta)
2. _expandir_query(pergunta, historico[-4:]) → LLM reformula com contexto de histórico
3. buscar_rag() → ChromaDB (1.763 chunks, 33 arquivos)
4. _carregar_historico()
5. llm.chat.completions → DeepSeek
6. _salvar_historico()
7. retorna {resposta, fontes, historico, timestamp}

### v3/assistente.py  (LangChain 1.x / LCEL)
Fluxo por pergunta:
1. _precisa_rag(pergunta) → idêntico à v2
2. _expandir_query() → LCEL: PromptTemplate | ChatOpenAI | StrOutputParser
3. buscar_rag() → vectorstore.similarity_search_with_score()
4. _carregar_historico() → idêntico à v2
5. rag_chain.invoke() → LCEL: ChatPromptTemplate | ChatOpenAI | StrOutputParser
6. _salvar_historico() → idêntico à v2
7. retorna {resposta, fontes, historico, timestamp}

Diferenças v2 → v3:
- `chromadb.PersistentClient` → `Chroma(persist_directory=...)`
- `SentenceTransformer.encode()` manual → `HuggingFaceEmbeddings` (interno ao vectorstore)
- `openai.OpenAI().chat.completions.create()` → `ChatOpenAI` + LCEL chains
- `collection.query(query_embeddings=...)` → `vs.similarity_search_with_score(query, filter=...)`
- Lista `[{'role':..., 'content':...}]` → `[HumanMessage(...) | AIMessage(...)]`
- `_get_modelo_emb()` removida — desnecessária
- Banco `data/chroma_db/` compartilhado: nenhuma reindexação ao trocar v2 → v3

## COMPONENTES COMPARTILHADOS

### Resolução de caminhos (idêntica em v1, v2, v3)
```python
for _candidate in [_HERE, _HERE.parent, _HERE.parent.parent, _HERE.parent.parent.parent]:
    if (_candidate / 'shared' / 'llm_factory.py').exists():
        sys.path.insert(0, str(_candidate))
        break
```

### _precisa_rag() (v2 e v3)
Detecta perguntas gerais e não envia contexto RAG.
Palavras-chave: 'módulos', 'lista', 'estrutura', 'visão geral', etc.
Exceção: se contém 'como', 'código', 'função', 'algoritmo' → usa RAG mesmo assim.

### Score mínimo RAG (v1, v2, v3)
score_min=0.45. v2/v3: `score = 1 - (dist / 2.0)` — converte distância cosine ChromaDB para [0,1].

### Query Expansion com consciência de histórico (v2 e v3)
- CONTINUAÇÃO: resolve pronomes usando historico[-4:]
- INDEPENDENTE: ignora histórico quando a pergunta menciona novo módulo/tema

### System prompt (v1, v2, v3 — idêntico)
- Estrutura dos 8 módulos hardcoded
- Regras: perguntas gerais → direto | técnicas → RAG
- Info de provider: como trocar DeepSeek/OpenAI/Ollama via .env

### Memória global (v1, v2, v3 — idêntica)
- Arquivo: data/historico_global.json
- Formato: [{role, content, timestamp}]
- Janela: 20 msgs no contexto. Sem distinção de usuário.
- No Docker: persistido em `./data` (volume montado no host)

### templates/index.html (Flask apenas)
- Dark mode, IBM Plex Mono + Sans, verde terminal (#4ade80)
- Chat com Markdown | Painel lateral com chunks RAG | Polling de status | Perguntas rápidas

## LIMITAÇÃO CONHECIDA
Índice RAG sobre AGENT_CONTEXT.md — não sobre arquivos .py.
Código específico retorna resposta gerada pelo LLM, não o código real.
Para código real: indice_codigo.pkl do 05_rag_codigo_especializado.ipynb.

## COMO EXECUTAR

### Local
```bash
conda activate eai07
pip install flask fastapi uvicorn chromadb
pip install langchain langchain-community langchain-chroma langchain-openai langchain-huggingface

cd EAI_07_AI_Generative/06_Projetos_Reais/Assistente_Tecnico_IA

# Edite o import em app.py e/ou api.py:
# from v1.assistente import ...   ← FAISS
# from v2.assistente import ...   ← ChromaDB
# from v3.assistente import ...   ← LangChain

python app.py                                             # Flask → localhost:5000
uvicorn api:app --host 0.0.0.0 --port 8000 --reload      # FastAPI → localhost:8000
```

### Docker
```bash
cd EAI_07_AI_Generative/06_Projetos_Reais/Assistente_Tecnico_IA

docker compose build

docker compose --profile flask   up      # Flask   → localhost:5000
docker compose --profile fastapi up      # FastAPI → localhost:8000  |  /docs
```

## REINDEXAÇÃO

v1 — rebuild completo:
1. Delete data/cache/indice_rag.pkl
2. Reexecute 03_rag_basico.ipynb (~67s)
3. Reinicie o servidor

v2/v3 — incremental:
```python
atualizar_modulo(PROJETO_BASE, 'EAI_09')   # ~5s
```

v2/v3 — rebuild completo:
```python
resetar_banco(confirmar=True)   # ~110s
```

## FAQ
Q: Como escolher entre Flask e FastAPI?
A: Flask tem a interface HTML completa (chat visual). FastAPI expõe REST JSON com Swagger —
   ideal para integração com outros sistemas ou para testar a API diretamente em /docs.
   Ambos usam o mesmo assistente.py e têm as mesmas rotas /chat, /limpar e /status.

Q: Posso rodar Flask e FastAPI ao mesmo tempo?
A: Sim, mas não com Docker Compose (os profiles sobem um por vez). Localmente, basta rodar
   `python app.py` em um terminal e `uvicorn api:app --port 8000` em outro.

Q: Como escolher entre v1, v2 e v3?
A: Editar a linha de import em app.py (e api.py se necessário):
   v1 → FAISS + pkl | v2 → ChromaDB direto | v3 → LangChain 1.x (mesmo banco da v2).

Q: Qual a diferença entre v2 e v3?
A: Mesmo banco, mesma lógica, mesma interface pública. v3 usa LangChain como camada de abstração
   (objetivo didático). Não é necessário reindexar ao trocar v2 → v3.

Q: O banco ChromaDB sobrevive ao docker compose down?
A: Sim. O volume `./data:/app/data` monta o diretório local dentro do container.
   O `chroma_db/` e o `historico_global.json` ficam no host e persistem entre execuções.

Q: O .env é copiado para a imagem Docker?
A: Não. O `.env` é montado como volume em runtime (`./.env:/app/.env`).
   Nunca é incluído na imagem — importante para não vazar credenciais.

Q: O que fazer se o assistente mostrar "índice indisponível" no status?
A: v1: executar 03_rag_basico.ipynb. v2/v3: executar 04_rag_avancado_chromadb.ipynb
   ou 04_rag_avancado_langchain.ipynb. No Docker, o banco deve estar em ./data/chroma_db/
   antes de subir o container (o volume é montado mas não indexado automaticamente).

Q: Por que v2/v3 têm mais chunks que v1 (1.763 vs 1.553)?
A: v1 foi indexado com 26 AGENT_CONTEXT.md. v2/v3 varrem subpastas recursivamente e
   encontram 33 arquivos — inclui AGENT_CONTEXT.md de subpastas do EAI_07 criados depois.

Q: Por que usar LangChain 1.x em vez de 0.x na v3?
A: LangChain 0.x está descontinuado. Na 1.x, `langchain.memory` e `langchain.chains` foram
   removidos — substitutos são LCEL e ChatMessageHistory. Na v3 do assistente o histórico
   ainda é gerenciado em JSON (compatibilidade com v1/v2), por isso usa
   `_historico_para_mensagens()` em vez de `RunnableWithMessageHistory`.

## TAGS DE BUSCA
Flask FastAPI uvicorn Docker Dockerfile docker-compose profiles volumes
REST JSON Swagger /docs /redoc Pydantic BaseModel HTTPException async await
RAG semântico FAISS ChromaDB LangChain LCEL assistente técnico EAI_07
v1 v2 v3 query expansion upsert incremental import switch versão backend
Chroma HuggingFaceEmbeddings ChatOpenAI PromptTemplate ChatPromptTemplate
MessagesPlaceholder StrOutputParser HumanMessage AIMessage LCEL pipe operator
langchain_core langchain_community langchain_chroma langchain_openai langchain_huggingface
similarity_search_with_score _contexto fallback banco compartilhado
threading background lazy load polling status histórico global memória conversacional
score mínimo system prompt IBM Plex dark mode painel fontes chunks

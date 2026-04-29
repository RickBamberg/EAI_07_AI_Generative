# Assistente Técnico IA

**Projeto de** EAI_07_AI_Generative / 06_Projetos_Reais  
**Ambiente:** `eai07` (Python 3.11)

Aplicação com interface de chat que responde perguntas sobre os módulos
EAI_01 a EAI_08 usando RAG semântico sobre os `AGENT_CONTEXT.md` do curso.
Disponível em dois servidores (**Flask** e **FastAPI**) e três versões de backend RAG,
com suporte a execução local e via **Docker**.

---

## Versões de backend RAG

| Versão | Backend       | Banco                           | Chunks | Query Expansion |  
|--------|---------------|---------------------------------|--------|-----------------|   
|  `v1`  | FAISS + pkl   | `data/cache/indice_rag.pkl`     |  1.553 | Não             |  
|  `v2`  | ChromaDB puro | `data/chroma_db/`               |  1.763 | Sim             |  
|  `v3`  | LangChain 1.x | `data/chroma_db/` (mesmo da v2) |  1.763 | Sim             |  

> **v3 não requer reindexação** se a v2 já foi executada — o banco é compartilhado.

---

## Servidores disponíveis

| Arquivo  | Framework | Porta | Interface                          |  
|----------|-----------|-------|------------------------------------|  
| `app.py` | Flask     | 5000  | Chat HTML (`templates/index.html`) |  
| `api.py` | FastAPI   | 8000  | REST JSON + Swagger (`/docs`)      | 

Ambos importam o mesmo `assistente.py` (v1, v2 ou v3) e expõem as mesmas rotas:
`POST /chat`, `POST /limpar`, `GET /status`.

---

## Estrutura

```
Assistente_Tecnico_IA/
├── app.py                     ← Servidor Flask (interface HTML)
├── api.py                     ← Servidor FastAPI (REST + Swagger)
├── Dockerfile                 ← Imagem única para Flask e FastAPI
├── docker-compose.yml         ← Profiles: flask (5000) | fastapi (8000)
├── requirements.txt           ← Dependências Python
├── templates/
│   └── index.html             ← Interface de chat dark mode
├── data/
│   └── historico_global.json  ← Histórico persistido (criado automaticamente)
├── v1/assistente.py           ← Núcleo v1: FAISS + pkl
├── v2/assistente.py           ← Núcleo v2: ChromaDB puro
├── v3/assistente.py           ← Núcleo v3: LangChain 1.x / LCEL
├── AGENT_CONTEXT.md
└── README.md
```

---

## Pré-requisitos

1. Ambiente `eai07` ativo (ou Docker instalado para execução containerizada)
2. **Índice RAG gerado** para a versão de backend escolhida:
   - v1 → execute `03_RAG/03_rag_basico.ipynb` → gera `data/cache/indice_rag.pkl`
   - v2 → execute `03_RAG/04_rag_avancado_chromadb.ipynb` → gera `data/chroma_db/`
   - v3 → mesmo banco da v2 (ou `03_RAG/04_rag_avancado_langchain.ipynb`)
3. `.env` configurado na raiz do projeto com `DEEPSEEK_API_KEY` e `LLM_MODEL`

---

## Execução local

### 1. Instalar dependências

```bash
conda activate eai07
pip install flask fastapi uvicorn

# v2 — ChromaDB puro
pip install chromadb

# v3 — LangChain 1.x
pip install langchain langchain-community langchain-chroma langchain-openai langchain-huggingface
```

### 2. Escolher a versão de backend

Edite a linha de import em `app.py` **e** em `api.py`:

```python
# Escolha uma linha e comente as outras:
from v1.assistente import responder, limpar_historico, _get_indice, _pronto  # FAISS + pkl
from v2.assistente import responder, limpar_historico, _get_indice, _pronto  # ChromaDB
from v3.assistente import responder, limpar_historico, _get_indice, _pronto  # LangChain
```

### 3. Iniciar o servidor

```bash
cd EAI_07_AI_Generative/06_Projetos_Reais/Assistente_Tecnico_IA

# Flask — interface HTML em localhost:5000
python app.py

# FastAPI — REST + Swagger em localhost:8000
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Execução com Docker

### Build da imagem

```bash
cd EAI_07_AI_Generative/06_Projetos_Reais/Assistente_Tecnico_IA
docker compose build
```

### Subir o servidor desejado

```bash
# Flask — acesse http://localhost:5000
docker compose --profile flask up

# FastAPI — acesse http://localhost:8000  |  Swagger: http://localhost:8000/docs
docker compose --profile fastapi up
```

### Volumes montados automaticamente

| Volume local | Container | Descrição |
|---|---|---|
| `./data` | `/app/data` | Histórico e banco ChromaDB persistidos fora do container |
| `./.env` | `/app/.env` | Credenciais e configurações (nunca incluídas na imagem) |

> O banco `data/chroma_db/` e o histórico `data/historico_global.json` sobrevivem
> ao `docker compose down` porque estão no volume local `./data`.

### Parar o container

```bash
docker compose --profile flask down    # ou --profile fastapi
```

---

## API — FastAPI (`api.py`)

A FastAPI expõe a mesma lógica do Flask em formato REST com validação Pydantic
e documentação interativa automática.

### Rotas

| Rota      | Método | Descrição                                                |  
|-----------|--------|----------------------------------------------------------|  
| `/`       | GET    | Redireciona para `/docs`                                 |  
| `/chat`   | POST   | Processa pergunta, retorna resposta + fontes + histórico |  
| `/limpar` | POST   | Apaga o histórico global                                 |  
| `/status` | GET    | Status do servidor e contagem de chunks RAG              |  
| `/docs`   | GET    | Swagger UI interativo                                    |  
| `/redoc`  | GET    | Documentação ReDoc                                       |  

### Exemplo de chamada

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"pergunta": "Como funciona o RAG no EAI_07?"}'
```

```json
{
  "resposta": "O RAG no EAI_07 ...",
  "fontes": [
    {
      "modulo": "EAI_07_AI_Generative",
      "titulo": "Busca semântica",
      "score": 0.812,
      "arquivo": "EAI_07.../AGENT_CONTEXT.md",
      "trecho": "..."
    }
  ],
  "historico": [...],
  "timestamp": "14:32"
}
```

### Schemas Pydantic

```python
class PerguntaRequest(BaseModel):
    pergunta: str   # min_length=1

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
```

---

## Arquitetura

```
Cliente (browser / curl)
        │
        ├── porta 5000 → app.py (Flask)    → templates/index.html
        └── porta 8000 → api.py (FastAPI)  → JSON + /docs
                │
                └── assistente.responder(pergunta)   ← mesma função nas 3 versões
                        │
                        ├── _precisa_rag()
                        ├── _expandir_query()         ← LCEL (v3) ou OpenAI direto (v2)
                        ├── buscar_rag()              ← FAISS (v1) | ChromaDB (v2) | LangChain (v3)
                        ├── _carregar_historico()
                        ├── LLM (system + histórico + contexto RAG)
                        └── _salvar_historico()
```

---

## Vantagens FastAPI vs Flask

| Aspecto      | Flask (`app.py`)          | FastAPI (`api.py`)                         |  
|--------------|---------------------------|--------------------------------------------|  
| Interface    | Chat HTML completo        | REST JSON + Swagger                        |  
| Validação    | Manual                    | Pydantic automático                        |  
| Documentação | —                         | `/docs` e `/redoc` gerados automaticamente |  
| Async        | Não nativo                | Suporte nativo a `async/await`             |  
| Erros        | Customizado               | `HTTPException` padronizado                |  
| Uso ideal    | Demo com interface visual | Integração com outros sistemas             |  

---

## Caminhos — nada precisa ser copiado

Cada `assistente.py` resolve os caminhos automaticamente subindo a hierarquia
até encontrar o `shared/llm_factory.py`. No Docker, o `WORKDIR /app` é a raiz do projeto
e os volumes montam `data/` e `.env` diretamente lá.

```
EAI_07_AI_Generative/                    ← detectado automaticamente (local)
├── shared/llm_factory.py
├── data/
│   ├── cache/indice_rag.pkl             ← v1
│   └── chroma_db/                       ← v2 e v3 (compartilhado)
└── 06_Projetos_Reais/
    └── Assistente_Tecnico_IA/
        ├── app.py  /  api.py
        ├── v1/ v2/ v3/
        └── data/                        ← histórico local do projeto
```

---

## Reindexação

**v1 — rebuild completo:**
1. Delete `data/cache/indice_rag.pkl`
2. Reexecute `03_RAG/03_rag_basico.ipynb` (~67s)
3. Reinicie o servidor

**v2 / v3 — atualização incremental:**
```python
# No notebook 04_rag_avancado_chromadb.ipynb ou 04_rag_avancado_langchain.ipynb
atualizar_modulo(PROJETO_BASE, 'EAI_09')   # só o módulo alterado (~5s)
```

**v2 / v3 — rebuild completo:**
```python
resetar_banco(confirmar=True)   # apaga e reconstrói (~110s)
```

---

## Integração com o curso

| Componente      | v1                    | v2                               | v3                                |
|-----------------|-----------------------|----------------------------------|-----------------------------------|
| Banco           | `indice_rag.pkl`      | `chroma_db/`                     | `chroma_db/` (mesmo)              |
| Gerado por      | `03_rag_basico.ipynb` | `04_rag_avancado_chromadb.ipynb` | `04_rag_avancado_langchain.ipynb` |
| Chunks          | 1.553 / 26 arquivos   | 1.763 / 33 arquivos              | 1.763 / 33 arquivos               |
| Embedding       | `SentenceTransformer` | `SentenceTransformer`            | `HuggingFaceEmbeddings`           |
| Query expansion | —                     | Sim                              | Sim (LCEL)                        |
| Reindexação     | Rebuild completo      | Upsert incremental               | Upsert incremental                |

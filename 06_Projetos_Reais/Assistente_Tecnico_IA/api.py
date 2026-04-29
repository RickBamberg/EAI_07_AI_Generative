"""
api.py
=======
Servidor FastAPI do Assistente Técnico IA.

Equivalente ao app.py (Flask), com as mesmas rotas e lógica de negócio.
Vantagens sobre o Flask:
    - Tipagem com Pydantic (validação automática do payload)
    - Documentação interativa em /docs (Swagger) e /redoc
    - Suporte nativo a async/await
    - Respostas de erro padronizadas via HTTPException

Rotas:
    GET  /          → redireciona para /docs (sem template HTML)
    POST /chat      → processa pergunta, retorna JSON
    POST /limpar    → apaga histórico global
    GET  /status    → saúde do servidor e índice RAG

Execução:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from v3.assistente import responder, limpar_historico, _get_indice, _pronto


# ── Aplicação ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Assistente Técnico IA",
    description="API do Assistente Técnico do curso Especialista em IA — RAG + LangChain + DeepSeek.",
    version="3.0.0",
)


# ── Schemas Pydantic ───────────────────────────────────────────────────────────

class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., min_length=1, description="Pergunta do usuário.")


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


# ── Rotas ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redireciona para a documentação interativa."""
    return RedirectResponse(url="/docs")


@app.post("/chat", response_model=ChatResponse)
async def chat(body: PerguntaRequest):
    """
    Processa uma pergunta e retorna a resposta do assistente.

    - Executa query expansion + busca RAG semântica
    - Gera resposta com contexto e histórico via LangChain LCEL
    - Retorna resposta, fontes e histórico atualizado
    """
    try:
        resultado = responder(body.pergunta)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/limpar", response_model=LimparResponse)
def limpar():
    """Apaga o histórico global de conversas."""
    return limpar_historico()


@app.get("/status", response_model=StatusResponse)
def status():
    """Retorna a saúde do servidor e o estado do índice RAG."""
    indice = _get_indice() if _pronto else None
    return {
        "status": "ok",
        "pronto": _pronto,
        "rag"   : "carregado" if indice else "carregando...",
        "chunks": indice._collection.count() if indice else 0,
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 55)
    print("  Assistente Técnico IA  [FastAPI]")
    print("  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 55 + "\n")

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

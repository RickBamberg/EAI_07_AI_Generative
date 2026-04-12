# Assistente Técnico IA

**Projeto de** EAI_07_AI_Generative / 06_Projetos_Reais  
**Ambiente:** `eai07` (Python 3.11)

Aplicação Flask com interface de chat que responde perguntas sobre os módulos
EAI_01 a EAI_08 usando RAG semântico sobre os `AGENT_CONTEXT.md` do curso.

---

## Pré-requisitos

1. Ambiente `eai07` ativo
2. **Índice RAG gerado** — execute o `03_RAG/03_rag_basico.ipynb` até o final.
   O cache deve existir em `data/cache/indice_rag.pkl`
3. Flask instalado:

```bash
conda activate eai07
pip install flask
```

---

## Como executar

```bash
cd EAI_07_AI_Generative/06_Projetos_Reais/Assistente_Tecnico_IA
python app.py
```

Acesse: **http://localhost:5000**

---

## Estrutura

```
Assistente_Tecnico_IA/
├── app.py              ← Flask: rotas GET / e POST /chat, /limpar, /status
├── assistente.py       ← Núcleo: RAG + memória + LLM
├── templates/
│   └── index.html      ← Interface de chat com painel de fontes
├── data/
│   └── historico_global.json  ← Histórico persistido (criado automaticamente)
├── AGENT_CONTEXT.md
└── README.md
```

---

## Arquitetura

```
POST /chat
    │
    ▼
assistente.responder(pergunta)
    │
    ├── buscar_rag()         → top-5 chunks semânticos do índice FAISS
    │                           (reutiliza data/cache/indice_rag.pkl do 03_RAG)
    │
    ├── _carregar_historico() → últimas 20 mensagens do historico_global.json
    │
    ├── llm.chat.completions  → DeepSeek com system + histórico + contexto RAG
    │
    └── _salvar_historico()  → persiste turno atual
    │
    └── retorna {resposta, fontes, historico, timestamp}
```

---

## Interface

- **Chat** à esquerda — mensagens com renderização Markdown (código, listas, bold)
- **Painel lateral** à direita — chunks RAG recuperados com módulo, título e score
  - Clique em qualquer chunk para expandir e ver o trecho completo
- **Perguntas rápidas** na tela inicial para começar a conversa
- **Limpar histórico** no rodapé do painel apaga o `historico_global.json`

---

## Caminhos — nada precisa ser copiado

O `assistente.py` resolve os caminhos automaticamente a partir do próprio local:

```
EAI_07_AI_Generative/           ← _HERE.parent.parent
├── shared/llm_factory.py       ← detectado automaticamente
├── data/cache/indice_rag.pkl   ← índice compartilhado com o 03_RAG
└── 06_Projetos_Reais/
    └── Assistente_Tecnico_IA/
        └── assistente.py       ← _HERE
```

Não é necessário copiar o índice nem o `shared/` — o assistente usa os arquivos
originais do projeto diretamente.

---

## ⚠️ Lembrete: reindexação ao criar novos módulos

O índice `indice_rag.pkl` é um snapshot dos `AGENT_CONTEXT.md` existentes no
momento em que o `03_rag_basico.ipynb` foi executado. Se você criar novos módulos
(ex: EAI_09, EAI_10) com seus próprios `AGENT_CONTEXT.md`, o assistente **não
saberá nada sobre eles** até o índice ser atualizado.

**Como reindexar:**
1. Delete `data/cache/indice_rag.pkl`
2. Abra e reexecute o `03_RAG/03_rag_basico.ipynb` do início
3. O novo índice será gerado (~67s) e o assistente passa a conhecer os novos módulos

---

## Integração com o curso

| Componente | Origem |
|---|---|
| Índice FAISS | `03_RAG/03_rag_basico.ipynb` |
| Modelo de embedding | `all-MiniLM-L6-v2` (384 dim) |
| LLM Provider | `shared/llm_factory.py` (DeepSeek via .env) |
| Histórico global | `data/memoria/historico_global.json` |

---

## Rotas da API

| Rota | Método | Descrição |
|---|---|---|
| `/` | GET | Interface de chat |
| `/chat` | POST | `{pergunta: str}` → `{resposta, fontes, historico, timestamp}` |
| `/limpar` | POST | Apaga o histórico global |
| `/status` | GET | Status do servidor e contagem de chunks do índice |

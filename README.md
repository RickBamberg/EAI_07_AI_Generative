# EAI_07 — IA Generativa

Módulo de IA Generativa da especialização **ESPECIALISTA_EM_IA**.

> ⚙️ Antes de começar, siga o [SETUP.md](./SETUP.md)

---

## Estrutura do Módulo

```
EAI_07_AI_Generative/
├── .env.example                 ← copie para .env e configure
├── .gitignore
├── SETUP.md                     ← guia de instalação e configuração
├── README.md
│
├── shared/                      ← compartilhado por TODOS os módulos
│   ├── llm_factory.py           ← factory provider-agnóstica
│   ├── requirements.txt         ← dependências base
│   └── __init__.py
│
├── 01_Fundamentos_LLM/
│   ├── tokenizacao.ipynb
│   ├── attention_mecanismo.ipynb
│   └── transformers_basico.ipynb
│
├── 02_Modelos_PreTreinados/
│   ├── 01_uso_apis_llm.ipynb        ← usa shared/llm_factory.py
│   ├── 02_prompt_engineering.ipynb  ← usa shared/llm_factory.py
│   ├── 03_function_calling.ipynb    ← usa shared/llm_factory.py
│   └── 04_modelos_locais.ipynb      ← usa shared/llm_factory.py
│
├── 03_RAG/
│   ├── 01_embeddings_vetores.ipynb
│   ├── 02_chunking_estrategias.ipynb
│   ├── 03_rag_basico.ipynb          ← usa shared/llm_factory.py
│   ├── 04_rag_avancado.ipynb        ← usa shared/llm_factory.py
│   └── 05_rag_codigo_especializado.ipynb
│
├── 04_Fine_Tuning/
│   ├── 01_preparacao_dados.ipynb
│   └── 02_fine_tuning_gpt.ipynb
│
├── 05_Agentes/
│   ├── 01_agente_simples.ipynb      ← usa shared/llm_factory.py
│   ├── 02_ferramentas_customizadas.ipynb
│   ├── 03_multi_agentes.ipynb
│   └── 04_memoria_conversacional.ipynb
│
└── 06_Projetos_Reais/
    └── Assistente_Tecnico_IA/       ← projeto final do módulo
        ├── src/llm_client.py        ← aponta para shared/llm_factory.py
        └── ...
```

---

## Providers Suportados

| Provider | Custo | Modelo padrão | Cartão BR |
|---|---|---|---|
| `deepseek` | 💰 Muito barato | `deepseek-chat` | ✅ |
| `ollama` | 🆓 Gratuito (local) | `llama3.2` | — |
| `anthropic` | 💵 Médio | `claude-haiku-4-5` | ✅ |
| `openai` | 💵 Médio | `gpt-4o-mini` | ❌ |

**Troca de provider:** edite apenas o `.env` — nenhum notebook ou script muda.

---

## Início rápido

```bash
# 1. Instalar dependências
pip install -r shared/requirements.txt

# 2. Configurar provider
cp .env.example .env
# edite .env com sua chave

# 3. Testar
python shared/llm_factory.py

# 4. Abrir qualquer notebook
jupyter notebook
```

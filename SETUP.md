# SETUP — EAI_07_AI_Generative

Guia de configuração para todos os notebooks e projetos do módulo.

---

## 1. Instalar dependências

```bash
# Na raiz do EAI_07_AI_Generative/
pip install -r shared/requirements.txt
```

---

## 2. Configurar o provider LLM

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o .env com seu provider e chave
```

### Opção A — DeepSeek (recomendado: barato + aceita cartão BR)
1. Crie conta em [platform.deepseek.com](https://platform.deepseek.com)
2. Gere uma API key em **API Keys**
3. Configure o `.env`:
```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-...
```

### Opção B — Ollama local (gratuito, sem internet)
1. Instale o Ollama: [ollama.com/download](https://ollama.com/download)
2. Baixe um modelo:
```bash
ollama pull llama3.2      # 2GB — recomendado para começar
ollama pull mistral       # 4GB — mais capaz
```
3. Configure o `.env`:
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```

### Opção C — Anthropic (Claude Haiku)
1. Crie conta em [console.anthropic.com](https://console.anthropic.com)
2. Configure o `.env`:
```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5
ANTHROPIC_API_KEY=sk-ant-...
```

### Opção D — OpenAI
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

---

## 3. Testar a conexão

```bash
python shared/llm_factory.py
```

Saída esperada:
```
==================================================
  Provider : deepseek
  Modelo   : deepseek-chat
==================================================

Testando chat()...
  Conexão OK com deepseek-chat

Testando chat_stream()...
  1
  2
  3
  4
  5
```

---

## 4. Como usar nos notebooks

Todos os notebooks do módulo importam o LLM assim:

```python
import sys, os
sys.path.append(os.path.abspath('..'))   # aponta para EAI_07/

from shared.llm_factory import chat, chat_stream, get_provider_info

# Ver qual provider está ativo
print(get_provider_info())

# Fazer uma pergunta
resposta = chat("O que é um Transformer?")
print(resposta)

# Streaming (tokens em tempo real)
for chunk in chat_stream("Explique atenção em deep learning"):
    print(chunk, end="", flush=True)
```

> **Nota:** O `.env` é lido automaticamente da raiz do EAI_07 —
> não é preciso configurar nada a mais nos notebooks.

---

## 5. Trocar de provider a qualquer momento

Basta editar **uma linha** no `.env` na raiz do EAI_07:

```env
# Antes
LLM_PROVIDER=deepseek

# Depois (nenhum código muda)
LLM_PROVIDER=ollama
```

---

## Estrutura de dependências

```
EAI_07_AI_Generative/
├── .env                        ← configure aqui (único para tudo)
├── shared/
│   ├── llm_factory.py          ← importado por todos
│   └── requirements.txt        ← instale primeiro
└── 06_Projetos_Reais/
    └── Assistente_Tecnico_IA/
        └── requirements.txt    ← instale depois (só Flask + pytest)
```

# 02 — Modelos Pré-Treinados

## 📚 Sobre este Submódulo

Uso prático de LLMs via API com arquitetura **provider-agnóstica** — o mesmo código funciona com DeepSeek, Anthropic, OpenAI ou Ollama local, apenas mudando o `.env`.

## 🎯 Objetivos de Aprendizagem

- ✅ Usar APIs de LLM com o padrão provider-agnóstico do projeto
- ✅ Dominar técnicas de Prompt Engineering
- ✅ Implementar Function Calling com tratamento de DSML (DeepSeek)
- ✅ Rodar modelos locais com Ollama

## 📂 Estrutura

```
02_Modelos_PreTreinados/
├── 01_uso_apis_llm.ipynb          # Chamadas básicas, temperatura, streaming
├── 02_prompt_engineering.ipynb    # Zero-shot, few-shot, CoT, templates
├── 03_function_calling.ipynb      # Tools, ciclo completo, ToolRunner
└── 04_modelos_locais.ipynb        # Ollama, comparativo API vs local
```

## 📖 Conteúdo

### 01_uso_apis_llm.ipynb
- Estrutura de mensagens: `system` / `user` / `assistant`
- Parâmetros: `temperature`, `max_tokens`
- Resposta completa vs streaming (`chat_stream()`)
- Conversa multi-turno com histórico
- Estimativa de custo (DeepSeek ≈ $0.27/1M tokens input)

### 02_prompt_engineering.ipynb
- **Zero-shot**: pergunta direta sem exemplos
- **Few-shot**: exemplos antes da pergunta para formatar a saída
- **Chain-of-Thought (CoT)**: forçar raciocínio passo a passo
- **Role Prompting**: definir persona/papel do modelo
- **Output Formatting**: saída em JSON, tabela markdown
- **PromptTemplate**: classe reutilizável com variáveis — usada no Assistente Técnico

### 03_function_calling.ipynb
- Definição de tools: `name` + `description` + `parameters` (JSON schema)
- Ciclo completo: pergunta → tool_call → execução Python → resposta final
- Tratamento do formato DSML do DeepSeek (via `shared/tool_runner.py`)
- `ToolRunner`: interface OO para registrar e usar ferramentas
- Mini-agente do projeto com `resumo_modulo` tool

### 04_modelos_locais.ipynb
- Instalação do Ollama (Windows/Linux/macOS)
- Modelos recomendados para 8GB RAM: `llama3.2`, `phi3`
- Mesma interface do SDK OpenAI — só muda `base_url`
- Comparativo de tempo e qualidade: DeepSeek vs Ollama
- Gerenciamento de modelos via terminal e Python

## 🔧 Arquivos Compartilhados

```
shared/
├── llm_factory.py    # chat(), chat_stream(), get_provider_info()
└── tool_runner.py    # executar_com_tools(), ToolRunner — com fallback DSML
```

### Como importar nos notebooks

```python
import sys, os
sys.path.append(os.path.abspath('..'))

from shared.llm_factory import chat, chat_stream, get_provider_info
from shared.tool_runner import executar_com_tools, ToolRunner
```

## ⚙️ Configuração

O provider é definido no `.env` na raiz do EAI_07 — **nenhum notebook muda**:

```env
# DeepSeek (padrão — recomendado)
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-...

# Ollama (local, gratuito)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```

## ⚠️ Nota sobre DeepSeek e DSML

O DeepSeek às vezes retorna tool calls no formato proprietário DSML em vez do padrão OpenAI. O `shared/tool_runner.py` detecta e trata automaticamente com um loop de até 8 iterações e mapeamento de parâmetros alternativos (`diretorio → caminho`, etc).

## 🔗 Conexão com os Próximos Submódulos

- **03_RAG**: usa `chat()` do llm_factory para gerar respostas com contexto
- **05_Agentes**: usa `ToolRunner` como base para agentes ReAct completos
- **06_Projetos_Reais**: `Assistente_Tecnico_IA` importa tudo de `shared/`

---
*Parte do projeto ESPECIALISTA_EM_IA — EAI_07 IA Generativa*

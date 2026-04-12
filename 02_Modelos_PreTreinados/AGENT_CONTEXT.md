# AGENT_CONTEXT.md — EAI_07 / 02_Modelos_PreTreinados

> **Propósito**: Contexto estruturado para o Assistente Técnico responder questões sobre este submódulo.
> **Última atualização**: Março 2026

## RESUMO EXECUTIVO

**Submódulo**: 02_Modelos_PreTreinados  
**Módulo pai**: EAI_07_AI_Generative  
**Objetivo**: Uso prático de LLMs via API com arquitetura provider-agnóstica  
**Provider padrão**: DeepSeek (`deepseek-chat`)  
**Notebooks**: 4 (uso de APIs, prompt engineering, function calling, modelos locais)

---

## ESTRUTURA DE ARQUIVOS

```
02_Modelos_PreTreinados/
├── 01_uso_apis_llm.ipynb          [Chamadas básicas, temperatura, streaming, multi-turno]
├── 02_prompt_engineering.ipynb    [Zero-shot, few-shot, CoT, templates]
├── 03_function_calling.ipynb      [Tools, ToolRunner, fallback DSML]
└── 04_modelos_locais.ipynb        [Ollama, comparativo, gerenciamento]
```

---

## ARQUITETURA PROVIDER-AGNÓSTICA

### Configuração (.env na raiz do EAI_07)
```env
LLM_PROVIDER=deepseek      # deepseek | anthropic | openai | ollama
LLM_MODEL=deepseek-chat    # modelo específico do provider
DEEPSEEK_API_KEY=sk-...    # chave do provider escolhido
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048
```

### Uso nos notebooks (sempre igual, independente do provider)
```python
import sys, os
sys.path.append(os.path.abspath('..'))
from shared.llm_factory import chat, chat_stream, get_provider_info

resposta = chat("Qual módulo estudou KNN?")
for chunk in chat_stream("Explique RAG passo a passo"):
    print(chunk, end="", flush=True)
```

### Providers suportados
| Provider | SDK usado | Base URL | Chave |
|----------|-----------|----------|-------|
| deepseek | openai | https://api.deepseek.com | DEEPSEEK_API_KEY |
| openai | openai | padrão | OPENAI_API_KEY |
| ollama | openai | http://localhost:11434/v1 | não precisa |
| anthropic | anthropic | padrão | ANTHROPIC_API_KEY |

---

## NOTEBOOKS — CONTEXTO DETALHADO

### 1. 01_uso_apis_llm.ipynb

**Estrutura de mensagens**:
```python
messages = [
    {"role": "system",    "content": "Você é um assistente técnico."},
    {"role": "user",      "content": "O que é RAG?"},
    {"role": "assistant", "content": "RAG é..."},  # histórico
    {"role": "user",      "content": "Próxima pergunta"},
]
```

**Parâmetros importantes**:
- `temperature=0.0`: determinístico (código, dados estruturados)
- `temperature=0.2`: padrão do projeto (Q&A técnico)
- `temperature=0.7`: criativo (texto livre)
- `max_tokens`: limite da resposta (não do input)

**Streaming**:
```python
for chunk in chat_stream("Explique atenção"):
    print(chunk, end="", flush=True)
```

**Estimativa de custo DeepSeek**:
```python
PRECO_INPUT  = 0.27   # USD por 1M tokens
PRECO_OUTPUT = 1.10   # USD por 1M tokens
# 100 perguntas/dia (20k input + 15k output) ≈ $0.022/dia
```

**Conversa multi-turno**:
```python
# LLMs são stateless — enviar histórico completo a cada chamada
historico = []
historico.append({"role": "user",      "content": pergunta})
historico.append({"role": "assistant", "content": resposta})
# Próxima chamada recebe todo o historico
```

---

### 2. 02_prompt_engineering.ipynb

**Técnicas implementadas**:

**Zero-shot** — direto sem exemplos:
```python
chat("Classifique: 'O modelo foi excelente'. Responda: Positivo, Negativo ou Neutro.")
```

**Few-shot** — exemplos antes da pergunta:
```
Texto: "API respondeu em 200ms" → POSITIVO | alta | velocidade
Texto: "Funcionou às vezes"     → NEUTRO | média | inconsistência
Texto: [novo texto]             → ?
```

**Chain-of-Thought**:
```python
system = """Ao responder, siga:
1. ANÁLISE DAS RESTRIÇÕES
2. AVALIAÇÃO DAS OPÇÕES
3. RECOMENDAÇÃO
4. PRÓXIMOS PASSOS"""
```

**PromptTemplate** (classe reutilizável):
```python
TEMPLATE_RAG = PromptTemplate(
    template="CONTEXTO:\n{contexto}\n\nPERGUNTA: {pergunta}",
    variaveis=["contexto", "pergunta"]
)
prompt = TEMPLATE_RAG.formatar(contexto="...", pergunta="...")
```

**Templates do projeto**:
- `TEMPLATE_RAG`: contexto + pergunta (usado no Assistente Técnico)
- `TEMPLATE_REVISAO_CODIGO`: revisão com linguagem e contexto do projeto
- `TEMPLATE_EXPLICACAO`: conceito + nível + estilo + max_linhas

**Regra prática**:
```
Zero-shot → se formato errado → Output Formatting
          → se qualidade ruim → CoT ou Few-shot
```

---

### 3. 03_function_calling.ipynb

**Definição de tool** (formato OpenAI):
```python
tool = {
    "type": "function",
    "function": {
        "name": "calcular",
        "description": "Executa operações matemáticas básicas.",
        "parameters": {
            "type": "object",
            "properties": {
                "operacao": {"type": "string", "enum": ["soma", "subtracao", "multiplicacao", "divisao"]},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operacao", "a", "b"]
        }
    }
}
```

**Ciclo completo**:
```
1. Pergunta + lista de tools → LLM
2. LLM retorna tool_call (nome + argumentos)
3. Python executa a função
4. Resultado → LLM
5. LLM gera resposta final em linguagem natural
```

**Problema DSML do DeepSeek**:
O DeepSeek às vezes retorna tool calls em formato proprietário:
```
<｜DSML｜function_calls>
<｜DSML｜invoke name="funcao">
<｜DSML｜parameter name="param">valor</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>
```

**Solução — shared/tool_runner.py**:
- Loop de até 8 iterações (DSML pode aparecer em qualquer rodada)
- Parser regex com normalização correta de closing tags
- Mapeamento de aliases: `diretorio→caminho`, `extension→extensao`, etc.

**ToolRunner** (interface OO):
```python
runner = (
    ToolRunner(system="Assistente técnico.", verbose=True)
    .registrar(tool_calculadora, calcular)
    .registrar(tool_sistema,     info_sistema)
)
resposta = runner.perguntar("Que horas são?")
```

**Ferramentas implementadas no notebook**:
- `calcular(operacao, a, b)` → soma/subtração/multiplicação/divisão
- `info_sistema(incluir)` → data/hora, versão Python, diretório
- `listar_arquivos(caminho, extensao)` → lista arquivos do projeto
- `resumo_modulo(modulo)` → info de EAI_01 a EAI_08

---

### 4. 04_modelos_locais.ipynb

**Ollama — servidor local de LLMs**:
```
Notebook → HTTP → Ollama (localhost:11434) → Modelo .gguf
```

**Instalação**:
```bash
# Windows: instalador em ollama.com/download
# Linux:   curl -fsSL https://ollama.com/install.sh | sh
# macOS:   brew install ollama
```

**Modelos recomendados para 8GB RAM**:
| Modelo | Tamanho | Comando |
|--------|---------|---------|
| llama3.2 | ~2GB | `ollama pull llama3.2` |
| phi3 | ~2.3GB | `ollama pull phi3` |
| llama3.2:1b | ~1.3GB | `ollama pull llama3.2:1b` |

**Uso com SDK OpenAI (mesmo do projeto)**:
```python
from openai import OpenAI
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
```

**Verificação via Python**:
```python
import requests
r = requests.get("http://localhost:11434")  # retorna "Ollama is running"
r = requests.get("http://localhost:11434/api/tags")  # lista modelos
```

**Estratégia de uso no projeto**:
```
Desenvolvimento  → DeepSeek (rápido, barato)
Créditos esgotados → LLM_PROVIDER=ollama no .env
Dados confidenciais → Ollama (nada sai da máquina)
```

---

## PERGUNTAS FREQUENTES

**Q: Como trocar de provider sem mudar código?**
A: Editar apenas o `.env` na raiz do EAI_07: `LLM_PROVIDER=ollama` e `LLM_MODEL=llama3.2`. O `shared/llm_factory.py` lê o `.env` a cada chamada.

**Q: O que é o formato DSML do DeepSeek?**
A: Formato proprietário de tool calls que o DeepSeek às vezes usa. O `shared/tool_runner.py` detecta e parseia automaticamente, convertendo para o formato padrão OpenAI.

**Q: Qual a diferença entre `chat()` e `chat_stream()`?**
A: `chat()` aguarda a resposta completa e retorna string. `chat_stream()` é um gerador que entrega tokens em tempo real — ideal para interfaces que mostram a resposta enquanto ela é gerada.

**Q: Por que usar Few-shot em vez de Zero-shot?**
A: Quando o formato de saída precisa ser exato (ex: JSON, tabela). Os exemplos no Few-shot mostram ao modelo exatamente o padrão esperado.

**Q: O Ollama precisa de internet?**
A: Só para baixar o modelo pela primeira vez (`ollama pull`). Após o download, funciona 100% offline.

**Q: Como verificar qual provider está ativo?**
A: `from shared.llm_factory import get_provider_info; print(get_provider_info())`

---

## CÓDIGO DE REFERÊNCIA

### Chamada básica
```python
from shared.llm_factory import chat
resposta = chat("O que é embeddings?", system="Responda em 3 linhas.")
```

### Function calling com ToolRunner
```python
from shared.tool_runner import ToolRunner
runner = ToolRunner(system="Assistente do projeto.", verbose=False)
runner.registrar(tool_def, funcao_python)
resposta = runner.perguntar("Qual módulo estudou CNN?")
```

### Verificar provider ativo
```python
from shared.llm_factory import get_provider_info
print(get_provider_info())
# {'provider': 'deepseek', 'model': 'deepseek-chat', 'temperature': 0.2, 'max_tokens': 2048}
```

---

## TAGS DE BUSCA

`#llm-api` `#deepseek` `#ollama` `#provider-agnostico` `#function-calling` `#tool-runner` `#dsml` `#prompt-engineering` `#zero-shot` `#few-shot` `#chain-of-thought` `#streaming` `#temperatura` `#prompt-template` `#modelos-locais` `#llm_factory`

---
**Versão**: 1.0 | **Módulo**: EAI_07_AI_Generative

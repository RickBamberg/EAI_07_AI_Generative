# 05_Agentes — Agentes Autônomos com LLMs

**Submódulo de** EAI_07_AI_Generative  
**Ambiente:** `eai07` (Python 3.11)

---

## O que você vai aprender

Agentes são LLMs que decidem autonomamente quando e como usar ferramentas para
responder perguntas. Em vez de uma chamada única ao modelo, o agente itera —
raciocina, age, observa o resultado, e repete até ter uma resposta completa.

Este submódulo cobre agentes do básico ao avançado em 4 notebooks progressivos:

| Notebook | Foco |
|---|---|
| `01_agente_simples.ipynb` | Padrão ReAct com ferramentas básicas e o `ToolRunner` do shared/ |
| `02_ferramentas_customizadas.ipynb` | Busca na web, leitura de arquivos, geração de código |
| `03_multi_agentes.ipynb` | Pipeline condicional com roteador e especialistas |
| `04_memoria_conversacional.ipynb` | Janela deslizante, persistência e sumarização automática |

---

## Pré-requisitos

- Submódulo `02_Function_Calling` concluído (conceito de tool use)
- Ambiente `eai07` ativo
- `shared/tool_runner.py` disponível na raiz do EAI_07

### Dependências adicionais (notebook 02)

```bash
pip install requests beautifulsoup4
```

---

## Arquitetura base — ToolRunner

Todos os notebooks usam o `ToolRunner` do `shared/tool_runner.py`:

```python
from shared.tool_runner import ToolRunner

agente = ToolRunner(system="Você é um assistente técnico.", verbose=True)
agente.registrar(schema_da_ferramenta, funcao_python)
resposta = agente.perguntar("Qual é a raiz de 144?")
```

O `ToolRunner` trata automaticamente:
- **Tool calls padrão** OpenAI (formato JSON)
- **DSML** — formato alternativo que o DeepSeek usa em alguns casos
- **Aliases de parâmetros** — ex: DeepSeek usa `caminho` onde o código define `diretorio`

---

## Padrão ReAct

O ciclo fundamental de qualquer agente:

```
Pergunta
   │
   ▼
LLM raciocina ──── sem ferramenta ────► Resposta final
   │
   │ precisa de ferramenta
   ▼
Chama ferramenta  (Act)
   │
   ▼
Recebe resultado  (Observe)
   │
   └──────────────────────────────────► LLM raciocina (próxima iteração)
```

---

## Pipeline Multi-Agente

O notebook 03 implementa roteamento inteligente:

```
Pergunta → Roteador (classifica) → Especialista(s) → [Síntese] → Resposta
```

O roteador usa `temperature=0.0` para classificação determinística e retorna
JSON com `tipo` (simples/composto) e `especialistas` (pesquisa/codigo/matematica).

---

## Estratégias de Memória

O notebook 04 implementa três abordagens para gerenciar contexto em conversas longas:

| Estratégia | Quando usar |
|---|---|
| **Janela deslizante** | Conversas curtas, sessão única |
| **Longo prazo** (arquivo JSON) | Assistente pessoal, múltiplas sessões |
| **Sumarização automática** | Conversas técnicas longas e densas |

---

## Boas práticas

**Ferramentas** sempre retornam string — erros incluídos. Nunca lançar exceções dentro de uma ferramenta.

**Descrições** dos schemas são o principal fator para o LLM decidir qual ferramenta usar. Seja específico.

**Parâmetros** devem ter os nomes que o LLM vai usar naturalmente. O `tool_runner.py` tem um mapa de aliases mas é melhor nomear certo desde o início.

**Iterações** limitadas a 8 por padrão. Ajustar se a tarefa precisar de mais passos, mas monitorar loops.

**Memória** — para produção com usuários reais, combinar longo prazo (fatos persistidos) com sumarização (contexto comprimido).

---

## Referências

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [shared/tool_runner.py — documentação interna](../shared/tool_runner.py)

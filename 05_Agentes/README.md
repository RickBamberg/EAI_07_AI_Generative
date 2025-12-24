# 🤖 Agentes Autônomos com LLMs

## 📖 Sobre este Módulo

**Agentes** são LLMs que podem usar **ferramentas** (tools) para resolver tarefas complexas de forma autônoma. Eles decidem quais ações tomar, executam ferramentas, analisam resultados e repetem até resolver o problema.

Este módulo prepara você para o **Projeto Final**, onde criará um agente que analisa código!

---

## 🎯 Objetivos de Aprendizado

Ao completar este módulo, você será capaz de:

- ✅ Entender o padrão **ReAct** (Reasoning + Acting)
- ✅ Criar **ferramentas customizadas** para agentes
- ✅ Implementar agentes com **LangChain**
- ✅ Orquestrar **múltiplos agentes** especializados
- ✅ Gerenciar **memória conversacional**
- ✅ Integrar agentes com **RAG**
- ✅ Debugar e otimizar agentes

---

## 📚 Conteúdo

### 📓 Notebooks

1. **agente_simples.ipynb** 🤖
   - Conceito de agentes
   - Padrão ReAct (Thought → Action → Observation)
   - Criar ferramentas básicas
   - AgentExecutor do LangChain
   - Análise do raciocínio do agente
   - Tratamento de erros

2. **02_ferramentas_customizadas.ipynb** 🔧
   - Criar Tools com `BaseTool`
   - Ferramentas para análise de código:
     - SearchCodeTool (busca no repositório)
     - CompareCodeTool (compara implementações)
     - ExecuteCodeTool (roda código com segurança)
     - AnalyzeComplexityTool (métricas)
     - GenerateDocTool (documentação)
     - SuggestImprovementsTool (melhorias)
   - Integração com RAG

3. **03_multi_agentes.ipynb** 👥
   - Especialização de agentes
   - Agente para ML
   - Agente para DL
   - Agente para NLP
   - Orquestração (supervisor)
   - Comunicação entre agentes

4. **04_memoria_conversacional.ipynb** 🧠
   - ConversationBufferMemory
   - ConversationSummaryMemory
   - Entity Memory
   - Contexto multi-turn
   - Persistência de memória

---

## 🔧 Tecnologias Utilizadas

### Core Agents
- `langchain` - Framework de agentes
- `langchain-openai` - Integração OpenAI
- `openai` - LLM backend

### Tools & Utilities
- `python-dotenv` - Variáveis de ambiente
- `pydantic` - Validação de dados
- `requests` - HTTP requests

### Para Análise de Código
- `ast` - Parse de Python AST
- `radon` - Complexidade ciclomática
- `pylint` - Análise estática

---

## 🚀 Como Usar

```bash
# 1. Instalar dependências
pip install langchain langchain-openai openai python-dotenv

# 2. Configurar API Key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Começar pelo básico
jupyter notebook agente_simples.ipynb

# 4. Criar ferramentas customizadas
jupyter notebook 02_ferramentas_customizadas.ipynb
```

---

## 💰 Custos Estimados

| Atividade | Custo |
|-----------|-------|
| Testar agentes básicos | $0.50 |
| Desenvolver ferramentas | $1.00 |
| Multi-agentes | $0.80 |
| Integração com RAG | $1.00 |
| **Total módulo** | **~$3-4** |

**Dica:** Agentes fazem múltiplas chamadas, use GPT-3.5!

---

## 📊 Pré-requisitos

- ✅ **Módulo 02:** Modelos Pré-Treinados
- ✅ **Módulo 03:** RAG (recomendado)
- ✅ **OpenAI API Key**
- 💡 **Paciência:** Agentes podem errar, é normal!

---

## 🎓 Conceitos-Chave

### 🤖 Agent (Agente)
LLM que pode usar ferramentas para resolver tarefas autonomamente.

### 🔧 Tool (Ferramenta)
Função Python que o agente pode chamar (buscar, calcular, executar, etc).

### 🎯 ReAct Pattern
```
Loop:
  1. THOUGHT: O que preciso fazer?
  2. ACTION: Qual ferramenta usar?
  3. OBSERVATION: O que a ferramenta retornou?
  4. Repeat até resolver
  5. ANSWER: Resposta final
```

### 🧠 Memory
Sistema que mantém contexto entre interações.

### 👥 Multi-Agent
Múltiplos agentes especializados trabalhando juntos.

---

## 💡 Arquitetura de um Agente

```
┌────────────────────────────────────────┐
│           USUÁRIO                      │
│   "Analise o código X"                 │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│           AGENTE (LLM)                 │
│                                        │
│  LOOP ReAct:                           │
│  1. Pensar                             │
│  2. Decidir ação                       │
│  3. Executar tool                      │
│  4. Analisar resultado                 │
│  5. Repetir ou responder              │
└───────────┬────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────┐
│           FERRAMENTAS                  │
│                                        │
│  • BuscarCodigo(query)                │
│  • CompararCodigos(code1, code2)      │
│  • ExecutarCodigo(code)               │
│  • AnalisarComplexidade(code)         │
│  • GerarDocumentacao(code)            │
└────────────────────────────────────────┘
```

---

## 🎯 Exemplo Real

### Query do Usuário:
```
"Quantos algoritmos de Deep Learning temos? 
Depois multiplique por 2."
```

### Raciocínio do Agente:
```
THOUGHT: Preciso buscar algoritmos de DL e depois fazer um cálculo

ACTION: BuscarAlgoritmos
INPUT: {"category": "DL"}
OBSERVATION: ["CNN", "RNN", "LSTM", "Transformers"] - Total: 4

THOUGHT: Agora preciso multiplicar 4 por 2

ACTION: Calculadora
INPUT: {"expression": "4 * 2"}
OBSERVATION: 8

THOUGHT: Tenho a resposta final

ANSWER: "Temos 4 algoritmos de Deep Learning: CNN, RNN, LSTM e 
Transformers. 4 multiplicado por 2 é igual a 8."
```

---

## 🔧 Ferramentas para o Projeto Final

### 1. SearchCodeTool
```python
def search_code(query: str) -> str:
    """Busca código no repositório via RAG"""
    docs = vectorstore.similarity_search(query, k=5)
    return format_results(docs)
```

### 2. CompareCodeTool
```python
def compare_code(code1: str, code2: str) -> str:
    """Compara duas implementações"""
    # Análise de complexidade
    # Diferenças estruturais
    # Recomendações
```

### 3. ExecuteCodeTool
```python
def execute_code(code: str) -> str:
    """Executa código Python com segurança"""
    # Timeout de 5s
    # Sandbox isolado
    # Retorna output
```

### 4. AnalyzeComplexityTool
```python
def analyze_complexity(code: str) -> str:
    """Analisa complexidade do código"""
    # Complexidade ciclomática
    # Contagem de nós AST
    # Métricas de qualidade
```

---

## 💡 Casos de Uso

### 1. Assistente de Código (Seu Projeto!)
```python
User: "Como foi implementado o KNN?"
Agent: 
  → BuscarCodigo("KNN")
  → AnalisarComplexidade(code)
  → GerarDocumentacao(code)
  → Resposta completa com análise
```

### 2. Analista de Dados
```python
User: "Analise o CSV e faça um gráfico"
Agent:
  → LerCSV(file)
  → AnalisarDados(data)
  → GerarGrafico(data, type="bar")
  → SalvarResultado()
```

### 3. Pesquisador
```python
User: "Pesquise sobre Transformers e resuma"
Agent:
  → BuscaWeb("Transformers architecture")
  → AnalisarArtigos(results)
  → GerarResumo()
  → SalvarPDF()
```

---

## ⚠️ Limitações e Cuidados

### Agentes podem:
- ❌ Entrar em loops infinitos
- ❌ Fazer chamadas caras demais
- ❌ Escolher ferramentas erradas
- ❌ Produzir resultados inesperados

### Como mitigar:
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,        # Limite de iterações
    max_execution_time=30,   # Timeout
    early_stopping_method="force",
    handle_parsing_errors=True
)
```

---

## 🎯 Exercícios Práticos

### Exercício 1: Agente Analisador
Criar agente que analisa notebook e gera relatório com:
- Número de células
- Funções definidas
- Complexidade média
- Recomendações

### Exercício 2: Multi-Agente
Criar sistema com 3 agentes:
- Agente ML (especialista em sklearn)
- Agente DL (especialista em keras/pytorch)
- Supervisor (roteia queries)

### Exercício 3: Agente com Memória
Implementar agente que "lembra" de análises anteriores.

---

## 🔗 Próximos Passos

- 🎯 **Módulo 06:** Projeto Final
  - Integrar tudo: RAG + Agentes + Ferramentas
  - Criar Assistente Técnico completo
  - Deploy em produção

---

## 📚 Recursos Adicionais

### Papers
- [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [Generative Agents](https://arxiv.org/abs/2304.03442)

### Documentação
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

### Exemplos
- [LangChain Agent Examples](https://github.com/langchain-ai/langchain/tree/master/docs/docs/use_cases/agent_simulations)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [BabyAGI](https://github.com/yoheinakajima/babyagi)

---

## 💡 Dicas Importantes

### 1. Descrições de Ferramentas
```python
# ❌ Ruim
description = "Busca código"

# ✅ Bom
description = """
Busca código no repositório baseado em query semântica.
Use quando precisar encontrar implementações específicas.
Input: String descrevendo o que buscar (ex: "implementação de KNN")
"""
```

### 2. Validação de Inputs
```python
class MyTool(BaseTool):
    def _run(self, query: str) -> str:
        # Validar input
        if not query or len(query) < 3:
            return "Query muito curta, seja mais específico"
        
        # Executar
        ...
```

### 3. Error Handling
```python
try:
    result = risky_operation()
except Exception as e:
    return f"Erro: {str(e)}. Tente outra abordagem."
```

---

## 📈 Status

⏳ **Próximo**

### Roadmap
- [ ] Notebook 1: Agente Simples ✅ (já criado!)
- [ ] Notebook 2: Ferramentas Customizadas
- [ ] Notebook 3: Multi-Agentes
- [ ] Notebook 4: Memória Conversacional

---

## 👨‍💻 Autor

**Carlos H. B. Marques**
- GitHub: [@RickBamberg](https://github.com/RickBamberg)
- LinkedIn: [Carlos Henrique Bamberg Marques](https://www.linkedin.com/in/carlos-henrique-bamberg-marques/)

---

⬅️ [Voltar: Fine-Tuning](../04_Fine_Tuning/README.md) | ➡️ [Próximo: Projeto Final](../06_Projetos_Reais/README.md)

# 🤖 Modelos Pré-Treinados

## 📖 Sobre este Módulo

Neste módulo você aprenderá a **usar modelos de linguagem já treinados** através de APIs comerciais (OpenAI, Anthropic) e locais (Ollama). Este é o módulo mais prático para começar a construir aplicações reais com IA Generativa.

---

## 🎯 Objetivos de Aprendizado

Ao completar este módulo, você será capaz de:

- ✅ Fazer chamadas para **APIs de LLMs** (OpenAI GPT, Claude)
- ✅ Dominar técnicas de **Prompt Engineering**
- ✅ Usar **Function Calling** para integrar LLMs com ferramentas
- ✅ Rodar **modelos locais** com Ollama (grátis e privado)
- ✅ Comparar diferentes modelos e escolher o melhor para cada caso
- ✅ Gerenciar custos e otimizar uso de APIs

---

## 📚 Conteúdo

### 📓 Notebooks

1. **02_uso_apis_llm.ipynb** 🔥
   - Setup de APIs (OpenAI, Anthropic)
   - Parâmetros fundamentais (temperature, max_tokens, top_p)
   - Streaming de respostas
   - Conversas com histórico (chat)
   - Comparação: GPT-4 vs GPT-3.5 vs Claude
   - Gestão de custos e otimização

2. **prompt_engineering.ipynb** 🎨
   - Zero-shot prompting
   - Few-shot learning (1-shot, 3-shot, 5-shot)
   - Chain-of-Thought (CoT) reasoning
   - ReAct pattern (Reasoning + Acting)
   - System prompts efetivos
   - Self-consistency
   - Tree of Thoughts
   - Prompt templates reutilizáveis

3. **03_function_calling.ipynb** 🔧
   - Function calling com OpenAI
   - Structured outputs (JSON mode)
   - Definir schemas de funções
   - Múltiplas ferramentas em sequência
   - Casos de uso práticos
   - Simulação no Ollama

4. **04_modelos_locais.ipynb** 💻
   - Instalar e configurar Ollama
   - Rodar Llama 3 localmente
   - Comparar com APIs comerciais
   - Quantização de modelos
   - Trade-offs: custo vs qualidade vs velocidade
   - Quando usar local vs API

---

## 🔧 Tecnologias Utilizadas

### APIs Comerciais
- `openai` - OpenAI GPT-3.5/GPT-4
- `anthropic` - Claude (opcional)

### Modelos Locais
- `ollama` - Framework para rodar LLMs localmente
- `llama3` - Meta's Llama 3 (open source)

### Utilitários
- `python-dotenv` - Gerenciar API keys
- `requests` - HTTP requests
- `langchain` - Framework para LLMs

---

## 🚀 Como Usar

### Opção 1: Com APIs Comerciais

```bash
# 1. Instalar dependências
pip install openai anthropic python-dotenv langchain

# 2. Criar arquivo .env
echo "OPENAI_API_KEY=sk-..." > .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# 3. Abrir notebooks
jupyter notebook 02_uso_apis_llm.ipynb
```

### Opção 2: Com Ollama (Grátis)

```bash
# 1. Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Baixar Llama 3
ollama pull llama3

# 3. Verificar instalação
ollama run llama3 "Hello!"

# 4. Abrir notebooks
jupyter notebook 04_modelos_locais.ipynb
```

---

## 💰 Custos Estimados

### APIs Comerciais

| Modelo | Custo/1k tokens | Uso neste módulo |
|--------|----------------|------------------|
| GPT-3.5-turbo | $0.0015 | ~$1-2 |
| GPT-4 | $0.03 | ~$3-5 |
| Claude Sonnet | $0.003 | ~$1-2 |

**Total estimado:** $2-5 (com $5 iniciais, sobra crédito!)

### Ollama
- **Custo:** $0 (100% gratuito)
- **Hardware:** 8GB RAM mínimo
- **Modelos:** Llama 3, Mistral, CodeLlama, etc.

---

## 📊 Pré-requisitos

- ✅ **Módulo 01:** Fundamentos de LLM (recomendado)
- ✅ **Python 3.11+**
- ✅ **API Key da OpenAI** (ou usar Ollama gratuitamente)
- ⚠️ **8GB RAM** (para rodar Ollama)

---

## 🎓 Conceitos-Chave

### 🎨 Prompt Engineering
Arte de criar instruções efetivas para LLMs obterem os melhores resultados.

### 🔧 Function Calling
Permite que LLMs chamem funções externas e retornem dados estruturados (JSON).

### 💻 Modelos Locais
Rodar LLMs no seu próprio computador, sem APIs, sem custo, com privacidade total.

### 🌡️ Temperature
Controla a criatividade do modelo:
- `0.0` = Determinístico (mesma resposta sempre)
- `1.0` = Criativo (respostas variadas)
- `2.0` = Muito criativo (pode ser incoerente)

---

## 💡 Dicas Importantes

### Para Economizar
```python
# Use GPT-3.5 para testes
model="gpt-3.5-turbo"  # 20x mais barato que GPT-4

# Limite tokens
max_tokens=500

# Temperature baixa evita retries
temperature=0
```

### Para Melhor Qualidade
```python
# Use GPT-4 para produção
model="gpt-4"

# Mais tokens para respostas completas
max_tokens=2000

# Temperature moderada
temperature=0.7
```

---

## 🔗 Próximos Passos

Após dominar modelos pré-treinados, você estará pronto para:

- 🔍 **Módulo 03:** Implementar RAG (busca + geração)
- 🤖 **Módulo 05:** Criar agentes autônomos
- 🎯 **Módulo 06:** Projeto final (Assistente Técnico)

---

## 📚 Recursos Adicionais

### Documentação Oficial
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Ollama Documentation](https://ollama.com/)

### Tutoriais e Cursos
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Documentation](https://python.langchain.com/docs/)

### Papers Importantes
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629)

---

## 🎯 Exercícios Práticos

Cada notebook inclui exercícios práticos:

1. **APIs:** Criar analisador de código automático
2. **Prompts:** Gerar testes unitários com few-shot
3. **Functions:** Integrar com busca em seus projetos
4. **Local:** Comparar Ollama vs OpenAI

---

## 📈 Status

🚧 **Em Andamento**

### Progresso
- [x] Notebook 1: Uso de APIs ✅
- [x] Notebook 2: Prompt Engineering ✅
- [x] Notebook 3: Function Calling ✅
- [ ] Notebook 4: Modelos Locais ⏳

---

## 👨‍💻 Autor

**Carlos H. B. Marques**
- GitHub: [@RickBamberg](https://github.com/RickBamberg)
- LinkedIn: [Carlos Henrique Bamberg Marques](https://www.linkedin.com/in/carlos-henrique-bamberg-marques/)
- Email: rick.bamberg@gmail.com

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

⬅️ [Voltar: Fundamentos LLM](../01_Fundamentos_LLM/README.md) | ➡️ [Próximo: RAG](../03_RAG/README.md)

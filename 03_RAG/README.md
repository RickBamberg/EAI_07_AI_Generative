# 🔍 RAG - Retrieval Augmented Generation

## 📖 Sobre este Módulo

**RAG** é uma das técnicas mais importantes da IA Generativa moderna! Ela combina **busca de informações** (retrieval) com **geração de texto** (generation) para criar sistemas que respondem perguntas baseadas em documentos específicos.

Este é o **核心 (coração)** do projeto final - o Assistente Técnico que entenderá todo o seu código!

---

## 🎯 Objetivos de Aprendizado

Ao completar este módulo, você será capaz de:

- ✅ Entender **embeddings** e busca vetorial
- ✅ Implementar diferentes estratégias de **chunking**
- ✅ Criar um **pipeline RAG completo**
- ✅ Usar **FAISS** para busca eficiente
- ✅ Implementar **reranking** e busca híbrida
- ✅ Criar RAG especializado para **código Python**
- ✅ Avaliar e otimizar sistemas RAG

---

## 📚 Conteúdo

### 📓 Notebooks

1. **embeddings_vetores.ipynb** 🔢
   - O que são embeddings
   - Modelos de embedding (OpenAI, Sentence-Transformers)
   - Similaridade de cosseno
   - FAISS: busca vetorial eficiente
   - Diferentes tipos de índices
   - Visualização com PCA
   - Comparação entre modelos

2. **02_chunking_estrategias.ipynb** ✂️
   - Por que fazer chunking?
   - Estratégias: tamanho fixo, recursivo, semântico
   - Overlap entre chunks
   - Chunking para código (por função, classe)
   - Metadata e filtragem
   - Comparação de estratégias

3. **03_rag_basico.ipynb** 🏗️ (核心!)
   - Pipeline RAG completo
   - Document loaders (PDF, TXT, CSV, código)
   - Text splitters
   - Vector stores (FAISS, ChromaDB)
   - Retrieval strategies
   - Chain de Q&A com LangChain
   - Prompts customizados
   - Avaliação de RAG

4. **04_rag_avancado.ipynb** 🚀
   - Reranking com Cohere
   - Hybrid search (keyword + semantic)
   - Query rewriting/expansion
   - Multi-query retrieval
   - Contexto multi-hop
   - Self-query retriever
   - Citações precisas

5. **05_rag_codigo_especializado.ipynb** 💻
   - Parse de AST (Abstract Syntax Tree)
   - Extração de funções e classes
   - Linking entre arquivos
   - Análise de dependências
   - Indexação de notebooks (.ipynb)
   - Busca por tipo (função, classe, variável)
   - Metadata rica para código

---

## 🔧 Tecnologias Utilizadas

### Core RAG
- `langchain` - Framework RAG completo
- `langchain-openai` - Integração OpenAI
- `openai` - Embeddings e LLM

### Vector Stores
- `faiss-cpu` - Busca vetorial eficiente
- `chromadb` - Vector database alternativa
- `pinecone-client` - Cloud vector database (opcional)

### Embeddings
- `sentence-transformers` - Modelos open source
- `openai` - text-embedding-3-small/large

### Utilities
- `pypdf` - Processar PDFs
- `python-docx` - Processar Word
- `nbformat` - Processar notebooks

---

## 🚀 Como Usar

```bash
# 1. Instalar dependências
pip install langchain langchain-openai faiss-cpu sentence-transformers

# 2. Configurar API Key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Começar pelos fundamentos
jupyter notebook embeddings_vetores.ipynb

# 4. Pipeline completo
jupyter notebook 03_rag_basico.ipynb
```

---

## 💰 Custos Estimados

| Atividade | Custo |
|-----------|-------|
| Criar embeddings (1000 docs) | $0.10 |
| Queries de teste (100x) | $0.50 |
| Desenvolvimento completo | $2-3 |
| **Total módulo** | **~$3-4** |

**Dica:** Use `text-embedding-3-small` (mais barato e suficiente)

---

## 📊 Pré-requisitos

- ✅ **Módulo 02:** Modelos Pré-Treinados
- ✅ **OpenAI API Key** (para embeddings + LLM)
- ✅ **8GB RAM** (para FAISS com datasets médios)
- 📚 **Documentos para indexar** (PDFs, código, etc)

---

## 🎓 Conceitos-Chave

### 📊 Embeddings
Representação vetorial de texto que captura significado semântico.
```python
"cachorro" = [0.2, 0.8, -0.3, ...]
"cão" = [0.21, 0.79, -0.31, ...]  # Vetores similares!
```

### 🔍 Vector Search
Busca por similaridade em espaço vetorial, não por palavras-chave.

### ✂️ Chunking
Divisão de documentos em pedaços menores para melhor recuperação.

### 🎯 RAG Pipeline
```
Query → Embedding → Busca → Top-K Docs → Prompt + Docs → LLM → Answer
```

### 🔁 Reranking
Segunda etapa de busca que reordena resultados para maior precisão.

---

## 💡 Arquitetura RAG

```
┌──────────────────────────────────────────────────┐
│              1. INDEXAÇÃO (offline)              │
├──────────────────────────────────────────────────┤
│  Documentos → Chunking → Embeddings → Vector DB │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│              2. QUERY (runtime)                  │
├──────────────────────────────────────────────────┤
│  Query → Embedding → Search → Top-K → Context   │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│              3. GENERATION                       │
├──────────────────────────────────────────────────┤
│  Context + Query → LLM → Answer com citações    │
└──────────────────────────────────────────────────┘
```

---

## 🎯 Casos de Uso

### 1. Chatbot Corporativo
- Responde baseado em docs internos
- PDFs, Wikis, Confluence, etc.

### 2. Assistente de Código
- Busca em repositórios Git
- Explica implementações
- **← Este é nosso projeto!**

### 3. Sistema de FAQ
- Busca semântica em perguntas frequentes
- Melhor que busca por palavra-chave

### 4. Análise de Documentos
- Extração de informações
- Sumarização de múltiplos docs

---

## 📈 Métricas de Avaliação

### Retrieval
- **Precision@K:** Quantos dos top-K são relevantes?
- **Recall@K:** Quantos relevantes estão no top-K?
- **MRR:** Mean Reciprocal Rank

### Generation
- **Faithfulness:** Resposta fiel aos documentos?
- **Relevance:** Resposta relevante à pergunta?
- **Coherence:** Resposta coerente?

---

## 🔗 Próximos Passos

Após dominar RAG, você pode:

- 🤖 **Módulo 05:** Criar agentes que usam RAG como ferramenta
- 🎯 **Módulo 06:** Projeto Final - Assistente Técnico
- 🔧 **Integrar:** RAG em aplicações web (Flask, Streamlit)

---

## 📚 Recursos Adicionais

### Papers Fundamentais
- [RAG Paper (Original)](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [ColBERT: Efficient Passage Search](https://arxiv.org/abs/2004.12832)

### Tutoriais
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch)
- [Building Production RAG](https://www.anthropic.com/index/building-effective-agents)

### Ferramentas
- [FAISS Documentation](https://faiss.ai/)
- [ChromaDB](https://www.trychroma.com/)
- [Weaviate](https://weaviate.io/)

---

## 🎯 Exercícios Práticos

### Exercício 1: RAG para seus projetos
Indexar descrições de seus projetos anteriores (EAI_01 a EAI_06) e criar sistema de busca.

### Exercício 2: Comparar estratégias
Testar 3 estratégias de chunking diferentes e comparar qualidade das respostas.

### Exercício 3: Multi-índice
Criar índices separados para ML, DL e NLP. Implementar roteamento inteligente.

### Exercício 4: Avaliação
Criar dataset de 20 perguntas e respostas esperadas. Medir precisão do RAG.

---

## 💡 Dicas Importantes

### Para Melhor Qualidade
```python
# 1. Chunk size adequado
chunk_size=1000  # Não muito pequeno, não muito grande

# 2. Overlap suficiente
chunk_overlap=200  # Evita perder contexto

# 3. Top-K apropriado
k=5  # Nem muito pouco, nem muito

# 4. Reranking
# Use quando precisar de máxima precisão
```

### Para Economizar
```python
# 1. Use embedding local
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Cache embeddings
# Salve o vectorstore para não recriar

# 3. Use GPT-3.5 para geração
model="gpt-3.5-turbo"
```

---

## 📈 Status

⏳ **Próximo**

### Roadmap
- [ ] Notebook 1: Embeddings e Vetores
- [ ] Notebook 2: Chunking Strategies
- [ ] Notebook 3: RAG Básico
- [ ] Notebook 4: RAG Avançado
- [ ] Notebook 5: RAG para Código

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

⬅️ [Voltar: Modelos Pré-Treinados](../02_Modelos_PreTreinados/README.md) | ➡️ [Próximo: Fine-Tuning](../04_Fine_Tuning/README.md)

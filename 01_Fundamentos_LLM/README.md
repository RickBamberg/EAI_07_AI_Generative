# 01 — Fundamentos de LLM

## 📚 Sobre este Submódulo

Estabelece a base teórica e conceitual dos Large Language Models antes de usar qualquer API.
Os notebooks foram desenvolvidos no módulo **EAI_05_NLP_com_Transformers** e são reaproveitados aqui como fundamento para o EAI_07.

## 🎯 Objetivos de Aprendizagem

- ✅ Entender como texto é convertido em tokens
- ✅ Compreender o mecanismo de atenção (Attention)
- ✅ Visualizar a arquitetura Transformer completa
- ✅ Conectar teoria com uso prático de APIs LLM

## 📂 Estrutura

```
01_Fundamentos_LLM/
├── tokenizacao.ipynb            # Como texto vira números
├── attention_mecanismo.ipynb    # Self-attention e multi-head attention
└── transformers_basico.ipynb    # Arquitetura completa do Transformer
```

## 📖 Conteúdo

### tokenizacao.ipynb
- Tokenização por subpalavras (BPE, WordPiece)
- Vocabulário e IDs de tokens
- Tokens especiais: `[CLS]`, `[SEP]`, `[PAD]`
- Por que "tokenização" importa para custo de API

### attention_mecanismo.ipynb
- Queries, Keys e Values (Q, K, V)
- Cálculo de atenção: `softmax(QK^T / √d_k) · V`
- Multi-head attention
- Visualização dos pesos de atenção

### transformers_basico.ipynb
- Encoder e Decoder
- Positional Encoding
- Feed-forward layers
- Conexão com modelos modernos (GPT, BERT, Claude)

## 🔗 Conexão com os Próximos Submódulos

- **02_Modelos_PreTreinados**: usa LLMs via API com base nessa teoria
- **03_RAG**: embeddings são saídas do encoder do Transformer
- **05_Agentes**: function calling é construído sobre a geração do decoder

---
*Parte do projeto ESPECIALISTA_EM_IA — EAI_07 IA Generativa*

# AGENT_CONTEXT.md — EAI_07 / 01_Fundamentos_LLM

> **Propósito**: Contexto estruturado para o Assistente Técnico responder questões sobre este submódulo.
> **Última atualização**: Março 2026

## RESUMO EXECUTIVO

**Submódulo**: 01_Fundamentos_LLM  
**Módulo pai**: EAI_07_AI_Generative  
**Objetivo**: Base teórica de LLMs antes do uso prático de APIs  
**Notebooks**: tokenizacao.ipynb, attention_mecanismo.ipynb, transformers_basico.ipynb  
**Origem**: Reaproveitados do EAI_05_NLP_com_Transformers

---

## ESTRUTURA DE ARQUIVOS

```
01_Fundamentos_LLM/
├── tokenizacao.ipynb            [Tokenização, vocabulário, tokens especiais]
├── attention_mecanismo.ipynb    [Q, K, V, self-attention, multi-head]
└── transformers_basico.ipynb    [Encoder, decoder, positional encoding]
```

---

## NOTEBOOKS — CONTEXTO DETALHADO

### 1. tokenizacao.ipynb

**Conceitos implementados**:
- Tokenização por subpalavras: BPE (Byte-Pair Encoding) e WordPiece
- Conversão texto → IDs de tokens
- Tokens especiais: `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`
- Diferença entre tokens e palavras

**Por que importa para APIs**:
- Custo de API é cobrado por token, não por palavra
- 1 palavra em português ≈ 1.3 tokens (mais que inglês)
- `max_tokens` nos parâmetros refere-se a tokens, não palavras

**Bibliotecas usadas**:
- `transformers` (HuggingFace)
- `tokenizers`

**Fórmula de estimativa de custo**:
```
tokens ≈ palavras × 1.3   (português)
custo  = tokens / 1_000_000 × preco_por_milhao
```

---

### 2. attention_mecanismo.ipynb

**Conceitos implementados**:
- Queries (Q), Keys (K), Values (V)
- Cálculo da atenção scaled dot-product
- Multi-head attention
- Visualização dos pesos de atenção

**Fórmula central**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**Parâmetros**:
- `d_k`: dimensão das Keys (fator de escala para estabilidade numérica)
- `d_model`: dimensão do modelo (ex: 512, 768, 4096)
- `h`: número de heads no multi-head attention

**Multi-head attention**:
```python
# Cada head aprende a prestar atenção em aspectos diferentes
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)
```

**Bibliotecas usadas**:
- `torch` (implementação)
- `matplotlib` (visualização dos pesos)
- `numpy`

---

### 3. transformers_basico.ipynb

**Conceitos implementados**:
- Arquitetura completa Encoder-Decoder
- Positional Encoding (sin/cos)
- Feed-forward layers
- Layer Normalization
- Conexão com modelos modernos

**Arquitetura**:
```
Input → Embedding + Positional Encoding
      → [Encoder Block × N]
         → Multi-Head Self-Attention
         → Add & Norm
         → Feed-Forward
         → Add & Norm
      → [Decoder Block × N]
         → Masked Multi-Head Attention
         → Cross-Attention (com encoder)
         → Feed-Forward
      → Linear + Softmax → Output
```

**Positional Encoding**:
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Modelos baseados em Transformer**:
| Modelo | Tipo | Uso principal |
|--------|------|---------------|
| BERT | Encoder only | Classificação, embeddings |
| GPT | Decoder only | Geração de texto |
| T5 | Encoder-Decoder | Tradução, sumarização |
| DeepSeek | Decoder only | Geração, chat |
| Claude | Decoder only | Geração, chat, raciocínio |

**Bibliotecas usadas**:
- `torch`, `torch.nn`
- `numpy`, `matplotlib`

---

## CONCEITOS — REFERÊNCIA RÁPIDA

**Token**: unidade mínima de texto processada pelo modelo (subpalavra)

**Embedding**: vetor numérico que representa um token no espaço semântico

**Atenção**: mecanismo que pondera quais tokens são relevantes para cada posição

**Context window**: número máximo de tokens que o modelo processa de uma vez
- DeepSeek: 64k tokens
- Claude: 200k tokens

**Temperature**: parâmetro que controla a aleatoriedade da geração
- 0.0 → determinístico
- 1.0 → muito criativo

---

## PERGUNTAS FREQUENTES

**Q: Qual a diferença entre token e palavra?**
A: Palavras são divididas em subpalavras. "tokenização" pode virar ["token", "ização"]. Em português, 1 palavra ≈ 1.3 tokens em média.

**Q: Por que dividir por √d_k na atenção?**
A: Para evitar que o produto escalar QK^T fique muito grande com dimensões altas, o que causaria gradientes muito pequenos após o softmax.

**Q: Qual a diferença entre encoder e decoder?**
A: Encoder processa a entrada inteira de uma vez (bidirecional). Decoder gera token por token, só vê tokens anteriores (causal/autoregressive).

**Q: Por que LLMs como GPT e DeepSeek são só decoder?**
A: Geração de texto é autoregressive — cada novo token depende dos anteriores. O encoder bidirecional é melhor para tarefas de compreensão (classificação).

**Q: O que é positional encoding?**
A: Como Transformers não têm recorrência, precisam de uma forma de saber a posição de cada token. O positional encoding injeta essa informação via funções seno/cosseno.

---

## CONEXÕES COM OUTROS SUBMÓDULOS

**→ 02_Modelos_PreTreinados**:
- APIs de LLM são modelos Transformer em produção
- `max_tokens`, `temperature` fazem mais sentido com essa base

**→ 03_RAG**:
- Embeddings do RAG são saídas do encoder do Transformer
- `sentence-transformers` usa BERT (encoder only)

**→ 05_Agentes**:
- Function calling é gerado pelo decoder escolhendo tokens especiais
- O modelo "decide" chamar uma função pelo mecanismo de atenção

---

## TAGS DE BUSCA

`#transformer` `#attention` `#tokenizacao` `#llm-fundamentos` `#encoder` `#decoder` `#positional-encoding` `#multi-head-attention` `#bert` `#gpt` `#embeddings` `#tokens` `#temperatura` `#context-window`

---
**Versão**: 1.0 | **Módulo**: EAI_07_AI_Generative

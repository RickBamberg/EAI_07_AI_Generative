# 🎯 Fine-Tuning de Modelos de Linguagem

## 📖 Sobre este Módulo

**Fine-Tuning** é o processo de adaptar um modelo pré-treinado para tarefas específicas usando seus próprios dados. Embora RAG seja suficiente para a maioria dos casos, fine-tuning é ideal quando você precisa que o modelo aprenda padrões específicos do seu domínio.

⚠️ **Nota:** Este módulo é **OPCIONAL**. RAG (Módulo 03) resolve 90% dos problemas de forma mais simples e barata!

---

## 🎯 Objetivos de Aprendizado

Ao completar este módulo, você será capaz de:

- ✅ Decidir quando usar **Fine-Tuning vs RAG**
- ✅ Preparar datasets para fine-tuning
- ✅ Fazer fine-tuning via **OpenAI API**
- ✅ Usar **LoRA/PEFT** para fine-tuning eficiente
- ✅ Avaliar modelos fine-tuned
- ✅ Comparar custos e benefícios

---

## 📚 Conteúdo

### 📓 Notebooks

1. **01_preparacao_dados.ipynb** 📊
   - Quando fazer fine-tuning?
   - Formato de dados (JSONL)
   - Qualidade vs quantidade
   - Validação de dataset
   - Exemplos práticos

2. **02_fine_tuning_gpt.ipynb** 🔧
   - Fine-tuning via OpenAI API
   - Upload de dataset
   - Monitoramento do job
   - Testar modelo fine-tuned
   - Custos e pricing

3. **03_lora_peft.ipynb** 💡 (Avançado)
   - LoRA (Low-Rank Adaptation)
   - PEFT (Parameter-Efficient Fine-Tuning)
   - Fine-tuning local com menos recursos
   - Comparação com full fine-tuning

4. **04_avaliacao_modelo.ipynb** 📈
   - Métricas de avaliação
   - Comparar: base model vs fine-tuned
   - Testes A/B
   - Quando vale a pena?

---

## 🤔 Fine-Tuning vs RAG

### Quando usar Fine-Tuning?
✅ **Use quando:**
- Tem milhares de exemplos de alta qualidade
- Precisa de consistência de estilo/tom
- Domínio muito específico (médico, jurídico)
- Quer reduzir latência (modelo menor)

❌ **NÃO use quando:**
- Tem poucos exemplos (< 100)
- Dados mudam frequentemente
- RAG já resolve o problema
- Orçamento limitado

### Quando usar RAG?
✅ **Use quando:**
- Precisa de informações atualizadas
- Quer transparência (cita fontes)
- Não tem muitos exemplos
- **← 90% dos casos!**

---

## 🔧 Tecnologias Utilizadas

### Fine-Tuning via API
- `openai` - Fine-tuning GPT-3.5
- `anthropic` - Fine-tuning Claude (beta)

### Fine-Tuning Local
- `transformers` - HuggingFace
- `peft` - Parameter-Efficient Fine-Tuning
- `bitsandbytes` - Quantização
- `accelerate` - Treinamento distribuído

### Avaliação
- `datasets` - Carregar datasets
- `evaluate` - Métricas
- `wandb` - Tracking de experimentos

---

## 🚀 Como Usar

### Opção 1: Fine-Tuning via OpenAI API

```bash
# 1. Instalar dependências
pip install openai python-dotenv

# 2. Preparar dados
jupyter notebook 01_preparacao_dados.ipynb

# 3. Fazer fine-tuning
jupyter notebook 02_fine_tuning_gpt.ipynb
```

### Opção 2: Fine-Tuning Local (Avançado)

```bash
# Requer GPU com 16GB+ VRAM
pip install transformers peft bitsandbytes accelerate

jupyter notebook 03_lora_peft.ipynb
```

---

## 💰 Custos

### OpenAI Fine-Tuning

| Modelo | Training (por 1k tokens) | Usage (por 1k tokens) |
|--------|-------------------------|----------------------|
| GPT-3.5-turbo | $0.008 | $0.012 (8x base) |

**Exemplo real:**
```
Dataset: 1000 exemplos @ 500 tokens cada
Training: 1000 * 0.5 * $0.008 = $4
Uso (10k queries): 10 * 0.5 * $0.012 = $60/mês

Total primeiro mês: ~$64
```

### Local (LoRA/PEFT)
- **Custo:** $0 (usa sua GPU)
- **Hardware:** GPU 16GB+ VRAM
- **Tempo:** 2-8 horas

---

## 📊 Pré-requisitos

### Para Fine-Tuning via API
- ✅ Dataset de qualidade (100+ exemplos)
- ✅ OpenAI API Key
- ✅ Budget (~$10-50 para testes)

### Para Fine-Tuning Local
- ✅ GPU NVIDIA (16GB+ VRAM)
- ✅ CUDA instalado
- ✅ Experiência com PyTorch
- ⚠️ Paciência (pode demorar horas!)

---

## 🎓 Conceitos-Chave

### 🎯 Fine-Tuning
Treinar camadas finais de um modelo pré-treinado com seus dados específicos.

### 💡 LoRA (Low-Rank Adaptation)
Técnica que treina apenas matrizes pequenas, economizando 90% de memória.

### 📊 PEFT (Parameter-Efficient Fine-Tuning)
Família de técnicas (LoRA, Prefix Tuning, etc) que tornam fine-tuning mais eficiente.

### 🔢 Quantização
Reduzir precisão (32-bit → 8-bit → 4-bit) para economizar memória.

---

## 💡 Exemplo de Caso de Uso

### Assistente de Atendimento ao Cliente

**Problema:** 
- Precisa seguir script específico
- Tom formal e consistente
- Milhares de exemplos de conversas

**Solução:**
```python
# 1. Coletar 1000+ conversas históricas
# 2. Formatar em JSONL
# 3. Fine-tune GPT-3.5
# 4. Deploy do modelo customizado

# Resultado:
# - Respostas consistentes
# - Tom correto sempre
# - Menos tokens (modelo menor)
```

---

## 🎯 Quando Fine-Tuning Vale a Pena?

### ✅ Casos Bons para Fine-Tuning

1. **Classificação com padrões**
   - Análise de sentimento específica
   - Categorização de tickets
   - Detecção de intenções

2. **Geração com estilo específico**
   - Comunicados formais
   - Relatórios técnicos
   - Respostas padronizadas

3. **Extração estruturada**
   - JSON sempre no mesmo formato
   - Parsing de documentos específicos

### ❌ Casos onde RAG é Melhor

1. **Q&A sobre documentos**
   - Manuais técnicos
   - Base de conhecimento
   - **← Seu caso! Use RAG**

2. **Informações atualizadas**
   - Notícias
   - Preços
   - Dados que mudam

3. **Transparência necessária**
   - Precisa citar fontes
   - Auditoria de respostas

---

## 📈 Comparação Prática

```python
# Cenário: Assistente de código Python

# OPÇÃO 1: RAG
└── Custo: $0.01/query
└── Atualização: Instantânea (reindex)
└── Qualidade: 8/10
└── Implementação: 1 semana

# OPÇÃO 2: Fine-Tuning
└── Custo: $50 treino + $0.12/query
└── Atualização: Retreinar (horas/dias)
└── Qualidade: 9/10
└── Implementação: 3-4 semanas

# ESCOLHA: RAG! (90% dos casos)
```

---

## 🔗 Próximos Passos

- 🤖 **Módulo 05:** Agentes (podem usar modelos fine-tuned)
- 🎯 **Módulo 06:** Projeto Final (RAG é suficiente!)
- 🚀 **Avançado:** Combinar RAG + Fine-Tuning

---

## 📚 Recursos Adicionais

### Documentação
- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Tutoriais
- [Fine-Tuning GPT-3.5](https://platform.openai.com/docs/guides/fine-tuning)
- [LoRA Training Guide](https://huggingface.co/blog/lora)

### Papers
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Fine-Tuning](https://arxiv.org/abs/2305.14314)

---

## 📈 Status

⏳ **Próximo** (Opcional)

---

## 💡 Recomendação Final

**Para o seu projeto (Assistente Técnico):**
- ✅ Use **RAG** (Módulo 03)
- ❌ Pule Fine-Tuning (economize tempo e $$$)
- 🎯 Foque no projeto final!

**Fine-tuning** é interessante para aprender, mas não é necessário para seu objetivo.

---

## 👨‍💻 Autor

**Carlos H. B. Marques**
- GitHub: [@RickBamberg](https://github.com/RickBamberg)
- LinkedIn: [Carlos Henrique Bamberg Marques](https://www.linkedin.com/in/carlos-henrique-bamberg-marques/)

---

⬅️ [Voltar: RAG](../03_RAG/README.md) | ➡️ [Próximo: Agentes](../05_Agentes/README.md)

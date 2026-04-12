# 04_Fine_Tuning — Fine-Tuning de LLMs

**Submódulo de** EAI_07_AI_Generative  
**Ambiente:** `eai07` (Python 3.11)

---

## O que você vai aprender

Fine-tuning é o processo de continuar o treinamento de um modelo pré-treinado em um
dataset específico. O resultado é um modelo especializado que conhece profundamente
um domínio — no nosso caso, o conteúdo técnico do curso Especialista em IA.

Este submódulo cobre o ciclo completo em 2 notebooks:

| Notebook | Foco |
|---|---|
| `01_preparacao_dados.ipynb` | Gerar dataset sintético Q&A a partir dos AGENT_CONTEXT.md |
| `02_fine_tuning_openai.ipynb` | Executar fine-tuning via API e avaliar o modelo resultante |

---

## ⚠️ Limitação importante: DeepSeek não suporta fine-tuning via API

A DeepSeek API é compatível com OpenAI **apenas para inferência** (chat completions).
Os endpoints de fine-tuning (`/files`, `/fine_tuning/jobs`) **não existem** na DeepSeek —
qualquer chamada retorna erro 404.

| Tarefa | Provider usado | Motivo |
|---|---|---|
| Geração do dataset (synthetic data) | DeepSeek ✅ | É inferência normal |
| Fine-tuning gerenciado | OpenAI ⚠️ | Único provider configurado com suporte |

O notebook `02_fine_tuning_openai.ipynb` é uma **demonstração conceitual** — requer
`OPENAI_API_KEY` com créditos para execução. O dataset gerado é compatível com qualquer
plataforma que suporte fine-tuning gerenciado.

**Alternativas se não tiver OpenAI API:**
- **Together AI** / **Fireworks AI** — aceitam o mesmo formato JSONL, pagamento internacional
- **Fine-tuning local** — Unsloth + LoRA sobre pesos open-source do DeepSeek (requer GPU)

---

## Pré-requisitos

- Ambiente `eai07` ativo
- Submódulo `03_RAG` concluído (os AGENT_CONTEXT.md são o corpus do dataset)
- Para o notebook 02: `OPENAI_API_KEY` no `.env` com créditos disponíveis

---

## Configuração do `.env`

```env
# Já existente — usado para geração do dataset
DEEPSEEK_API_KEY=sk-...
LLM_MODEL=deepseek-chat

# Necessário apenas para o notebook 02
OPENAI_API_KEY=sk-...
```

---

## Estratégia: Synthetic Data Generation

Em vez de escrever exemplos manualmente, usamos o próprio LLM para gerar os dados:

```
26 AGENT_CONTEXT.md
        │
        ▼ chunking por seção (~1.500 chunks)
        │
        ▼ LLM gera 3 Q&A por chunk
        │
        ▼ ~4.500 pares pergunta/resposta
        │
        ▼ conversão para formato JSONL
        │
        ▼ split 90/10 estratificado por módulo
        │
   train.jsonl + val.jsonl
```

Cada chunk gera 3 perguntas com estilos diferentes:
- **Direta** — "O que é X?", "Como funciona Y?"
- **Aplicada** — "Quando usar X?", "Qual a diferença entre X e Y?"
- **Técnica** — "Qual o código para X?", "Quais os parâmetros de Y?"

---

## Arquivos gerados

```
data/finetune/
├── qa_gerado.jsonl    ← cache bruto da geração (com metadados)
├── train.jsonl        ← dataset de treino (~90%)
├── val.jsonl          ← dataset de validação (~10%)
└── job_ids.json       ← IDs persistidos (file_id, job_id, fine_tuned_model)
```

> **Nota:** A geração completa (`01_preparacao_dados.ipynb`) faz ~1.500 chamadas à API
> e leva entre 20 e 40 minutos. O cache salva progressivamente — se interrompido,
> retoma de onde parou na próxima execução.

---

## Fluxo do Fine-Tuning (notebook 02)

```
train.jsonl ──┐
               ├─→ Upload API ──→ file_id
val.jsonl ────┘         │
                         ▼
                   Criar Job ──→ job_id
                         │
                         ▼
                   Monitorar ──→ queued → running → succeeded
                         │
                         ▼
                   fine_tuned_model
                         │
                         ▼
                   Avaliação: base vs fine-tunado (LLM-as-judge)
```

Todos os IDs são persistidos em `job_ids.json` — a execução pode ser retomada
em qualquer etapa sem refazer uploads ou criar jobs duplicados.

---

## Conceitos-chave

**Synthetic data generation**  
Usar um LLM para criar exemplos de treinamento a partir de documentos existentes.
Eficiente quando o corpus já existe e escrever exemplos manualmente seria inviável.

**Formato JSONL para fine-tuning**  
Cada linha é um exemplo independente com `messages` (system / user / assistant).
É o formato padrão adotado por OpenAI, Together AI, Fireworks AI e Azure OpenAI.

**Split estratificado**  
Garantir que todos os módulos do curso aparecem tanto no treino quanto na validação.
Sem estratificação, módulos com menos chunks poderiam ficar só no treino ou só na validação.

**LLM-as-judge**  
Usar um LLM para avaliar automaticamente qual resposta é melhor entre base e fine-tunado.
Escala bem para grandes amostras, mas tem viés potencial (o juiz pode favorecer seu próprio estilo).

**Hiperparâmetros de fine-tuning**  
`n_epochs=3` é o padrão recomendado para fine-tuning supervisionado. `batch_size=auto` e
`learning_rate_multiplier=auto` deixam a plataforma ajustar baseado no tamanho do dataset.

---

## Referências

- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [OpenAI Fine-tuning Pricing](https://platform.openai.com/docs/pricing)
- [Together AI Fine-tuning](https://docs.together.ai/docs/fine-tuning)
- [Unsloth — Fine-tuning local eficiente](https://github.com/unslothai/unsloth)

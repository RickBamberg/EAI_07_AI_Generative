# AGENT_CONTEXT — 04_Fine_Tuning
# Submódulo do EAI_07_AI_Generative
# Para uso por agentes de IA. Versão legível: README.md

## IDENTIFICAÇÃO
- Submódulo: 04_Fine_Tuning
- Módulo pai: EAI_07_AI_Generative
- Ambiente: eai07 (Python 3.11, conda)
- Dependências: openai, python-dotenv, pathlib (stdlib)

## VISÃO GERAL
Fine-tuning supervisionado usando synthetic data gerada a partir dos AGENT_CONTEXT.md
do curso. O notebook 01 gera o dataset; o notebook 02 executa o fine-tuning via API.
Fine-tuning DeepSeek não é suportado via API — o notebook 02 usa OpenAI como demo conceitual.

## LIMITAÇÃO IMPORTANTE: DeepSeek não suporta fine-tuning via API
- A DeepSeek API (api.deepseek.com) é OpenAI-compatible APENAS para inferência
- Endpoints /files e /fine_tuning/jobs retornam 404 na DeepSeek
- Fine-tuning DeepSeek só é possível via pesos open-source localmente (Unsloth + LoRA)
  ou em plataformas terceiras: Together AI, Fireworks AI, Azure OpenAI
- O dataset gerado (train.jsonl / val.jsonl) é compatível com qualquer dessas plataformas

## NOTEBOOKS

### 01_preparacao_dados.ipynb
Geração de dataset sintético para fine-tuning a partir dos AGENT_CONTEXT.md do curso.

#### Coleta de corpus
- Fonte: todos os AGENT_CONTEXT.md do curso (EAI_01 a EAI_08)
- Chunking: por seções markdown (mesmo padrão do 03_RAG)
- Filtro: chunks com menos de 50 caracteres são descartados
- Resultado esperado: ~1.400-1.500 chunks aproveitáveis

#### Geração sintética (synthetic data generation)
- LLM usado: DeepSeek (deepseek-chat) — geração funciona normalmente
- Por chunk: 3 pares Q&A com estilos variados (direta, aplicada, técnica)
- Prompt retorna JSON: `{"pares": [{"pergunta": "...", "resposta": "..."}]}`
- Limpeza: strip de blocos ```json com regex antes do json.loads()
- Retry: até 3 tentativas por chunk em caso de JSON inválido
- Total esperado: ~4.500 pares Q&A

#### Cache progressivo
- Arquivo: `data/finetune/qa_gerado.jsonl`
- Salva linha a linha durante a geração — retoma de onde parou se interrompido
- Chave de deduplicação: `f"{modulo}||{titulo}"`
- Estimativa: 20-40 minutos para corpus completo com DELAY_ENTRE=0.3s

#### Formato fine-tuning
```json
{"messages": [
  {"role": "system",    "content": "<identidade do assistente>"},
  {"role": "user",      "content": "<pergunta>"},
  {"role": "assistant", "content": "<resposta>"}
]}
```

#### Validação
- Schema: messages não vazio, roles user e assistant presentes, content não vazio
- Tamanho: aviso para respostas < 20 chars ou > 2000 chars
- Duplicatas: contagem de perguntas idênticas
- Retorna dict: {total, erros, avisos, valido, resp_media, resp_min, resp_max, duplicatas}

#### Split estratificado
- Ratio: 90% treino / 10% validação
- Estratificação por módulo: todos os módulos aparecem nos dois splits
- Seed: 42 (reprodutível)
- Função: split_estratificado(dataset, qa_raw, val_ratio=0.1)

#### Arquivos gerados
```
data/finetune/
├── qa_gerado.jsonl   ← cache bruto com metadados (modulo, titulo, pergunta, resposta)
├── train.jsonl       ← formato fine-tuning, ~90% dos exemplos
└── val.jsonl         ← formato fine-tuning, ~10% dos exemplos
```

### 02_fine_tuning_deepseek.ipynb (versão original — NÃO FUNCIONA)
- Tentativa de fine-tuning via DeepSeek API
- Erro: NotFoundError 404 em llm.files.create() — endpoint não existe na DeepSeek
- Arquivo mantido como registro do problema encontrado

### 02_fine_tuning_openai.ipynb (demo conceitual — requer OPENAI_API_KEY)
Fine-tuning gerenciado via OpenAI API. Não executável sem créditos OpenAI,
mantido como referência do fluxo completo.

#### Persistência de IDs
- Arquivo: `data/finetune/job_ids.json`
- Salva: train_file_id, val_file_id, job_id, fine_tuned_model
- Comportamento: não refaz upload nem cria job duplicado se IDs já existem
- Permite retomar em qualquer etapa entre sessões

#### Upload
```python
llm.files.create(file=f, purpose='fine-tune')  # retorna file_id
```

#### Criação do job
```python
llm.fine_tuning.jobs.create(
    training_file   = train_file_id,
    validation_file = val_file_id,
    model           = 'gpt-4o-mini',
    hyperparameters = {'n_epochs': 3, 'batch_size': 'auto', 'learning_rate_multiplier': 'auto'}
)  # retorna job_id
```

#### Monitoramento
```python
llm.fine_tuning.jobs.retrieve(job_id)
# .status: queued → running → succeeded | failed | cancelled
# .fine_tuned_model: preenchido quando succeeded
# .trained_tokens: total de tokens usados no treino
```
- Loop com intervalo de 60s, timeout de 2h
- Imprime apenas quando o status muda (não polui output)

#### Uso do modelo fine-tunado
- Mesmo client OpenAI, apenas model=fine_tuned_model no lugar de 'gpt-4o-mini'
- fine_tuned_model ID formato: `ft:gpt-4o-mini-...:sufixo`

#### Avaliação: LLM-as-judge
- Amostra: 10 perguntas aleatórias do val.jsonl (seed=42)
- Gera respostas dos dois modelos (base + fine-tunado) para cada pergunta
- Juiz: mesmo LLM base avalia qual resposta é melhor (vencedor: 'base'|'ft'|'empate')
- Saída: placar final com percentuais
- Limitação documentada: juiz pode ter viés por favorecer seu próprio estilo

## PADRÕES DE CÓDIGO
```python
# Inicialização — notebook 01 (geração usa DeepSeek)
llm = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com')
LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-chat')
DATA_DIR  = Path('../data/finetune')

# Inicialização — notebook 02 (fine-tuning usa OpenAI)
llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # sem base_url
LLM_MODEL = 'gpt-4o-mini'
```

## FAQ
Q: Por que o 01_preparacao_dados usa DeepSeek mas o 02_fine_tuning usa OpenAI?
A: São tarefas diferentes. Geração de synthetic data é inferência normal — DeepSeek funciona.
   Fine-tuning gerenciado via API requer endpoints /files e /fine_tuning/jobs que só a OpenAI
   (e plataformas compatíveis) implementam.

Q: O dataset gerado serve para outras plataformas além de OpenAI?
A: Sim. O formato JSONL com messages (system/user/assistant) é o padrão adotado por
   Together AI, Fireworks AI, Anyscale e Azure OpenAI. Só precisa trocar o client.

Q: Por que usar synthetic data em vez de exemplos manuais?
A: O corpus (AGENT_CONTEXT.md) já existe e tem boa qualidade técnica. Escrever ~4.500
   exemplos manualmente seria inviável. O LLM gera variações de estilo (direta/aplicada/técnica)
   que aumentam a diversidade do dataset automaticamente.

Q: Quanto custa executar o fine-tuning na OpenAI?
A: gpt-4o-mini fine-tuning: ~$3-8 por 1M tokens de treino (2025). Com ~4.500 exemplos
   de tamanho médio, estimativa de $5-15 USD para 3 épocas. Verificar pricing atual em
   platform.openai.com/docs/guides/fine-tuning.

Q: Como retomar se a geração do notebook 01 foi interrompida?
A: Basta reexecutar. O cache qa_gerado.jsonl é lido no início e os chunks já processados
   são pulados via deduplicação por chave "modulo||titulo".

## TAGS DE BUSCA
fine-tuning fine tuning supervised learning dataset JSONL synthetic data generation
LLM-as-judge DeepSeek OpenAI gpt-4o-mini upload file_id job_id fine_tuned_model
train.jsonl val.jsonl split estratificado validação schema Q&A pares pergunta resposta
hyperparameters n_epochs batch_size learning_rate_multiplier monitoramento job status

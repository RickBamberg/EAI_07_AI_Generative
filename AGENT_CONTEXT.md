# AGENT_CONTEXT — EAI_07_AI_Generative
# Módulo do curso Especialista em IA
# Para uso por agentes de IA. Versão legível: README.md

## IDENTIFICAÇÃO
- Módulo: EAI_07_AI_Generative
- Curso: Especialista em IA (EAI_01 a EAI_08+)
- Ambiente: eai07 (conda, Python 3.11)
- Provider LLM: DeepSeek (primary), configurável via .env
- Repositório GitHub: RickBamberg

## VISÃO GERAL
Módulo de IA Generativa do curso. Cobre modelos pré-treinados, RAG básico e avançado,
RAG para código, function calling, modelos locais e fine-tuning. Arquitetura
provider-agnostic: trocar o LLM exige apenas mudar variáveis no .env.

## ESTRUTURA DE DIRETÓRIOS
```
EAI_07_AI_Generative/
├── .env                          ← chaves de API e configuração do provider
├── shared/
│   ├── llm_factory.py            ← cliente LLM unificado (chat, chat_stream)
│   └── tool_runner.py            ← execução de function calling
├── data/
│   └── cache/
│       ├── indice_rag.pkl        ← cache do índice RAG básico (5.4 MB)
│       └── indice_codigo.pkl     ← cache do índice RAG código
├── 01_Modelos_Pre_Treinados/
│   └── AGENT_CONTEXT.md
├── 02_Function_Calling/
│   └── AGENT_CONTEXT.md
├── 03_RAG/
│   ├── 03_rag_basico.ipynb
│   ├── 04_rag_avancado.ipynb
│   ├── 05_rag_codigo_especializado.ipynb
│   └── AGENT_CONTEXT.md
├── 04_Fine_Tuning/
│   ├── 01_preparacao_dados.ipynb
│   └── 02_fine_tuning_deepseek.ipynb
└── 05_Modelos_Locais/
    └── AGENT_CONTEXT.md
```

## CONFIGURAÇÃO DO AMBIENTE

### Arquivo .env (raiz do módulo)
```
DEEPSEEK_API_KEY=sk-...
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
```

### Criação do ambiente conda
```bash
conda create -n eai07 python=3.11 -y
conda activate eai07
pip install openai python-dotenv sentence-transformers faiss-cpu numpy
```

## SHARED — UTILITÁRIOS COMPARTILHADOS

### llm_factory.py
Ponto único de acesso ao LLM em todos os notebooks do módulo.
```python
from shared.llm_factory import chat, chat_stream, get_provider_info

# Chamada simples
resposta = chat([{'role': 'user', 'content': 'Olá'}])

# Streaming
for chunk in chat_stream([{'role': 'user', 'content': 'Olá'}]):
    print(chunk, end='', flush=True)

# Info do provider ativo
info = get_provider_info()  # {'provider': 'deepseek', 'model': 'deepseek-chat', ...}
```

### tool_runner.py
Executa function calling com fallback para respostas não-padrão do DeepSeek.
- Loop de até 8 iterações para tool use
- Regex com ordem correta para substituição de parâmetros
- Mapeamento de aliases de parâmetros (DeepSeek usa nomes diferentes em alguns casos)
- Formato DSML como fallback quando a resposta não segue JSON padrão

## SUBMÓDULOS

### 01_Modelos_Pre_Treinados
Uso de modelos pré-treinados via Hugging Face e API.
Ver: `01_Modelos_Pre_Treinados/AGENT_CONTEXT.md`

### 02_Function_Calling
Function calling / tool use com DeepSeek.
Ver: `02_Function_Calling/AGENT_CONTEXT.md`

### 03_RAG
Pipeline RAG completo em 5 notebooks progressivos.

#### Notebooks
| Notebook | Técnicas principais |
|---|---|
| 03_rag_basico | Embedding, FAISS IndexFlatIP, chunking por seção, enriquecimento |
| 04_rag_avancado | Cache disco, filtro módulo, query expansion, reranking, histórico |
| 05_rag_codigo_especializado | AST chunking, índice híbrido, BM25, busca híbrida (α) |

#### Stack técnica
- Embedding: all-MiniLM-L6-v2 (384 dim, sentence-transformers)
- Índice: faiss.IndexFlatIP com normalize_embeddings=True
- Corpus: 26 AGENT_CONTEXT.md → 1.553 chunks
- BM25: implementação from scratch (re, math, Counter)
- Score híbrido: `α × semântico + (1-α) × BM25`, α=0.7 padrão

#### Estrutura de chunk (padrão unificado)
```python
{
    'chunk_busca'   : str,   # curto + enriquecido → embedding/BM25
    'chunk_contexto': str,   # completo → contexto do LLM
    'titulo'        : str,
    'modulo'        : str,   # 'EAI_01_Fundamentos_Matemática_para_IA'
    'tipo'          : str,   # 'doc' | 'codigo'
    'tipo_chunk'    : str,   # 'seção' | 'função' | 'classe' | 'módulo'
    'arquivo'       : str,   # caminho relativo ao PROJETO_BASE
    'assinatura'    : str,
    'docstring'     : str,
    'linhas'        : tuple,
}
```

### 04_Fine_Tuning
Fine-tuning via API DeepSeek.

#### 01_preparacao_dados.ipynb
Preparação do dataset para fine-tuning no formato JSONL.
- Formato: `{"messages": [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]}`
- Validação de schema antes do upload
- Split treino/validação

#### 02_fine_tuning_deepseek.ipynb
Fine-tuning via DeepSeek API.
- Upload do dataset, criação do job, monitoramento
- Avaliação do modelo fine-tunado vs base
- Provider: DeepSeek fine-tuning API

### 05_Modelos_Locais
Execução de LLMs localmente (Ollama).
Ver: `05_Modelos_Locais/AGENT_CONTEXT.md`

## PADRÕES DO MÓDULO

### Inicialização padrão dos notebooks RAG
```python
import sys, os
sys.path.append(os.path.abspath('..'))
from dotenv import load_dotenv
load_dotenv('../.env')

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

modelo_emb   = SentenceTransformer('all-MiniLM-L6-v2')
llm          = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com')
LLM_MODEL    = os.getenv('LLM_MODEL', 'deepseek-chat')
PROJETO_BASE = os.path.abspath('../..')   # raiz do curso
EAI07_BASE   = os.path.abspath('..')     # raiz do EAI_07
```

### Variáveis de ambiente
| Variável | Uso |
|---|---|
| DEEPSEEK_API_KEY | Autenticação DeepSeek |
| LLM_MODEL | Modelo ativo (deepseek-chat, deepseek-coder, etc.) |
| LLM_BASE_URL | URL da API (permite trocar provider) |

## FAQ
Q: Como trocar o provider LLM?
A: Editar o .env: mudar API_KEY, LLM_MODEL e LLM_BASE_URL. O llm_factory abstrai o resto.

Q: Por que DeepSeek e não OpenAI?
A: Dificuldade de adicionar crédito OpenAI do Brasil. DeepSeek aceita pagamento internacional e tem API compatível com OpenAI SDK.

Q: Os notebooks são independentes?
A: Cada notebook copia as funções necessárias para ser executável isoladamente. Cache em disco é o único estado persistido entre sessões.

Q: Onde fica o cache dos índices RAG?
A: `EAI_07_AI_Generative/data/cache/`. indice_rag.pkl (RAG básico/avançado) e indice_codigo.pkl (RAG código).

Q: Como usar o shared/ a partir de um notebook?
A: `sys.path.append(os.path.abspath('..'))` antes do import. O `..` aponta para a raiz do EAI_07 onde fica o shared/.

## TAGS DE BUSCA
EAI_07 IA generativa LLM DeepSeek OpenAI provider-agnostic llm_factory tool_runner
function calling RAG retrieval augmented generation embeddings FAISS BM25 busca híbrida
fine-tuning modelos locais Ollama sentence-transformers all-MiniLM-L6-v2
eai07 conda Python 3.11 ambiente generative AI

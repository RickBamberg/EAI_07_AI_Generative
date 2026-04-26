# AGENT_CONTEXT — Assistente_Tecnico_IA
# Projeto do EAI_07_AI_Generative / 06_Projetos_Reais
# Para uso por agentes de IA. Versão legível: README.md

## IDENTIFICAÇÃO
- Projeto: Assistente_Tecnico_IA
- Módulo pai: EAI_07_AI_Generative / 06_Projetos_Reais
- Ambiente: eai07 (Python 3.11, conda)
- Dependências base: flask, openai, python-dotenv, sentence-transformers, numpy
- Dependências v1: faiss-cpu
- Dependências v2: chromadb
- Dependências v3: langchain, langchain-community, langchain-chroma, langchain-openai, langchain-huggingface

## VISÃO GERAL
Aplicação Flask com interface de chat que responde perguntas sobre os módulos
EAI_01 a EAI_08 usando RAG semântico sobre os AGENT_CONTEXT.md do curso.
Projeto integrador — usa o shared/llm_factory do EAI_07 e padrões de memória do 05_Agentes.
Existe em três versões de backend RAG:
- v1/assistente.py: índice FAISS + pkl (1.553 chunks, 26 arquivos)
- v2/assistente.py: banco ChromaDB persistente (1.763 chunks, 33 arquivos, upsert incremental)
- v3/assistente.py: banco ChromaDB via LangChain 1.x / LCEL (mesmo banco da v2, nova camada de código)
O app.py fica na raiz e importa de v1/, v2/ ou v3/ conforme a linha de import ativa.

## ARQUIVOS

### app.py  (raiz do projeto — idêntico para v1, v2 e v3)
Servidor Flask com 4 rotas:
- GET  /        → serve templates/index.html
- POST /chat    → recebe {pergunta: str}, retorna {resposta, fontes, historico, timestamp}
- POST /limpar  → apaga data/historico_global.json
- GET  /status  → retorna {status, pronto, rag, chunks}

Importa o assistente pela linha ativa (escolher uma antes de iniciar):
```python
from v1.assistente import responder, limpar_historico, _get_indice, _pronto  # FAISS
from v2.assistente import responder, limpar_historico, _get_indice, _pronto  # ChromaDB
from v3.assistente import responder, limpar_historico, _get_indice, _pronto  # LangChain
```

### v1/assistente.py  (FAISS + pkl)
Núcleo v1. Fluxo por pergunta:
1. _precisa_rag(pergunta) → decide se busca RAG ou responde direto do system
2. buscar_rag(query, top_k=5, score_min=0.45) → FAISS sobre indice_rag.pkl (1.553 chunks)
3. _carregar_historico() → últimas 20 msgs do historico_global.json
4. llm.chat.completions → DeepSeek com system + histórico + contexto RAG
5. _salvar_historico() → persiste turno
6. retorna {resposta, fontes, historico, timestamp}

### v2/assistente.py  (ChromaDB puro)
Núcleo v2. Fluxo por pergunta:
1. _precisa_rag(pergunta) → decide se busca RAG ou responde direto do system
2. _expandir_query(pergunta, historico[-4:]) → LLM reformula com contexto de histórico
3. buscar_rag(query, top_k=5, score_min=0.45) → ChromaDB (1.763 chunks, 33 arquivos)
4. _carregar_historico() → últimas 20 msgs do historico_global.json
5. llm.chat.completions → DeepSeek com system + histórico + contexto RAG
6. _salvar_historico() → persiste turno
7. retorna {resposta, fontes, historico, timestamp}  (+ campo 'arquivo' nas fontes)

### v3/assistente.py  (LangChain 1.x / LCEL)
Núcleo v3. Mesmo banco ChromaDB da v2 — apenas a camada de código Python muda.
Fluxo por pergunta:
1. _precisa_rag(pergunta) → idêntico à v2
2. _expandir_query(pergunta, historico[-4:]) → LCEL: PromptTemplate | ChatOpenAI | StrOutputParser
3. buscar_rag(query, top_k=5, score_min=0.45) → vectorstore.similarity_search_with_score()
4. _carregar_historico() → idêntico à v2
5. rag_chain.invoke() → LCEL: ChatPromptTemplate | ChatOpenAI | StrOutputParser
6. _salvar_historico() → idêntico à v2
7. retorna {resposta, fontes, historico, timestamp}  (interface idêntica à v2)

#### Diferenças internas v2 → v3

| Componente | v2 (ChromaDB puro) | v3 (LangChain 1.x) |
|---|---|---|
| Vectorstore | `chromadb.PersistentClient` + `collection` | `Chroma(persist_directory=...)` |
| Embeddings | `SentenceTransformer.encode()` manual | `HuggingFaceEmbeddings` (interno ao vectorstore) |
| LLM | `openai.OpenAI().chat.completions.create()` | `ChatOpenAI` + LCEL chains |
| Query Expansion | `_llm.chat.completions.create(messages=[...])` | `expansion_chain.invoke({'pergunta':..., 'ctx_bloco':...})` |
| Busca | `collection.query(query_embeddings=emb_q.tolist(), ...)` | `vs.similarity_search_with_score(query, filter=...)` |
| Histórico no LLM | `[{'role':..., 'content':...}]` | `[HumanMessage(...) \| AIMessage(...)]` |
| `_get_modelo_emb()` | presente (retorna SentenceTransformer) | **removida** — vectorstore gera embedding internamente |
| Banco em disco | `data/chroma_db/` | **mesmo** `data/chroma_db/` — sem reindexação |

#### Imports LangChain 1.x (v3)
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
```

#### LCEL chains construídas no pré-carregamento (v3)
```python
# Query Expansion — temperatura 0 e max_tokens curto (só precisa de uma linha)
_expansion_chain = PromptTemplate.from_template("...{pergunta}...{ctx_bloco}...") \
                   | ChatOpenAI(temperature=0.0, max_tokens=120) \
                   | StrOutputParser()

# RAG principal — system + histórico via MessagesPlaceholder + pergunta do usuário
_rag_chain = ChatPromptTemplate.from_messages([
    ('system', '{system}'),
    MessagesPlaceholder(variable_name='historico'),
    ('human', '{prompt_usuario}'),
]) | _llm | StrOutputParser()
```

#### Busca via LangChain (v3)
```python
# similarity_search_with_score retorna list[tuple[Document, float]]
# float = distância cosine ChromaDB [0, 2] → score = 1 - (dist / 2.0)
pares = vs.similarity_search_with_score(
    query_expandida, k=top_k,
    filter={'modulo_prefixo': {'$eq': filtro}}   # mesma sintaxe $eq do ChromaDB puro
)
for doc, dist in pares:
    score    = round(1.0 - (dist / 2.0), 3)
    contexto = doc.metadata.get('_contexto', doc.page_content)   # fallback compatível com v2
```

#### Histórico convertido para objetos LangChain (v3)
```python
# v2: [{'role': m['role'], 'content': m['content']} for m in historico]
# v3: _historico_para_mensagens(historico) → [HumanMessage | AIMessage]
def _historico_para_mensagens(historico: list) -> list:
    msgs = []
    for m in historico:
        if m['role'] == 'user':
            msgs.append(HumanMessage(content=m['content']))
        elif m['role'] == 'assistant':
            msgs.append(AIMessage(content=m['content']))
    return msgs
```

#### Compatibilidade do banco entre v2 e v3
- Banco `data/chroma_db/` é **compartilhado**: v2 e v3 apontam para o mesmo diretório
- Banco gerado pelo notebook ChromaDB puro: documentos **sem** campo `_contexto` nos metadados
- Banco gerado pelo notebook LangChain: documentos **com** `_contexto` como metadado extra
- `buscar_rag()` da v3 usa `doc.metadata.get('_contexto', doc.page_content)` — fallback garante compatibilidade com ambos os formatos

#### Pré-carregamento em background (v3)
- `_precarregar()`: thread daemon inicializa `_embedding_function`, `_vectorstore`, `_llm`, `_expansion_chain`, `_rag_chain`
- `_get_vectorstore()`: aguarda com `time.sleep(0.5)` até 30s
- `_get_chains()`: retorna `(expansion_chain, rag_chain)` após o pré-carregamento
- `_get_indice()`: mantido para compatibilidade com `app.py` — retorna o vectorstore LangChain
- `_pronto`: bool — True quando todos os componentes estão carregados

#### Resolução de caminhos (v3 — idêntica à v2)
```
EAI_07_AI_Generative/           ← _HERE.parent.parent.parent
├── shared/llm_factory.py
├── data/chroma_db/             ← mesmo banco da v2
└── 06_Projetos_Reais/
    └── Assistente_Tecnico_IA/
        └── v3/
            └── assistente.py   ← _HERE
```

#### Interface pública preservada (v3 idêntica à v2)
- `responder(pergunta)` → mesmo retorno `{resposta, fontes, historico, timestamp}`
- `limpar_historico()` → idêntico
- `_precisa_rag()` → idêntico
- `_carregar_historico()` / `_salvar_historico()` → idênticos
- `_MAX_HISTORICO = 20` → idêntico
- `score_min=0.45` → idêntico
- Lógica CONTINUAÇÃO / INDEPENDENTE no query expansion → preservada, só muda a chamada ao LLM

## COMPONENTES COMPARTILHADOS ENTRE AS TRÊS VERSÕES

### Resolução de caminhos (automática — idêntica em v1, v2, v3)
Cada assistente.py sobe a hierarquia procurando `shared/llm_factory.py`:
```python
for _candidate in [_HERE, _HERE.parent, _HERE.parent.parent, _HERE.parent.parent.parent]:
    if (_candidate / 'shared' / 'llm_factory.py').exists():
        sys.path.insert(0, str(_candidate))
        break
```

### _precisa_rag() (v2 e v3)
Detecta perguntas gerais (lista de módulos, estrutura) e não envia contexto RAG.
Palavras-chave: 'módulos', 'modulos', 'lista', 'estrutura', 'visão geral', etc.
Exceção: se a pergunta também contém 'como', 'código', 'função', 'algoritmo' → usa RAG.

### Score mínimo RAG (v1, v2, v3)
score_min=0.45 — chunks com score abaixo não são enviados ao LLM.
v2/v3: score convertido de distância cosine ChromaDB → `1 - (dist / 2.0)`, mesma escala [0,1].

### Query Expansion com consciência de histórico (v2 e v3)
Antes de buscar, o LLM reformula a pergunta em dois modos:
- CONTINUAÇÃO: resolve pronomes ("desse projeto", "nele") usando historico[-4:]
- INDEPENDENTE: ignora histórico quando a pergunta menciona explicitamente novo módulo/tema
Evita contaminação entre módulos em perguntas como "e no EAI_06, como foi avaliado?".

### System prompt (v1, v2, v3 — idêntico)
Contém:
- Estrutura completa dos 8 módulos com tópicos principais (hardcoded)
- Regras: perguntas gerais → responde direto | perguntas técnicas → usa RAG
- Info de provider: como editar .env para trocar DeepSeek/OpenAI/Ollama
- Nunca mencionar "não está no contexto" para perguntas sobre estrutura do curso

### Memória global (v1, v2, v3 — idêntica)
- Arquivo: data/historico_global.json
- Formato: lista de {role, content, timestamp}
- Janela: 20 msgs (10 turns) no contexto — salva tudo mas envia só as últimas 20
- Sem distinção de usuário — histórico único compartilhado

### templates/index.html (idêntico para todas as versões)
Interface de chat em HTML/CSS/JS puro (sem framework).
- Estética: dark mode, IBM Plex Mono + Sans, verde terminal (#4ade80)
- Chat à esquerda com renderização Markdown (código, listas, bold, headers)
- Painel lateral direito: chunks RAG com módulo, score%, título, trecho expansível
- Polling de status a cada 2s — input desabilitado até _pronto=True
- Perguntas rápidas clicáveis na tela inicial
- Botão "Limpar histórico" no rodapé do painel

## LIMITAÇÃO CONHECIDA
O índice RAG é construído sobre AGENT_CONTEXT.md — não sobre arquivos .py.
Perguntas sobre código específico trazem resposta correta conceitualmente mas com código
gerado pelo LLM, não o código real. Para indexar código real, usar o indice_codigo.pkl
do 05_rag_codigo_especializado.ipynb.

## COMO EXECUTAR
```bash
conda activate eai07
pip install flask

# Instalar dependências da versão desejada:
pip install chromadb                                                                            # v2
pip install langchain langchain-community langchain-chroma langchain-openai langchain-huggingface  # v3

cd EAI_07_AI_Generative/06_Projetos_Reais/Assistente_Tecnico_IA

# Edite o import no app.py para a versão desejada:
# from v1.assistente import ...   ← FAISS + pkl
# from v2.assistente import ...   ← ChromaDB puro
# from v3.assistente import ...   ← LangChain 1.x (mesmo banco da v2)

# Certifique-se que o banco da versão escolhida existe:
# v1: data/cache/indice_rag.pkl          (03_rag_basico.ipynb)
# v2: data/chroma_db/                    (04_rag_avancado_chromadb.ipynb)
# v3: data/chroma_db/                    (mesmo da v2 — sem reindexação)

python app.py
# Acesse: http://localhost:5000
```

## REINDEXAÇÃO

**v1 — rebuild completo:**
1. Delete `EAI_07_AI_Generative/data/cache/indice_rag.pkl`
2. Reexecute `03_RAG/03_rag_basico.ipynb` completo (~67s)
3. Reinicie o servidor Flask

**v2/v3 — incremental (só o módulo alterado):**
```python
# No 04_rag_avancado_chromadb.ipynb ou 04_rag_avancado_langchain.ipynb
atualizar_modulo(PROJETO_BASE, 'EAI_09')   # reindexará apenas o EAI_09 (~5s)
```

**v2/v3 — rebuild completo:**
```python
resetar_banco(confirmar=True)   # apaga tudo e reconstrói (~110s)
```

## FAQ
Q: Como escolher entre v1, v2 e v3?
A: Editar a linha de import no app.py:
   `from v1.assistente import ...`  → FAISS + pkl
   `from v2.assistente import ...`  → ChromaDB puro
   `from v3.assistente import ...`  → LangChain 1.x (mesmo banco da v2)
   O resto da aplicação (Flask, templates, histórico) é idêntico entre as três versões.

Q: Qual a diferença entre v2 e v3?
A: Mesmo banco ChromaDB, mesma lógica e mesma interface pública. A diferença é na camada
   de código: v3 usa LangChain 1.x (Chroma, HuggingFaceEmbeddings, ChatOpenAI, LCEL chains)
   em vez de chromadb e openai diretamente. O objetivo da v3 é didático.

Q: Preciso reindexar ao trocar da v2 para a v3?
A: Não. O banco `data/chroma_db/` é compartilhado. A v3 abre o mesmo diretório via LangChain
   sem nenhuma migração de dados.

Q: O banco da v3 funciona se foi gerado pelo notebook ChromaDB puro?
A: Sim. `buscar_rag()` da v3 usa `doc.metadata.get('_contexto', doc.page_content)`.
   Se não há campo `_contexto` (banco gerado pela v2/notebook ChromaDB puro), usa page_content
   como fallback. Se há `_contexto` (banco gerado pelo notebook LangChain), usa o texto completo.

Q: O que fazer se o assistente mostrar "índice indisponível" no status?
A: v1: executar 03_rag_basico.ipynb.
   v2/v3: executar 04_rag_avancado_chromadb.ipynb ou 04_rag_avancado_langchain.ipynb.

Q: Por que usar LangChain 1.x em vez de 0.x?
A: LangChain 0.x está descontinuado. Na versão 1.x, `langchain.memory` e `langchain.chains`
   foram removidos. Os substitutos são LCEL (`prompt | llm | parser`) para pipelines e
   `ChatMessageHistory` + `RunnableWithMessageHistory` para histórico gerenciado pelo framework.
   Na v3 do assistente, o histórico ainda é gerenciado manualmente em JSON (compatibilidade
   com v1/v2), por isso usa `_historico_para_mensagens()` em vez de `RunnableWithMessageHistory`.

Q: Por que v2/v3 têm mais chunks que v1 (1.763 vs 1.553)?
A: O v1 foi indexado com 26 AGENT_CONTEXT.md existentes na época. O v2/v3 varrem todas as
   subpastas recursivamente e encontram 33 arquivos — inclui AGENT_CONTEXT.md em subpastas
   do EAI_07 criados depois da indexação original.

Q: Por que o índice é compartilhado e não copiado para dentro do projeto?
A: O assistente.py sobe a hierarquia até encontrar o EAI_07 onde o cache/chroma_db já existe.
   Copiar seria redundante e desincronizaria ao reindexar.

Q: Como expandir o assistente para responder sobre código real?
A: v1: trocar indice_rag.pkl por indice_codigo.pkl (notebook 05).
   v2/v3: criar uma segunda collection ChromaDB com o índice híbrido do notebook 05.

## TAGS DE BUSCA
Flask chat RAG semântico FAISS ChromaDB LangChain LCEL assistente técnico EAI_07
provider troca .env threading background lazy load polling status
IBM Plex dark mode painel fontes chunks score mínimo system prompt
histórico global memória conversacional indice_rag.pkl chroma_db
v1 v2 v3 query expansion upsert incremental import switch versão backend
Chroma HuggingFaceEmbeddings ChatOpenAI PromptTemplate ChatPromptTemplate
MessagesPlaceholder StrOutputParser HumanMessage AIMessage LCEL pipe operator
langchain_core langchain_community langchain_chroma langchain_openai langchain_huggingface
similarity_search_with_score _contexto metadado fallback compatibilidade banco compartilhado

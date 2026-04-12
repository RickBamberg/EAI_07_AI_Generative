# AGENT_CONTEXT — Assistente_Tecnico_IA
# Projeto do EAI_07_AI_Generative / 06_Projetos_Reais
# Para uso por agentes de IA. Versão legível: README.md

## IDENTIFICAÇÃO
- Projeto: Assistente_Tecnico_IA
- Módulo pai: EAI_07_AI_Generative / 06_Projetos_Reais
- Ambiente: eai07 (Python 3.11, conda)
- Dependências: flask, openai, python-dotenv, faiss-cpu, sentence-transformers, numpy

## VISÃO GERAL
Aplicação Flask com interface de chat que responde perguntas sobre os módulos
EAI_01 a EAI_08 usando RAG semântico sobre os AGENT_CONTEXT.md do curso.
Projeto integrador — usa o índice FAISS do 03_RAG, o shared/llm_factory do EAI_07
e padrões de memória do 05_Agentes.

## ARQUIVOS

### app.py
Servidor Flask com 4 rotas:
- GET  /        → serve templates/index.html
- POST /chat    → recebe {pergunta: str}, retorna {resposta, fontes, historico, timestamp}
- POST /limpar  → apaga data/historico_global.json
- GET  /status  → retorna {status, pronto, rag, chunks}

Importa de assistente.py: responder(), limpar_historico(), _get_indice(), _pronto

### assistente.py
Núcleo do sistema. Fluxo por pergunta:
1. _precisa_rag(pergunta) → decide se busca RAG ou responde direto do system
2. buscar_rag(pergunta, top_k=5, score_min=0.45) → chunks semânticos do FAISS
3. _carregar_historico() → últimas 20 msgs do historico_global.json
4. llm.chat.completions → DeepSeek com system + histórico + contexto RAG
5. _salvar_historico() → persiste turno
6. retorna {resposta, fontes, historico, timestamp}

#### Resolução de caminhos (automática)
```
EAI_07_AI_Generative/           ← _HERE.parent.parent
├── shared/llm_factory.py       ← detectado subindo hierarquia
├── data/cache/indice_rag.pkl   ← índice compartilhado do 03_RAG
└── 06_Projetos_Reais/
    └── Assistente_Tecnico_IA/
        └── assistente.py       ← _HERE
```
Nada precisa ser copiado — usa os arquivos originais do projeto.

#### Carregamento em background (threading)
- _precarregar(): thread daemon carrega embedding + índice ao importar o módulo
- Flask sobe imediatamente, página abre antes do modelo estar pronto
- _pronto: bool — True quando embedding + índice estão carregados
- _get_modelo_emb() e _get_indice(): aguardam com time.sleep(0.5) até 30s
- Frontend faz polling em /status a cada 2s e habilita o input quando pronto

#### _precisa_rag()
Detecta perguntas gerais (lista de módulos, estrutura) e não envia contexto RAG.
Palavras-chave: 'módulos', 'modulos', 'lista', 'estrutura', 'visão geral', etc.
Exceção: se a pergunta também contém 'como', 'código', 'função', 'algoritmo' → usa RAG.

#### Score mínimo RAG
score_min=0.45 — chunks com score abaixo não são enviados ao LLM.
Evita que chunks irrelevantes confundam a resposta (problema original com score 0.3).

#### System prompt
Contém:
- Estrutura completa dos 8 módulos com tópicos principais (hardcoded)
- Regras: perguntas gerais → responde direto | perguntas técnicas → usa RAG
- Info de provider: como editar .env para trocar DeepSeek/OpenAI/Ollama
- Nunca mencionar "não está no contexto" para perguntas sobre estrutura do curso

#### Memória global
- Arquivo: data/historico_global.json
- Formato: lista de {role, content, timestamp}
- Janela: 20 msgs (10 turns) no contexto — salva tudo mas envia só as últimas 20
- Sem distinção de usuário — histórico único compartilhado
- Criado automaticamente na primeira pergunta

### templates/index.html
Interface de chat em HTML/CSS/JS puro (sem framework).
- Estética: dark mode, IBM Plex Mono + Sans, verde terminal (#4ade80)
- Chat à esquerda com renderização Markdown (código, listas, bold, headers)
- Painel lateral direito: chunks RAG com módulo, score%, título, trecho expansível
- Polling de status a cada 2s — input desabilitado até _pronto=True
- Perguntas rápidas clicáveis na tela inicial
- Botão "Limpar histórico" no rodapé do painel

## LIMITAÇÃO CONHECIDA
O índice RAG é construído sobre AGENT_CONTEXT.md — não sobre arquivos .py.
Perguntas sobre código específico (ex: "me mostre o código do _parse_dsml") trazem
resposta correta conceitualmente mas com código gerado pelo LLM, não o código real.
Para indexar código real, usar o indice_codigo.pkl do 05_rag_codigo_especializado.ipynb.

## COMO EXECUTAR
```bash
conda activate eai07
pip install flask
cd EAI_07_AI_Generative/06_Projetos_Reais/Assistente_Tecnico_IA
python app.py
# Acesse: http://localhost:5000
```

## REINDEXAÇÃO
Necessária quando novos módulos com AGENT_CONTEXT.md forem criados:
1. Delete EAI_07_AI_Generative/data/cache/indice_rag.pkl
2. Reexecute 03_RAG/03_rag_basico.ipynb completo (~67s)
3. Reinicie o servidor Flask

## FAQ
Q: Por que o índice é compartilhado e não copiado para dentro do projeto?
A: _HERE.parent.parent aponta para a raiz do EAI_07 onde o cache já existe.
   Copiar seria redundante e desincronizaria ao reindexar.

Q: O que fazer se o assistente mostrar "índice indisponível" no status?
A: Gerar o índice executando o 03_rag_basico.ipynb. O cache deve existir em
   EAI_07_AI_Generative/data/cache/indice_rag.pkl antes de iniciar o Flask.

Q: Por que a primeira pergunta demora mais?
A: O pré-carregamento em background começa ao importar o módulo. Se a pergunta
   chegar antes de _pronto=True, aguarda até 30s. O frontend impede isso com polling
   — o input só fica ativo depois que _pronto=True.

Q: Como expandir o assistente para responder sobre código real?
A: Trocar o cache de indice_rag.pkl por indice_codigo.pkl (gerado no notebook 05)
   que inclui chunks de .py extraídos por AST além dos AGENT_CONTEXT.md.

## TAGS DE BUSCA
Flask chat RAG semântico FAISS assistente técnico EAI_07 provider troca .env
threading background lazy load polling status IBM Plex dark mode painel fontes chunks
score mínimo system prompt histórico global memória conversacional indice_rag.pkl

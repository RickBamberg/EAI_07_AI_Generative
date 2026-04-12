# AGENT_CONTEXT — 05_Agentes
# Submódulo do EAI_07_AI_Generative
# Para uso por agentes de IA. Versão legível: README.md

## IDENTIFICAÇÃO
- Submódulo: 05_Agentes
- Módulo pai: EAI_07_AI_Generative
- Ambiente: eai07 (Python 3.11, conda)
- Dependências: openai, python-dotenv, requests, beautifulsoup4

## VISÃO GERAL
Implementação progressiva de agentes autônomos em 4 notebooks, do padrão ReAct básico
até gerenciamento de memória conversacional. Toda orquestração usa o ToolRunner do
shared/tool_runner.py que trata respostas padrão OpenAI e fallback DSML do DeepSeek.

## NOTEBOOKS

### 01_agente_simples.ipynb — Padrão ReAct
Implementação do ciclo Reason → Act → Observe usando ToolRunner do shared/.

#### Ferramentas
- calcular: eval com whitelist de funções math (sqrt, log, sin, cos, pi, e)
- data_hora_atual: datetime.now() com formato strftime configurável
- consultar_curso: busca por keywords nos AGENT_CONTEXT.md (mínimo 2 palavras match)

#### Loop ReAct
- Usa ToolRunner.registrar(schema, funcao) + ToolRunner.perguntar(pergunta)
- Internamente: executar_com_tools() do tool_runner.py
- max_iteracoes=8 (padrão do tool_runner)
- Verbose mostra [iter N] tool call(s) — OpenAI ou DSML detectado

#### consultar_curso — detalhes de implementação
- Tokeniza pergunta, filtra palavras com len > 3
- Exige pelo menos min(2, len(palavras)) hits por linha
- Ordena resultados por número de hits (mais relevante primeiro)
- Fallback: busca por qualquer palavra se 0 resultados com 2+
- Limitação: busca keyword, não semântica — use 03_RAG para busca semântica

#### Padrão de uso
```python
from shared.tool_runner import executar_com_tools, ToolRunner

agente = ToolRunner(system=SYSTEM_AGENTE, verbose=True)
agente.registrar(SCHEMA_CALCULAR, calcular)
agente.registrar(SCHEMA_DATA_HORA, data_hora_atual)
agente.registrar(SCHEMA_CURSO, consultar_curso)
resp = agente.perguntar('Quanto é sqrt(1764)?')
```

### 02_ferramentas_customizadas.ipynb — Tools diversas
5 ferramentas registradas num único ToolRunner.

#### Ferramentas
- calcular: mesma do notebook 01
- buscar_web: DuckDuckGo HTML, sem API key, requests + BeautifulSoup
  - URL: https://html.duckduckgo.com/html/?q={query}
  - Parseia .result__title e .result__snippet
  - Trata Timeout e exceções genéricas
- ler_arquivo: lê .py/.md/.txt/.json do projeto com validação de segurança
  - Resolve caminho relativo a EAI07_BASE primeiro, depois PROJETO_BASE
  - Valida que path está dentro do projeto via relative_to()
  - max_linhas=80 padrão, mostra "[... +N linhas omitidas]"
- listar_arquivos: rglob com filtro de extensão e ignorar pastas de cache
  - Parâmetro: caminho (não diretorio) — alinhado com alias map do tool_runner
- gerar_codigo: LLM especializado com temperature=0.2, strip de blocos markdown

#### Lição aprendida — aliases de parâmetros
O DeepSeek usa `caminho` naturalmente. O parâmetro deve se chamar `caminho`
(não `diretorio`) para evitar TypeError. Alternativa: adicionar ao _PARAM_ALIASES
do tool_runner.py.

#### Boas práticas estabelecidas
- Ferramentas sempre retornam string (não lançar exceções — capturar e retornar erro)
- Descrição do schema é o principal fator de decisão do LLM sobre qual ferramenta usar
- Parâmetros opcionais não devem estar em `required`

### 03_multi_agentes.ipynb — Pipeline Condicional
Roteador → especialistas → síntese. Fluxo condicional por tipo de tarefa.

#### Componentes
- rotear(pergunta): LLM com temperature=0.0 retorna JSON {tipo, especialistas, justificativa}
  - tipos: 'simples' | 'composto'
  - especialistas: ['pesquisa'] | ['codigo'] | ['matematica'] | combinações
  - Fallback: {'tipo': 'simples', 'especialistas': ['pesquisa']} se JSON inválido
- criar_especialista_pesquisa(): ToolRunner com buscar_web + consultar_curso
- criar_especialista_codigo(): ToolRunner com listar_arquivos + ler_arquivo + gerar_codigo
- criar_especialista_matematica(): ToolRunner com calcular
- sintetizar(pergunta, resultados): combina saídas de múltiplos especialistas via LLM

#### Pipeline executar_pipeline()
```python
decisao = rotear(pergunta)              # classifica
for nome in decisao['especialistas']:   # executa em sequência
    agente = ESPECIALISTAS[nome]()
    resultado = agente.perguntar(pergunta + contexto_acumulado)
    contexto_acumulado += resultado

if len(resultados) == 1:
    return resultado                         # simples: direto
else:
    return sintetizar(pergunta, resultados)  # composto: síntese
```

#### Contexto acumulado
Cada especialista recebe a pergunta original + saída dos especialistas anteriores,
permitindo que o especialista de código use informações do de pesquisa.

### 04_memoria_conversacional.ipynb — Gestão de Contexto
Três estratégias de memória implementadas como classes independentes.

#### AgenteCurtoPrazo
- max_mensagens: N últimas mensagens no contexto (user+assistant = 2 por turn)
- _janela(): historico[-max_mensagens:]
- Perde contexto antigo silenciosamente quando historico > max_mensagens
- Uso: conversas curtas, sessão única

#### AgenteLongoPrazo
- Persistência: data/memoria/memoria_longo_prazo.json (lista de strings)
- carregar_memoria() na inicialização, salvar_memoria() no encerrar_sessao()
- extrair_fatos(historico): LLM extrai fatos relevantes em JSON {fatos: [...]}
- _system_com_memoria(): injeta fatos no system prompt como lista com bullets
- encerrar_sessao(): extrai + merge via set() + salva + limpa historico
- limpar_memoria(): apaga o arquivo JSON
- Uso: assistente pessoal multi-sessão

#### AgenteComSumarizacao
- estimar_tokens(mensagens): sum(len(content)) // 4 (heurística sem tiktoken)
- limite_tokens: threshold para disparar sumarização (padrão: 2000)
- manter_recentes: msgs preservadas após resumo (padrão: 4)
- sumarizar_historico(): LLM comprime msgs antigas em texto de até 200 palavras
- _system_com_resumo(): injeta resumo acumulado no system prompt
- Acumula: novo resumo é concatenado ao self.resumo existente
- Uso: conversas técnicas longas e densas

#### Quando usar cada estratégia
```
Conversa curta / sessão única   → AgenteCurtoPrazo(max_mensagens=20)
Assistente pessoal multi-sessão → AgenteLongoPrazo
Conversa técnica longa e densa  → AgenteComSumarizacao(limite_tokens=2000)
Produção com usuários reais     → AgenteLongoPrazo + AgenteComSumarizacao combinados
```

## ARQUITETURA COMPARTILHADA — tool_runner.py

### executar_com_tools()
```python
executar_com_tools(
    pergunta      : str,
    tools         : list,   # schemas JSON
    funcoes       : dict,   # nome → função Python
    system        : str  = None,
    verbose       : bool = True,
    max_iteracoes : int  = 8
) -> str
```

### Fluxo interno
- Caso A: message.tool_calls → executa ferramentas → continua
- Caso B: DSML no content → parseia → executa → continua
- Caso C: texto sem tool_calls → retorna (resposta final)

### _PARAM_ALIASES (mapeamentos já configurados)
```python
"diretorio" → "caminho"
"directory" → "caminho"
"path"      → "caminho"
"ext"       → "extensao"
"extension" → "extensao"
```

## DADOS E PERSISTÊNCIA
- Memória longo prazo: `data/memoria/memoria_longo_prazo.json`

## FAQ
Q: Por que usar ToolRunner em vez de executar_com_tools diretamente?
A: ToolRunner é mais conveniente para múltiplas ferramentas — registrar() acumula
   schemas e funcoes. Para uma ferramenta ou debug, executar_com_tools é mais claro.

Q: Por que consultar_curso usa keyword search e não o FAISS do 03_RAG?
A: Mantém o notebook independente sem carregar o modelo de embedding (~90MB).
   Em produção, integrar o RAG como ferramenta é a abordagem recomendada.

Q: O que fazer quando o agente entra em loop?
A: max_iteracoes=8 evita loops infinitos. Se persistir, revisar a descrição do schema
   para deixar mais claro quando o LLM deve parar de chamar ferramentas.

Q: Como adicionar novo especialista no pipeline multi-agente?
A: 1) Criar criar_especialista_X() retornando ToolRunner configurado.
   2) Adicionar ao dict ESPECIALISTAS.
   3) Atualizar SYSTEM_ROTEADOR com o novo tipo.

Q: estimar_tokens() é preciso?
A: É heurística conservadora (chars/4). Para produção, usar tiktoken.

## TAGS DE BUSCA
agente ReAct Reason Act Observe ToolRunner executar_com_tools function calling
buscar_web DuckDuckGo ler_arquivo listar_arquivos gerar_codigo calcular consultar_curso
multi-agente roteador pipeline condicional especialista pesquisa codigo matematica sintetizador
memória curto prazo longo prazo sumarização janela deslizante persistência JSON extrair_fatos
estimar_tokens DSML DeepSeek aliases parâmetros tool_runner shared

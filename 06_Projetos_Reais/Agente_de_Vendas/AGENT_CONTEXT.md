# AGENT_CONTEXT — Agente_de_Vendas
# Agente conversacional sobre banco de dados de vendas com interface Flask e registro de requisições
# Para uso por agentes de IA. Versão legível: README.md

## IDENTIFICAÇÃO
- Projeto: Agente_de_Vendas
- Módulo pai: EAI_07_AI_Generative / 06_Projetos_Reais
- Ambiente: eai07 (Python 3.11, conda) — notebooks de ML rodaram em `dev` (Python 3.13)
- Localização: `EAI_07_AI_Generative/06_Projetos_Reais/Agente_de_Vendas/`
- Dependências principais: flask, openai, pandas, matplotlib, openpyxl, sqlalchemy, scikit-learn, python-dotenv

## VISÃO GERAL
Projeto em três camadas: (1) pipeline ETL de planilhas Excel/CSV → banco SQLite, com notebooks de ML
para backtest e previsão de demanda; (2) notebook MVP `agente_vendas.ipynb` que implementa o agente
diretamente no Jupyter para desenvolvimento e validação iterativa antes do Flask; (3) interface Flask
chatbot com histórico de requisições persistido em SQLite. O agente LLM (DeepSeek) usa 3 ferramentas
(sql_query, gerar_grafico, export_excel) e responde em linguagem natural sobre vendas, clientes,
produtos, modelos e previsões.

---

## ESTRUTURA DE ARQUIVOS

```
Agente_de_Vendas/
├── agente.py               ← núcleo do agente: ferramentas, loop ReAct, logging
├── app.py                  ← servidor Flask: rotas, sessões em memória, registro SQLite
├── templates/index.html    ← interface chatbot dark mode (sem JS framework)
├── conversor/              ← notebooks de ETL (planilhas → DBVendas.db)
│   ├── converte_clientes.ipynb
│   ├── converte_produtos.ipynb
│   ├── converte_vendas.ipynb
│   ├── cria_calendario.ipynb
│   ├── backtest_geral.ipynb
│   ├── modelo_escolhido_metricas.ipynb
│   └── previsao_multi_step.ipynb
├── data/DBVendas.db        ← banco SQLite com 14 tabelas (~100k vendas)
├── planilhas/              ← fontes originais (Excel + CSV)
├── notebooks/agente_vendas.ipynb  ← MVP do agente: desenvolvimento e validação iterativa no Jupyter
├── logs/
│   ├── requisicoes.db      ← registro SQLite de todas as requisições
│   └── requisicoes.log     ← log em texto plano
└── outputs/                ← arquivos Excel exportados pelo agente
```

---

## BANCO DE DADOS — DBVendas.db

### Tabelas transacionais
- `vendas` (99.942 registros): ID_Pedido, Data_Venda (YYYY-MM-DD TEXT), Data_Entrega, ID_Cliente, ID_Canal
- `itens_vendas` (100.000 registros): ID_Pedido, ID_Produto, Qtde, Valor_Unitario, Valor_Total

### Dimensões
- `produtos` (212): ID_Produto, ID_Subcategoria, Descricao_Produto, ID_Marca, Preco_Unitario, Tributos, Custo
- `subcategorias` (22): ID_Subcategoria, ID_Categoria, Subcategoria
- `categorias` (7): ID_Categoria, Categoria
- `marcas` (21): ID_Marca, Marca
- `canais` (2): ID_Canal, Descricao_Canal — valores: "Internet" e "Loja Física"
- `clientes` (18.484): ID_Cliente, Nome, Email, Data_Nascimento, Estado_Civil, Genero, Educacao, ID_Cidade
- `cidades` (31): ID_Cidade, Cidade, UF
- `calendario` (6.209): Data (PK TEXT YYYY-MM-DD), 28 colunas de calendário BR (feriados, dias úteis, etc.)

### Tabelas de ML
- `previsao_demanda` (636): ID_Produto, Ano, Mes, Previsao — 3 meses à frente por produto
- `previsao_pivot` (212): Produto × colunas de período (MM/YYYY)
- `backtest_multinivel` (36): Nivel, Ano, MAE, Erro_%, Qtd, Data_Execucao
- `modelo_escolhido` (9): Nivel, Modelo, Vencedor (bool), MAE, Erro_%, RMSE, R2, N_Treino, N_Teste, Data_Execucao

### Atenção crítica — datas
Datas armazenadas como TEXT em formato YYYY-MM-DD (SQLite não tem tipo DATE nativo).
Para filtros por ano: `strftime('%Y', Data_Venda) = '2021'`
Para filtros por mês: `strftime('%m', Data_Venda) = '11'`
A coluna `Ano` na tabela `calendario` é TEXT (ex: '2021') — comparar como texto.

---

## PIPELINE ETL — CONSTRUÇÃO DO BANCO

### Ordem obrigatória de execução dos notebooks
1. `converte_clientes.ipynb` — cria `cidades` e `clientes` (CSV + Excel)
2. `converte_produtos.ipynb` — cria `canais`, `categorias`, `marcas`, `subcategorias`, `produtos`
3. `converte_vendas.ipynb` — cria `vendas` e `itens_vendas` (3 arquivos Excel por período)
4. `cria_calendario.ipynb` — cria tabela `calendario` com 28 colunas e feriados brasileiros
5. `backtest_geral.ipynb` — cria `backtest_multinivel`
6. `modelo_escolhido_metricas.ipynb` — cria `modelo_escolhido`
7. `previsao_multi_step.ipynb` — cria `previsao_demanda` e `previsao_pivot`

### Lições aprendidas — ETL
- `converte_clientes.ipynb`: erro de UNIQUE constraint ao re-executar com dados já existentes.
  Solução: descomentar as linhas de `Base.metadata.drop_all(engine)` antes de `create_all`,
  ou usar `if_exists='replace'` no `to_sql`.
- `converte_clientes.ipynb`: Email gerado como `PrimeiroNome@gmail.com` — não estava na planilha.
  Nome montado concatenando `Primeiro Nome` + `Sobrenome` do CSV.
- `converte_produtos.ipynb`: `Canal.xlsx` pode estar ausente no primeiro run.
  Solução: rodar `converte_clientes` antes (cria tabela canais via SQL direto) ou criar manualmente.
- `converte_vendas.ipynb`: `Valor_Unitario` calculado como `Valor_Total / Qtde` — não existia na planilha.
  Vendas em 3 arquivos Excel por período: 2010-2013, 2014-2017, 2018-2021.
  Chaves estrangeiras validadas em memória antes de inserir (sets de IDs válidos).
- `cria_calendario.ipynb`: feriados móveis por Algoritmo de Gauss.
  `Consciência Negra` (20/nov) incluído apenas a partir de 2024 — Lei 14.759/2023.
  Colunas de dias úteis usam `groupby` por string `ano_mes_str` (não por `Period`) para evitar
  bug de tipo nas versões mais recentes do pandas.

---

## PIPELINE ML — MODELOS DE PREVISÃO

### Ambiente dos notebooks de ML
Os notebooks `backtest_geral.ipynb`, `modelo_escolhido_metricas.ipynb` e `previsao_multi_step.ipynb`
rodam no ambiente `dev` (Python 3.13) — diferente do `eai07` usado no agente Flask.
Kernel: `Python (Dev)`. Engine: `sqlite:///../data/DBVendas.db`.

### Feature engineering — padrão comum aos 3 notebooks
```python
# Lags diretos (shift garante que só valores passados são usados)
df['Lag_1'] = df.groupby(colunas)['Qtde'].shift(1)
df['Lag_2'] = df.groupby(colunas)['Qtde'].shift(2)
df['Lag_3'] = df.groupby(colunas)['Qtde'].shift(3)

# ✅ CORRETO — shift(1) ANTES do rolling evita data leakage
df['Media_3'] = df.groupby(colunas)['Qtde'].transform(
    lambda x: x.shift(1).rolling(3).mean()
)
# ❌ ERRADO — rolling direto inclui o valor atual no cálculo
df['Media_3'] = df.groupby(colunas)['Qtde'].transform(
    lambda x: x.rolling(3).mean()
)

# Sazonalidade cíclica (apenas no backtest_geral)
df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
```

### backtest_geral.ipynb — Walk-forward por ano (2013-2021)
Algoritmo: HistGradientBoostingRegressor (max_iter=200, learning_rate=0.05, max_depth=6, random_state=42).
Features: Lag_1, Lag_2, Lag_3, Media_3, Mes_sin, Mes_cos + colunas de agrupamento.
Niveis avaliados: produto, categoria, canal, produto_canal.
Walk-forward: treina com anos < ano_teste, testa no ano_teste. Começa em 2013 (mínimo 3 anos de treino).

Resultados consolidados (média dos 9 anos de teste):
| Nível | MAE médio | Erro % médio | Interpretação |
|---|---|---|---|
| canal | 264,69 | 5,88% | Ok ✅ |
| categoria | 140,79 | 9,38% | Ok ✅ |
| produto | 20,63 | 46,71% | Ruim ⚠️ |
| produto_canal | 13,77 | 52,32% | Ruim ⚠️ |

Regra de mercado: <3% excelente · 3-5% muito bom · 5-10% ok · >10% ruim.
Saída: tabela `backtest_multinivel` (36 registros: 4 níveis × 9 anos).

### modelo_escolhido_metricas.ipynb — Seleção de modelo por nível
Compara 3 candidatos com 4 métricas. Cutoff: 2021-01-01 (treino < cutoff, teste >= cutoff).
Candidatos: LinearRegression, RandomForestRegressor (n=100), HistGradientBoosting (max_iter=200, lr=0.05).
Features adicionais em relação ao backtest: Media_6 (shift(1).rolling(6).mean()).
Vencedor: menor MAE. Todos os candidatos salvos (não só o vencedor).

Resultados completos — todos os candidatos por nível:
| Nível | Modelo | Vencedor | MAE | Erro % | RMSE | R² | N_Treino | N_Teste |
|---|---|---|---|---|---|---|---|---|
| produto | HistGradientBoosting | ✅ | 6,69 | 48,54% | 8,01 | 0,0003 | 85815 | 7818 |
| produto | LinearRegression | | 6,69 | 48,56% | 8,01 | -0,0002 | 85815 | 7818 |
| produto | RandomForest | | 6,81 | 49,41% | 8,19 | -0,0453 | 85815 | 7818 |
| canal | LinearRegression | ✅ | 41,17 | 27,75% | 51,25 | -0,0044 | 8022 | 726 |
| canal | RandomForest | | 41,79 | 28,17% | 52,06 | -0,0364 | 8022 | 726 |
| canal | HistGradientBoosting | | 42,04 | 28,34% | 52,01 | -0,0343 | 8022 | 726 |
| produto_canal | HistGradientBoosting | ✅ | 6,48 | 48,37% | 7,65 | 0,0004 | 86819 | 8038 |
| produto_canal | LinearRegression | | 6,48 | 48,37% | 7,65 | 0,0001 | 86819 | 8038 |
| produto_canal | RandomForest | | 6,53 | 48,70% | 7,75 | -0,0252 | 86819 | 8038 |

R² próximo de zero ou negativo = modelo não explica variância melhor que a média histórica.
Para decisões estratégicas, preferir canal (Erro% 5,88%) ou categoria (9,38%).
Saída: tabela `modelo_escolhido` (9 registros: 3 candidatos × 3 níveis).

### previsao_multi_step.ipynb — Previsão recursiva 3 meses
Algoritmo: RandomForestRegressor (n_estimators=100, random_state=42).
Features: ID_Produto, Ano, Mes, Lag_1, Lag_2, Media_3.
Target: Qtde diretamente (shift(-1) removido — gerava target incomparável com backtest).
Cutoff: 2021-01-01. MAE no teste: 21,91 · Erro%: 49,67%.
Horizonte: nov/2021 → jan/fev/mar 2022 (3 meses por produto).

Lógica recursiva correta:
```python
# Inicialização com valores reais do último período observado
lag1 = df_prod.iloc[-1]['Qtde']   # último valor real
lag2 = df_prod.iloc[-2]['Qtde']   # penúltimo valor real

for step in range(1, 4):
    media_3 = (lag1 + lag2 + df_prod.iloc[-3]['Qtde']) / 3
    pred = model.predict(X_pred)[0]
    # Atualização: próxima iteração usa a previsão atual como lag
    lag2 = lag1
    lag1 = pred
```

Saídas:
- Tabela `previsao_demanda` (636 registros: 212 produtos × 3 meses)
- `planilhas/Previsao_3_Meses.xlsx` — planilha simples
- Tabela `previsao_pivot` (212 registros) + `planilhas/pivot_previsao.xlsx` — pivot horizontal formatado
  com cabeçalho azul (#1F4E79), fonte Arial 10, freeze_panes em B2, bordas finas, número ##,##0.00

---

## NOTEBOOK MVP — notebooks/agente_vendas.ipynb

Desenvolvido antes do Flask como ambiente de prototipagem e validação iterativa.
Kernel: Python (EAI_07) — ambiente eai07.
Localização: `notebooks/agente_vendas.ipynb` — outputs em `notebooks/outputs/`.

### Estrutura do notebook (10 células de código + markdown)
| Célula | Conteúdo |
|---|---|
| 1 | Imports, `.env`, cliente DeepSeek, `DB_PATH` |
| 2 | `get_connection()` + `listar_tabelas()` — inspeção do banco |
| 3 | `SCHEMA_DDL` — DDL resumido das tabelas expostas ao agente |
| 4 | Definição das 3 ferramentas: `sql_query`, `gerar_grafico`, `export_excel` |
| 5 | `TOOLS` (schemas OpenAI) + `FUNCOES` (mapa nome→função) |
| 6 | `SYSTEM_PROMPT` com schema + regras críticas + exemplos few-shot |
| 7 | Loop ReAct: `_parse_tool_calls_dsml()`, `_executar_tool()`, `agente_responder()` |
| 8 | `chat_agente()` — loop interativo com `input()`, comandos `limpar` e `sair` |
| 9 | Bateria de 10 perguntas de teste automatizadas |
| 10 | Testes unitários das ferramentas sem LLM |

### Inicialização — caminho do shared/ no notebook
```python
from pathlib import Path
current_dir = Path.cwd()   # notebooks/
shared_dir = current_dir.parent.parent.parent / 'shared'
sys.path.insert(0, str(shared_dir))
DB_PATH = os.path.abspath('../data/DBVendas.db')
load_dotenv('../../../.env')
```

### chat_agente() — interface interativa no notebook
```python
chat_agente(verbose=True)
# verbose=True: imprime [iter N] e [tool] para cada chamada
# comandos especiais: 'sair' / 'exit' / 'quit' encerra; 'limpar' reseta histórico
```

### agente_responder() — retorno
```python
resposta, historico = agente_responder(pergunta, historico, verbose=True)
# resposta: str com o texto final do LLM
# historico: lista atualizada de dicts role/content
```

### Perguntas de teste (célula 9) — 10 perguntas validadas
```python
PERGUNTAS_TESTE = [
    "Quais os 5 produtos mais vendidos em 2021?",
    "Qual foi o erro percentual médio do modelo por nível?",
    "Qual modelo foi escolhido para o nível produto?",
    "Qual a previsão de demanda para o produto 1010 nos próximos meses?",
    "Gere um gráfico de vendas mensais por canal nos últimos 2 anos",
    "Exporte a previsão dos próximos meses para Excel",
    "Quais os 10 clientes que mais compraram em valor total? Mostre nome, cidade, UF e total gasto.",
    "Qual o faturamento total por canal de venda em cada ano? Mostre canal, ano e receita.",
    "Quais as 5 categorias de produto mais vendidas em quantidade? Inclua subcategoria e total de unidades.",
    "Gere um gráfico de barras com o faturamento total por categoria de produto.",
]
```

### Testes unitários (célula 10) — sem LLM
```python
# Valida o banco antes de envolver o LLM
r  = sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
r2 = sql_query("SELECT p.Descricao_Produto, SUM(iv.Qtde) as Total FROM itens_vendas iv ...")
r3 = gerar_grafico(query="...", tipo='line', titulo='Receita Mensal', eixo_x='Mês', eixo_y='Receita (R$)')
r4 = export_excel(query="...", nome_arquivo='top20_produtos', incluir_grafico=True)
```

### Diferenças notebook vs Flask (agente.py)
| Aspecto | Notebook | Flask (agente.py) |
|---|---|---|
| Gráficos | `plt.show()` inline no Jupyter | `savefig` → base64 → data URI |
| Histórico | lista local por execução | dict em memória por sessao_id |
| Ferramentas | retornam string direta | prefixos `__GRAFICO__` e `__EXCEL__` |
| Log | verbose no stdout | `logging` + `requisicoes.db` |
| Backend Matplotlib | padrão (GUI) | `Agg` obrigatório |

---

## AGENTE LLM — agente.py

### Ferramentas
```python
# Executa SELECT, retorna tabela texto (máx 50 linhas + indicador do total real)
# Bloqueia queries não-SELECT. Usa sqlite3 + pd.read_sql via row_factory=sqlite3.Row
def sql_query(query: str) -> str

# Executa SELECT, plota Matplotlib com fundo dark (#1a1f2e), retorna '__GRAFICO__' + data URI PNG base64
# Paleta: ['#4f8ef7','#f7824f','#4ff79e','#f74f9e','#f7d44f']
# Tipos: 'bar' (padrão), 'barh', 'line', 'area'
# Com coluna_serie: df.pivot(index=col_x, columns=coluna_serie, values=col_val) antes de plotar
# Sem coluna_serie: plota SOMENTE numeric_cols[0] (evita sobreposição de escalas)
# DPI: 130, bbox_inches='tight', figsize=(11, 4.5)
def gerar_grafico(query: str, tipo: str = 'bar', titulo: str = '',
                  eixo_x: str = '', eixo_y: str = '',
                  coluna_serie: str = '') -> str

# Executa SELECT, gera .xlsx formatado em outputs/, retorna '__EXCEL__' + caminho + '||' + msg
# Formatação: cabeçalho azul #1F4E79, linhas alternadas #DCE6F1, bordas finas #CCCCCC
# Aba extra 'Gráfico' com BarChart do openpyxl se incluir_grafico=True e há coluna numérica
# nome_arquivo: espaços → _, / → -
def export_excel(query: str, nome_arquivo: str = 'relatorio',
                 incluir_grafico: bool = True) -> str
```

### Lógica de colunas do gerar_grafico
- Com `coluna_serie`: faz `df.pivot(index=eixo_x, columns=coluna_serie, values=valor)` — uma série por grupo
- Sem `coluna_serie`: plota SOMENTE `numeric_cols[0]` — evita sobreposição de escalas diferentes

### Prefixos de retorno especiais
- `__GRAFICO__data:image/png;base64,...` → app.py extrai e envia como `grafico` no JSON
- `__EXCEL__/caminho/arquivo.xlsx||mensagem` → app.py extrai caminho e serve em `/download/<arquivo>`

### Loop ReAct — responder()
```python
resultado = responder(
    pergunta=str, historico=list, llm=OpenAI, modelo=str, max_iter=8
)
# → dict: {texto, grafico, excel, iteracoes, tools_usadas}
```

Fluxo interno (3 casos):
- **Caso A** `msg.tool_calls` → executa cada tool, captura `__GRAFICO__`/`__EXCEL__`, continua
- **Caso B** DSML em `msg.content` → `_parse_dsml()` extrai nome+args, executa, continua
- **Caso C** texto sem tool_calls → resposta final, loga e retorna

`_parse_dsml()`: regex sobre `<|DSML|function_calls>...<invoke name="X">...<parameter name="Y">V</parameter>`.
Tenta `json.loads(pval)` para cada valor — fallback para string se falhar.

Prefixos de retorno das ferramentas:
- `__GRAFICO__data:image/png;base64,...` → extrai e coloca em `resultado['grafico']`
- `__EXCEL__/caminho.xlsx||mensagem` → extrai caminho e mensagem, coloca em `resultado['excel']`

Logging: `req_logger` escreve em `logs/requisicoes.log` a cada resposta final
(pergunta[:120], resposta[:120], iterações, tools usadas).

---

## FLASK — app.py

### Rotas
| Rota | Método | Descrição |
|---|---|---|
| `/` | GET | Interface chatbot |
| `/chat` | POST | `{pergunta, sessao_id}` → `{texto, grafico, excel, iteracoes, tools, duracao_ms}` |
| `/limpar` | POST | Limpa histórico da sessão em memória |
| `/historico` | GET | Últimas 50 requisições do banco (JSON) |
| `/download/<arquivo>` | GET | Serve Excel para download |
| `/status` | GET | `{ok, modelo, vendas, db_path}` |

### Sessões e registro
Histórico de conversa: dict em memória `sessao_id → list`, limitado a 40 mensagens.
Cada requisição gravada em `logs/requisicoes.db`: sessao_id, timestamp, pergunta, resposta (500 chars),
tem_grafico, tem_excel, tools_usadas, iteracoes, duracao_ms.
Também gravado em `logs/requisicoes.log` via Python logging.

---

## INTERFACE — templates/index.html

### Funcionalidades
- Dark mode: bg #0d1117, accent #4f8ef7. Fontes: IBM Plex Mono, Syne, Inter.
- Sidebar com histórico (últimas 50 requisições, atualiza a cada 15s).
- Clicar em requisição do histórico a re-envia (função `reConsultar(idx)`).
- Item clicado recebe classe `.ativo` com borda azul à esquerda como feedback.
- Tabelas Markdown convertidas para `<table class="md-table">` com `overflow-x: auto`.
- Gráficos exibidos como `<img>` com data URI inline.
- Meta por mensagem: ferramentas usadas, iterações, duração em ms.
- Bolha do agente: `max-width: 94%`. Bolha do usuário: `max-width: 72%`.

---

## SYSTEM_PROMPT E REGRAS CRÍTICAS DO AGENTE

`SYSTEM_PROMPT` definido em `agente.py` como f-string que embute o `SCHEMA_DDL`.
Estrutura: identidade → capacidades (3 ferramentas) → regras críticas → schema DDL → exemplos de queries.
`temperature=0` em todas as chamadas LLM — respostas determinísticas.

### Regras críticas (do SYSTEM_PROMPT real)

**REGRA — UMA ÚNICA QUERY COMPLETA**
```
NUNCA faça queries exploratórias. Inclua TODOS os campos necessários na primeira query.

❌ ERRADO: 2 queries para a mesma pergunta
✅ CORRETO:
  SELECT p.Descricao_Produto,
         SUM(iv.Qtde)                  AS Total_Vendido,
         ROUND(SUM(iv.Valor_Total), 2) AS Receita_Total,
         COUNT(DISTINCT v.ID_Pedido)   AS Pedidos_Unicos
  FROM itens_vendas iv
  JOIN vendas   v ON iv.ID_Pedido  = v.ID_Pedido
  JOIN produtos p ON iv.ID_Produto = p.ID_Produto
  WHERE strftime('%Y', v.Data_Venda) = '2021'
  GROUP BY p.ID_Produto ORDER BY Total_Vendido DESC LIMIT 5;
```

**REGRA — ALINHAMENTO DE TABELAS MARKDOWN**
```
sql_query() já retorna tabela com | alinhados verticalmente.
Preservar EXATAMENTE como recebido — não modificar espaços.
Apresentar tabela, pular linha, depois adicionar insights.
```

**REGRA — USO DE GRÁFICO**
```
Use gerar_grafico DIRETAMENTE — NÃO chame sql_query antes.
Após gráfico: obrigatório 3-5 insights (maior valor, menor valor, tendência, observação de negócio).
```

**REGRA — MÚLTIPLAS SÉRIES**
```
Para comparar grupos (ex: Internet vs Loja Física):
  - Use coluna_serie para pivot automático
  - Query com EXATAMENTE 3 colunas: (eixo_x, coluna_serie, valor)
  - NÃO inclua colunas extras — causam sobreposição de escalas

✅ CORRETO:
  query="SELECT Mes, Descricao_Canal, SUM(Valor_Total) AS Receita FROM ..."
  coluna_serie="Descricao_Canal"
```

### Exemplos de queries no system prompt
```sql
-- Top 5 produtos 2021:
SELECT p.Descricao_Produto, SUM(iv.Qtde) AS Total_Vendido,
       ROUND(SUM(iv.Valor_Total),2) AS Receita_Total,
       COUNT(DISTINCT v.ID_Pedido) AS Pedidos_Unicos
FROM itens_vendas iv
JOIN vendas v ON iv.ID_Pedido=v.ID_Pedido
JOIN produtos p ON iv.ID_Produto=p.ID_Produto
WHERE strftime('%Y',v.Data_Venda)='2021'
GROUP BY p.ID_Produto ORDER BY Total_Vendido DESC LIMIT 5;

-- Erro % por nível (backtest):
SELECT Nivel, ROUND(AVG("Erro_%"),2) AS Erro_Medio,
       ROUND(AVG(MAE),2) AS MAE_Medio, COUNT(*) AS N_Backtests
FROM backtest_multinivel GROUP BY Nivel ORDER BY Erro_Medio;

-- Faturamento por categoria (para gráfico de barras):
SELECT cat.Categoria, ROUND(SUM(iv.Valor_Total),2) AS Receita_Total
FROM itens_vendas iv
JOIN produtos p ON iv.ID_Produto=p.ID_Produto
JOIN subcategorias sub ON p.ID_Subcategoria=sub.ID_Subcategoria
JOIN categorias cat ON sub.ID_Categoria=cat.ID_Categoria
GROUP BY cat.ID_Categoria ORDER BY Receita_Total DESC;
```

### Detalhes do SCHEMA_DDL exposto ao agente
Comentário crítico no topo do DDL:
```sql
-- ATENÇÃO: datas em formato YYYY-MM-DD (texto)
-- Use strftime('%Y', Data_Venda) para filtrar por ano
```
Tabelas expostas: vendas, itens_vendas, produtos, subcategorias, categorias, marcas,
canais, clientes, cidades, calendario, previsao_demanda, backtest_multinivel, modelo_escolhido.
`previsao_pivot` NÃO está no DDL — o agente não sabe que ela existe.
`"Erro_%"` declarado com aspas duplas no DDL — lembrete para queries.
---

## CAMINHO DO SHARED/ — ATENÇÃO

O notebook MVP (`notebooks/agente_vendas.ipynb`) fica 3 níveis abaixo da raiz do EAI_07,
por isso o caminho para o `shared/` é diferente do padrão dos outros notebooks:

```python
# notebooks/agente_vendas.ipynb — 3 níveis acima
from pathlib import Path
current_dir = Path.cwd()                                    # .../Agente_de_Vendas/notebooks/
shared_dir  = current_dir.parent.parent.parent / 'shared'  # .../EAI_07_AI_Generative/shared/
sys.path.insert(0, str(shared_dir))
load_dotenv('../../../.env')                                # 3 níveis acima
DB_PATH = os.path.abspath('../data/DBVendas.db')           # 1 nível acima

# agente.py e app.py — ficam na raiz do projeto (2 níveis acima do EAI_07)
# Usam path resolution automático subindo na hierarquia até encontrar shared/
```

---

---

## FAQ

Q: Por que o agente chamava sql_query duas vezes para a mesma pergunta?
A: DeepSeek fazia query exploratória e depois enriquecia com campos adicionais.
   Corrigido no system prompt com exemplos ❌/✅ e instrução de query única completa.

Q: Por que gráficos mostravam séries sobrepostas em escalas erradas?
A: Query com 4+ colunas numéricas (Receita, Qtde, Pedidos) era plotada toda de uma vez.
   Corrigido: sem coluna_serie, plota apenas a primeira coluna numérica.
   Para múltiplas séries comparáveis, usar coluna_serie + query de 3 colunas.

Q: Como filtrar por ano em vendas?
A: `strftime('%Y', Data_Venda) = '2021'` — não `YEAR()` (não existe em SQLite).

Q: A coluna Erro_% precisa de aspas nas queries?
A: Sim. `"Erro_%"` com aspas duplas — o % é caractere especial em SQLite.

Q: Como re-executar o ETL sem erros de UNIQUE constraint?
A: Descomentar `Base.metadata.drop_all(engine)` antes de `create_all` em converte_clientes.
   Em converte_produtos, `session.merge()` já trata duplicatas.

Q: O histórico de requisições persiste entre restarts do Flask?
A: Sim. `logs/requisicoes.db` é permanente. O histórico de conversa (sessoes dict) perde no restart.

Q: Por que o erro % do modelo de produto é tão alto (46-52%)?
A: Variabilidade estocástica intrínseca ao nível produto individual. R² ≈ 0 confirma.
   Para decisões estratégicas, preferir previsões canal (5,88%) ou categoria (9,38%).

---

## TAGS DE BUSCA
agente vendas flask chatbot sqlite deepseek openai function calling tool use react loop
previsao demanda backtest multinivel random forest histgradientboosting sklearn media movel lag
etl planilhas excel csv sqlalchemy orm pipeline conversor converte clientes produtos vendas
calendario feriados brasil algoritmo gauss pascoa consciencia negra
data leakage rolling shift feature engineering media movel
gerar grafico matplotlib base64 data uri coluna serie pivot multiplas series
export excel openpyxl formatacao download
registro requisicoes historico sessao sqlite logging duracao ms
system prompt few shot query completa uma unica chamada regra critica
tabela markdown renderizacao html scroll horizontal jupyter display
dark mode ibm plex mono syne inter css variables frontend reConsultar historico sidebar
EAI07 EAI_07 generative AI modulo 07 especialista ia agente banco dados

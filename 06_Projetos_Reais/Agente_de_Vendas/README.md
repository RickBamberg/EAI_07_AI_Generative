# 🤖 Agente de Vendas

Agente conversacional em linguagem natural sobre o banco de dados de vendas `DBVendas.db`.
Responde perguntas, gera gráficos inline e exporta relatórios Excel — tudo via chat.

Desenvolvido em duas etapas: primeiro um **notebook Jupyter** (`agente_vendas.ipynb`) para prototipagem
e validação iterativa, depois uma **interface Flask** com painel de histórico de requisições.

---

## O que este projeto faz

Você digita uma pergunta como *"Quais os 5 produtos mais vendidos em 2021?"* e o agente:

1. Gera a query SQL correta para o banco
2. Executa e formata o resultado como tabela
3. Responde em português com análise dos dados

Para pedidos visuais (*"gere um gráfico de faturamento por categoria"*), plota o gráfico
diretamente no chat. Para exportações (*"salve a previsão em Excel"*), disponibiliza o arquivo
para download com um clique.

---

## Estrutura do projeto

```
Agente_de_Vendas/
├── agente.py               ← núcleo do agente (ferramentas, LLM, loop de decisão)
├── app.py                  ← servidor web Flask
├── templates/
│   └── index.html          ← interface chatbot
├── conversor/              ← notebooks para construir o banco de dados
│   ├── converte_clientes.ipynb
│   ├── converte_produtos.ipynb
│   ├── converte_vendas.ipynb
│   ├── cria_calendario.ipynb
│   ├── backtest_geral.ipynb
│   ├── modelo_escolhido_metricas.ipynb
│   └── previsao_multi_step.ipynb
├── data/
│   └── DBVendas.db         ← banco SQLite (~100k vendas, 14 tabelas)
├── planilhas/              ← arquivos fonte Excel e CSV
├── notebooks/
│   └── agente_vendas.ipynb ← MVP do agente: prototipagem e validação antes do Flask
├── logs/
│   ├── requisicoes.db      ← histórico de todas as consultas
│   └── requisicoes.log     ← log em texto
└── outputs/                ← arquivos Excel gerados pelo agente
```

---

## Pré-requisitos

```bash
conda activate eai07

pip install flask openai pandas matplotlib openpyxl sqlalchemy scikit-learn python-dotenv
```

O arquivo `.env` deve estar em `EAI_07_AI_Generative/`:

```
DEEPSEEK_API_KEY=sk-...
LLM_MODEL=deepseek-chat
LLM_PROVIDER=deepseek
```

---

## Como construir o banco de dados

Se o `DBVendas.db` ainda não existir, rode os notebooks da pasta `conversor/`
**nesta ordem**:

| # | Notebook | O que faz |
|---|---|---|
| 1 | `converte_clientes.ipynb` | Importa cidades (Excel) e clientes (CSV) |
| 2 | `converte_produtos.ipynb` | Importa canais, categorias, marcas, subcategorias e produtos |
| 3 | `converte_vendas.ipynb` | Importa vendas e itens de 3 arquivos Excel (2010-2021) |
| 4 | `cria_calendario.ipynb` | Gera dimensão de calendário com feriados brasileiros |
| 5 | `backtest_geral.ipynb` | Avalia modelos em walk-forward e salva `backtest_multinivel` |
| 6 | `modelo_escolhido_metricas.ipynb` | Seleciona modelo vencedor por nível e salva métricas |
| 7 | `previsao_multi_step.ipynb` | Gera previsão de 3 meses e salva `previsao_demanda` |

> **Atenção:** Se precisar re-executar do zero, descomente a linha `Base.metadata.drop_all(engine)`
> em `converte_clientes.ipynb` para evitar erros de chave duplicada.

---

## Notebook MVP — agente_vendas.ipynb

Antes de montar o Flask, o agente foi construído e validado no Jupyter.
O notebook é o ponto de entrada recomendado para entender o funcionamento do agente
e testar novas perguntas de forma rápida.

### Localização
```
notebooks/agente_vendas.ipynb
```

### Como usar o notebook

**Rode as células em ordem (1 a 7)** para inicializar o agente. Em seguida:

**Célula 8 — Chat interativo:**
```python
chat_agente(verbose=True)
# Digite sua pergunta e pressione Enter
# verbose=True mostra qual ferramenta foi chamada em cada iteração
# Comandos: 'limpar' reseta o histórico | 'sair' encerra
```

**Célula 9 — Bateria de testes automatizados:**
Roda 10 perguntas pré-definidas em sequência e imprime o resultado completo de cada uma.
Útil para validar o comportamento do agente após alterações no system prompt.

**Célula 10 — Testes unitários das ferramentas (sem LLM):**
Testa `sql_query`, `gerar_grafico` e `export_excel` diretamente, sem passar pelo LLM.
Rode esta célula primeiro para confirmar que o banco está acessível antes de usar o agente.

### Perguntas testadas e validadas no notebook
```
Quais os 5 produtos mais vendidos em 2021?
Qual foi o erro percentual médio do modelo por nível?
Qual modelo foi escolhido para o nível produto?
Qual a previsão de demanda para o produto 1010 nos próximos meses?
Gere um gráfico de vendas mensais por canal nos últimos 2 anos
Exporte a previsão dos próximos meses para Excel
Quais os 10 clientes que mais compraram em valor total?
Qual o faturamento total por canal de venda em cada ano?
Quais as 5 categorias mais vendidas em quantidade?
Gere um gráfico de faturamento total por categoria de produto
```

### Diferença entre notebook e Flask
O notebook usa `plt.show()` para exibir gráficos inline no Jupyter.
No Flask, os gráficos são convertidos para imagem base64 e exibidos diretamente no chat.
O histórico de conversa no notebook é uma lista local — no Flask é mantido em memória por sessão.

---

## Como rodar a interface web

```bash
cd EAI_07_AI_Generative/06_Projetos_Reais/Agente_de_Vendas
python app.py
```

Acesse **http://localhost:5000** no navegador.

---

## Como usar

### Interface chatbot
A tela inicial mostra 6 perguntas prontas para clicar. Você também pode digitar qualquer pergunta
no campo de texto e pressionar **Enter** (ou Shift+Enter para nova linha).

### Histórico de requisições
O painel lateral esquerdo lista todas as consultas realizadas com:
- Ferramenta utilizada (sql_query, gerar_grafico, export_excel)
- Indicadores de gráfico 📈 e Excel 📥
- Tempo de resposta em milissegundos

Clique em qualquer item do histórico para **re-consultar** aquela pergunta automaticamente.

### Tipos de resposta
| Pedido | Resposta do agente |
|---|---|
| Pergunta analítica | Tabela formatada + análise em texto |
| "gere um gráfico de..." | Gráfico inline no chat |
| "exporte para Excel" | Botão de download direto |

---

## O banco de dados

### Dados de vendas
- **99.942 pedidos** no período 2010-2021
- **100.000 itens** de venda
- **2 canais**: Internet e Loja Física
- **212 produtos** em 7 categorias e 22 subcategorias
- **18.484 clientes** em 31 cidades brasileiras

### Previsão de demanda
O banco inclui previsões de demanda para os próximos 3 meses (geradas por machine learning):

| Tabela | Conteúdo |
|---|---|
| `previsao_demanda` | Previsão mensal por produto (636 registros) |
| `backtest_multinivel` | Acurácia histórica por nível de agregação |
| `modelo_escolhido` | Qual modelo venceu em cada nível com métricas completas |

**Resultados do backtest (erro percentual médio):**
- Canal: 5,88% ✅
- Categoria: 9,38% ✅
- Produto individual: 46,71% ⚠️ (alta variabilidade é esperada neste nível)

---

## Rotas da API

| Rota | Método | Descrição |
|---|---|---|
| `/` | GET | Interface chatbot |
| `/chat` | POST | Processar pergunta |
| `/limpar` | POST | Iniciar nova conversa |
| `/historico` | GET | Últimas 50 requisições (JSON) |
| `/download/<arquivo>` | GET | Baixar Excel exportado |
| `/status` | GET | Verificar conexão com banco e modelo |

---

## Exemplos de perguntas

```
Quais os 5 produtos mais vendidos em 2021?
Qual o faturamento total por canal de venda em cada ano?
Quais os 10 clientes que mais compraram em valor total?
Gere um gráfico de barras com o faturamento por categoria de produto
Gere um gráfico de vendas mensais por canal nos últimos 2 anos
Qual foi o erro percentual médio do modelo por nível?
Qual modelo foi escolhido para o nível produto?
Qual a previsão de demanda para o produto 1010?
Exporte a previsão dos próximos meses para Excel
Quais as 5 categorias mais vendidas em quantidade?
```

---

## Detalhes técnicos

### Agente LLM
- Provider: DeepSeek via API compatível com OpenAI
- Modelo padrão: `deepseek-chat`
- Loop de decisão: ReAct com suporte a formato DSML proprietário do DeepSeek
- Máximo de 8 iterações por pergunta

### Gráficos
- Biblioteca: Matplotlib com backend `Agg` (sem janela gráfica — obrigatório para Flask)
- Tema: dark mode (fundo `#1a1f2e`)
- Retornados como imagem PNG em base64 — sem arquivos temporários

### Registro de requisições
Cada consulta é registrada automaticamente com:
- Timestamp, sessão, pergunta e resposta
- Ferramentas utilizadas e número de iterações do LLM
- Duração em milissegundos
- Indicadores de gráfico e Excel gerados

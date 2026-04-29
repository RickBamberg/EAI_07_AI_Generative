# Arquitetura do Assistente Técnico IA

## Componentes

- Flask (API)
- RAG Engine
- Vector Store
- LLM
- Memória

## Fluxo

1. Usuário envia pergunta
2. Sistema decide se usa RAG
3. Busca semântica
4. Monta prompt
5. LLM responde
6. Retorna resposta + fontes

## Observações

- Pré-carregamento em background
- Thread-safe (lock no índice)
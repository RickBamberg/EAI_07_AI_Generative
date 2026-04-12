from openai import OpenAI

# Configure sua chave
API_KEY = "sk-0bc47599cc2745249993a408822fa556"  # Substitua pela sua chave

# Cria o cliente
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

# Faz uma pergunta simples
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "Me diga 'Tudo funcionando!' se você estiver recebendo esta mensagem."}
    ],
    max_tokens=50
)

# Mostra a resposta
print("Resposta da DeepSeek:")
print(response.choices[0].message.content)
print(f"\nTokens usados: {response.usage.total_tokens}")
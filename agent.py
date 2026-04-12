from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# MODELO BARATO (CLASSIFICAÇÃO)
# -----------------------------
def classify_intent(text):

    prompt = f"""
    Classifique a complexidade da pergunta abaixo.

    Responda apenas com:
    simples
    medio
    complexo

    Pergunta: {text}
    """

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=prompt
    )

    intent = response.output_text.strip().lower()

    return intent


# -----------------------------
# MODELO MÉDIO
# -----------------------------
def medium_model(text):

    response = client.responses.create(
        model="gpt-5-mini",
        input=text
    )

    return response.output_text


# -----------------------------
# MODELO FORTE
# -----------------------------
def strong_model(text):

    response = client.responses.create(
        model="gpt-5.4",
        input=text
    )

    return response.output_text


# -----------------------------
# AGENTE
# -----------------------------
def agent(text):

    intent = classify_intent(text)

    print("Classificação:", intent)

    if intent == "simples":
        return medium_model(text)

    if intent == "medio":
        return medium_model(text)

    if intent == "complexo":
        return strong_model(text)

    return medium_model(text)


# -----------------------------
# LOOP DE CHAT
# -----------------------------
def main():

    print("AI Agent iniciado (digite 'sair')")

    while True:

        user_input = input("\nVocê: ")

        if user_input.lower() == "sair":
            break

        response = agent(user_input)

        print("\nAgente:", response)


if __name__ == "__main__":
    main()

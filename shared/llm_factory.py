"""
shared/llm_factory.py
=====================
Factory provider-agnóstica compartilhada por todos os módulos do EAI_07.

Como usar nos notebooks:
    import sys, os
    sys.path.append(os.path.abspath('..'))          # aponta para EAI_07/
    from shared.llm_factory import chat, get_provider_info

Como usar no projeto Assistente_Tecnico_IA:
    import sys, os
    sys.path.append(os.path.abspath('../../../'))   # aponta para EAI_07/
    from shared.llm_factory import chat

Providers suportados:
    deepseek  → API compatível com SDK OpenAI (recomendado: custo-benefício)
    anthropic → Claude Haiku/Sonnet via SDK próprio
    openai    → GPT-4o-mini/GPT-4o
    ollama    → Modelos locais gratuitos (Llama, Mistral, Phi...)

Configuração (arquivo .env na raiz do EAI_07):
    LLM_PROVIDER=deepseek
    LLM_MODEL=deepseek-chat
    DEEPSEEK_API_KEY=sk-...
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Sobe na hierarquia até encontrar o .env na raiz do EAI_07
def _find_and_load_env():
    """Procura o .env subindo pelos diretórios pai."""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):  # máximo 5 níveis acima
        env_path = os.path.join(current, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            return env_path
        current = os.path.dirname(current)
    load_dotenv()  # fallback: busca padrão
    return None

_find_and_load_env()

# ── Configurações lidas do .env ──────────────────────────────
PROVIDER    = os.getenv("LLM_PROVIDER", "ollama").lower()
MODEL       = os.getenv("LLM_MODEL", "llama3.2")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "2048"))


# ── Interface unificada ──────────────────────────────────────
def chat(
    prompt: str,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Envia uma mensagem para o LLM configurado no .env e retorna a resposta.

    Parâmetros:
        prompt      : Mensagem do usuário
        system      : Prompt de sistema (comportamento do modelo)
        temperature : Sobrescreve LLM_TEMPERATURE do .env se informado
        max_tokens  : Sobrescreve LLM_MAX_TOKENS do .env se informado

    Retorno:
        Resposta do modelo como string

    Exemplo:
        resposta = chat("O que é RAG?", system="Responda de forma didática.")
    """
    temp   = temperature if temperature is not None else TEMPERATURE
    tokens = max_tokens  if max_tokens  is not None else MAX_TOKENS

    if PROVIDER in ("deepseek", "openai", "ollama"):
        return _chat_openai_sdk(prompt, system, temp, tokens)
    elif PROVIDER == "anthropic":
        return _chat_anthropic(prompt, system, temp, tokens)
    else:
        raise ValueError(
            f"Provider '{PROVIDER}' não suportado.\n"
            "Valores válidos: deepseek | anthropic | openai | ollama\n"
            "Verifique LLM_PROVIDER no seu .env"
        )


def chat_stream(
    prompt: str,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    """
    Versão streaming do chat — gera tokens em tempo real.

    Exemplo:
        for chunk in chat_stream("Explique transformers passo a passo"):
            print(chunk, end="", flush=True)
    """
    temp   = temperature if temperature is not None else TEMPERATURE
    tokens = max_tokens  if max_tokens  is not None else MAX_TOKENS

    if PROVIDER in ("deepseek", "openai", "ollama"):
        yield from _stream_openai_sdk(prompt, system, temp, tokens)
    elif PROVIDER == "anthropic":
        yield from _stream_anthropic(prompt, system, temp, tokens)
    else:
        raise ValueError(f"Provider '{PROVIDER}' não suportado.")


def get_provider_info() -> dict:
    """Retorna informações sobre o provider e modelo ativos."""
    return {
        "provider"   : PROVIDER,
        "model"      : MODEL,
        "temperature": TEMPERATURE,
        "max_tokens" : MAX_TOKENS,
    }


# ── Implementações por provider ──────────────────────────────

def _get_openai_client():
    """Instancia o client OpenAI-compatível conforme o provider."""
    from openai import OpenAI

    if PROVIDER == "deepseek":
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    elif PROVIDER == "ollama":
        return OpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    else:  # openai
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _build_messages(prompt: str, system: Optional[str]) -> list:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _chat_openai_sdk(prompt, system, temperature, max_tokens) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=_build_messages(prompt, system),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _stream_openai_sdk(prompt, system, temperature, max_tokens):
    client = _get_openai_client()
    with client.chat.completions.create(
        model=MODEL,
        messages=_build_messages(prompt, system),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def _chat_anthropic(prompt, system, temperature, max_tokens) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text


def _stream_anthropic(prompt, system, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


# ── Teste rápido ─────────────────────────────────────────────
if __name__ == "__main__":
    info = get_provider_info()
    print(f"\n{'='*50}")
    print(f"  Provider : {info['provider']}")
    print(f"  Modelo   : {info['model']}")
    print(f"  Temp     : {info['temperature']}")
    print(f"{'='*50}\n")

    print("Testando chat()...")
    resposta = chat(
        prompt="Responda apenas: 'Conexão OK com [nome do modelo]'",
        system="Seja extremamente breve.",
    )
    print(f"  {resposta}\n")

    print("Testando chat_stream()...")
    print("  ", end="")
    for chunk in chat_stream("Conte até 5 lentamente, um número por linha."):
        print(chunk, end="", flush=True)
    print("\n")

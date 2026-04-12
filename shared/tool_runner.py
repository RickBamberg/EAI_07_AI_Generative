"""
shared/tool_runner.py
=====================
Executor de function calling compatível com todos os providers,
com tratamento robusto do formato DSML do DeepSeek.

O DeepSeek às vezes retorna tool calls assim em vez do padrão OpenAI:
    <|DSML|function_calls>
    <|DSML|invoke name="nome_funcao">
    <|DSML|parameter name="param">valor</|DSML|parameter>
    </|DSML|invoke>
    </|DSML|function_calls>

Problemas tratados:
    1. DSML pode aparecer em qualquer rodada (não só na primeira)
    2. DSML pode vir misturado com texto ("Vou listar... <DSML>...")
    3. O modelo pode usar nomes de parâmetros diferentes dos definidos
       ex: "diretorio" em vez de "caminho"

Uso:
    from shared.tool_runner import executar_com_tools, ToolRunner
"""

import json
import re
import os
import sys
from typing import Callable
from dotenv import load_dotenv


def _find_and_load_env():
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        env_path = os.path.join(current, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            return
        current = os.path.dirname(current)
    load_dotenv()

_find_and_load_env()


# ── Parser DSML ───────────────────────────────────────────────

# Mapeamento de nomes alternativos de parâmetros que o DeepSeek
# costuma usar para os nomes corretos das funções Python.
_PARAM_ALIASES = {
    "diretorio" : "caminho",
    "directory" : "caminho",
    "path"      : "caminho",
    "folder"    : "caminho",
    "ext"       : "extensao",
    "extension" : "extensao",
    "operation" : "operacao",
    "num_a"     : "a",
    "num_b"     : "b",
    "number_a"  : "a",
    "number_b"  : "b",
}

def _normalizar_args(args: dict) -> dict:
    """Substitui aliases de parâmetros pelos nomes corretos."""
    return {_PARAM_ALIASES.get(k, k): v for k, v in args.items()}

def _contem_dsml(texto: str) -> bool:
    return bool(texto) and "invoke" in texto and ("DSML" in texto or "function_calls" in texto)

def _parse_dsml(texto: str) -> list[dict]:
    """
    Parseia blocos DSML e retorna [{name, arguments}].
    Trata múltiplos separadores que o DeepSeek pode usar.
    """
    calls = []

    # Normaliza variações do separador DSML
    # IMPORTANTE: substituir </... antes de <... para nao quebrar o regex
    texto_norm = texto
    texto_norm = texto_norm.replace("</｜DSML｜", "</DSML_")
    texto_norm = texto_norm.replace("<｜DSML｜",  "<DSML_")
    texto_norm = texto_norm.replace("</|DSML|",   "</DSML_")
    texto_norm = texto_norm.replace("<|DSML|",    "<DSML_")

    # Extrai blocos invoke
    invokes = re.findall(
        r'<DSML_invoke name="([^"]+)">(.*?)</DSML_invoke>',
        texto_norm,
        re.DOTALL
    )

    for nome_funcao, corpo in invokes:
        argumentos = {}
        params = re.findall(
            r'<DSML_parameter name="([^"]+)"[^>]*>(.*?)</DSML_parameter>',
            corpo,
            re.DOTALL
        )
        for nome_param, valor in params:
            valor = valor.strip()
            try:    valor = int(valor)
            except ValueError:
                try: valor = float(valor)
                except ValueError: pass
            argumentos[nome_param] = valor

        calls.append({
            "name"     : nome_funcao,
            "arguments": _normalizar_args(argumentos)
        })

    return calls

def _limpar_dsml(texto: str) -> str:
    """Remove todas as tags DSML, mantendo só o texto legível."""
    texto = re.sub(r'<[｜|]DSML[｜|][^>]*>', '', texto)
    texto = re.sub(r'</[｜|]DSML[｜|][^>]*>', '', texto)
    return texto.strip()


# ── Cliente OpenAI-compatible ─────────────────────────────────

def _get_client():
    from openai import OpenAI
    from shared.llm_factory import PROVIDER, MODEL

    if PROVIDER == "deepseek":
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        ), MODEL
    elif PROVIDER == "ollama":
        return OpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ), MODEL
    else:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), MODEL


# ── Executor de resultados ────────────────────────────────────

def _adicionar_resultados(messages, calls, funcoes, verbose):
    """Executa as funções e adiciona os resultados ao histórico."""
    for call in calls:
        nome = call["name"]
        args = call["arguments"]

        if nome not in funcoes:
            resultado = {"erro": f"Funcao '{nome}' nao encontrada. Disponiveis: {list(funcoes.keys())}"}
        else:
            try:
                resultado = funcoes[nome](**args)
            except TypeError as e:
                # Parâmetro errado mesmo após alias — reporta claramente
                resultado = {"erro": f"Parametros invalidos para '{nome}': {e}. Recebido: {args}"}
            except Exception as e:
                resultado = {"erro": str(e)}

        if verbose:
            print(f"  -> {nome}({args})")
            print(f"     = {resultado}")

        messages.append({
            "role"        : "tool",
            "tool_call_id": call["id"],
            "content"     : json.dumps(resultado, ensure_ascii=False)
        })


# ── Executor principal ────────────────────────────────────────

def executar_com_tools(
    pergunta: str,
    tools: list,
    funcoes: dict,
    system: str = None,
    verbose: bool = True,
    max_iteracoes: int = 8
) -> str:
    """
    Ciclo completo de function calling com loop robusto.

    Continua executando enquanto o modelo retornar tool calls
    (formato OpenAI padrão ou DSML do DeepSeek).
    """
    client, model = _get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": pergunta})

    for iteracao in range(max_iteracoes):

        resposta = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        message = resposta.choices[0].message
        content = message.content or ""

        # ── Caso A: tool_calls padrão OpenAI ─────────────────
        if message.tool_calls:
            if verbose:
                print(f"[iter {iteracao+1}] {len(message.tool_calls)} tool call(s) — OpenAI")
            messages.append(message)
            calls = [
                {
                    "id"       : tc.id,
                    "name"     : tc.function.name,
                    "arguments": _normalizar_args(json.loads(tc.function.arguments))
                }
                for tc in message.tool_calls
            ]
            _adicionar_resultados(messages, calls, funcoes, verbose)
            continue

        # ── Caso B: DSML no content ───────────────────────────
        if _contem_dsml(content):
            if verbose:
                print(f"[iter {iteracao+1}] DSML detectado")
            calls_dsml = _parse_dsml(content)

            if not calls_dsml:
                if verbose:
                    print("  DSML nao parseavel — retornando texto limpo")
                return _limpar_dsml(content)

            calls = [
                {"id": f"dsml_{iteracao}_{i}", **c}
                for i, c in enumerate(calls_dsml)
            ]

            # Injeta no histórico como tool_calls padrão
            messages.append({
                "role"      : "assistant",
                "content"   : None,
                "tool_calls": [
                    {
                        "id"      : c["id"],
                        "type"    : "function",
                        "function": {
                            "name"     : c["name"],
                            "arguments": json.dumps(c["arguments"])
                        }
                    }
                    for c in calls
                ]
            })
            _adicionar_resultados(messages, calls, funcoes, verbose)
            continue

        # ── Caso C: resposta de texto — fim do ciclo ─────────
        if verbose:
            label = "direta" if iteracao == 0 else "final"
            print(f"[iter {iteracao+1}] Resposta {label}")
        return content

    if verbose:
        print(f"[aviso] Limite de {max_iteracoes} iteracoes atingido")
    return ""


# ── Classe ToolRunner ─────────────────────────────────────────

class ToolRunner:
    """
    Interface OO para registrar ferramentas e fazer perguntas.

    Exemplo:
        runner = ToolRunner(system="Você é um assistente técnico.")
        runner.registrar(tool_def, funcao_python)
        resposta = runner.perguntar("Quantos notebooks existem?")
    """

    def __init__(self, system: str = None, verbose: bool = True):
        self.system  = system
        self.verbose = verbose
        self.tools   = []
        self.funcoes = {}

    def registrar(self, tool_definition: dict, funcao: Callable):
        self.tools.append(tool_definition)
        self.funcoes[tool_definition["function"]["name"]] = funcao
        return self

    def perguntar(self, pergunta: str) -> str:
        return executar_com_tools(
            pergunta=pergunta,
            tools=self.tools,
            funcoes=self.funcoes,
            system=self.system,
            verbose=self.verbose
        )

    def __repr__(self):
        nomes = [t["function"]["name"] for t in self.tools]
        return f"ToolRunner(tools={nomes})"

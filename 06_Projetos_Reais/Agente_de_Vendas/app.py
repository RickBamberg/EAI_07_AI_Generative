"""
app.py — Servidor Flask do Agente de Vendas
Interface chatbot com registro de requisições em SQLite e arquivo de log.
"""

import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from openai import OpenAI
import agente

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '..', '.env'))

app = Flask(__name__)
app.secret_key = os.urandom(24)

llm = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url='https://api.deepseek.com'
)
LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-chat')

# Banco de registro de requisições
REG_DB = os.path.join(BASE_DIR, 'logs', 'requisicoes.db')

# ─────────────────────────────────────────────────────────────
# BANCO DE REGISTRO DE REQUISIÇÕES
# ─────────────────────────────────────────────────────────────

def init_registro_db():
    """Cria tabela de requisições se não existir."""
    os.makedirs(os.path.dirname(REG_DB), exist_ok=True)
    with sqlite3.connect(REG_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requisicoes (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                sessao_id     TEXT,
                timestamp     TEXT,
                pergunta      TEXT,
                resposta      TEXT,
                tem_grafico   INTEGER DEFAULT 0,
                tem_excel     INTEGER DEFAULT 0,
                tools_usadas  TEXT,
                iteracoes     INTEGER,
                duracao_ms    INTEGER
            )
        """)
        conn.commit()

def registrar_requisicao(sessao_id: str, pergunta: str, resultado: dict,
                          duracao_ms: int):
    """Persiste uma requisição no banco de registro."""
    with sqlite3.connect(REG_DB) as conn:
        conn.execute("""
            INSERT INTO requisicoes
              (sessao_id, timestamp, pergunta, resposta,
               tem_grafico, tem_excel, tools_usadas, iteracoes, duracao_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sessao_id,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            pergunta,
            resultado['texto'][:500],
            1 if resultado['grafico'] else 0,
            1 if resultado['excel']   else 0,
            ', '.join(resultado['tools_usadas']) or 'nenhuma',
            resultado['iteracoes'],
            duracao_ms,
        ))
        conn.commit()

# ─────────────────────────────────────────────────────────────
# SESSÕES EM MEMÓRIA (histórico por sessão)
# ─────────────────────────────────────────────────────────────

sessoes: dict[str, list] = {}  # sessao_id → lista de mensagens

# ─────────────────────────────────────────────────────────────
# ROTAS
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Recebe pergunta, chama agente, retorna resposta JSON."""
    import time
    data = request.get_json()
    pergunta   = data.get('pergunta', '').strip()
    sessao_id  = data.get('sessao_id', 'default')

    if not pergunta:
        return jsonify({'erro': 'Pergunta vazia'}), 400

    # Recuperar ou criar histórico da sessão
    historico = sessoes.get(sessao_id, [])

    t0 = time.time()
    try:
        resultado = agente.responder(
            pergunta=pergunta,
            historico=historico,
            llm=llm,
            modelo=LLM_MODEL,
        )
    except Exception as e:
        return jsonify({'erro': str(e)}), 500

    duracao_ms = int((time.time() - t0) * 1000)

    # Atualizar histórico da sessão
    historico.append({'role': 'user',      'content': pergunta})
    historico.append({'role': 'assistant', 'content': resultado['texto']})
    sessoes[sessao_id] = historico[-40:]  # manter últimas 20 trocas

    # Registrar no banco
    registrar_requisicao(sessao_id, pergunta, resultado, duracao_ms)

    return jsonify({
        'texto':       resultado['texto'],
        'grafico':     resultado['grafico'],
        'excel':       os.path.basename(resultado['excel']) if resultado['excel'] else None,
        'iteracoes':   resultado['iteracoes'],
        'tools':       resultado['tools_usadas'],
        'duracao_ms':  duracao_ms,
    })


@app.route('/limpar', methods=['POST'])
def limpar():
    """Limpa o histórico da sessão."""
    data = request.get_json()
    sessao_id = data.get('sessao_id', 'default')
    sessoes.pop(sessao_id, None)
    return jsonify({'ok': True})


@app.route('/download/<nome_arquivo>')
def download(nome_arquivo):
    """Serve arquivo Excel para download."""
    caminho = os.path.join(BASE_DIR, 'outputs', nome_arquivo)
    if os.path.exists(caminho):
        return send_file(caminho, as_attachment=True)
    return jsonify({'erro': 'Arquivo não encontrado'}), 404


@app.route('/historico')
def historico():
    """Retorna últimas 50 requisições do banco de registro."""
    with sqlite3.connect(REG_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, sessao_id, timestamp, pergunta, resposta,
                   tem_grafico, tem_excel, tools_usadas, iteracoes, duracao_ms
            FROM requisicoes
            ORDER BY id DESC LIMIT 50
        """).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/status')
def status():
    """Healthcheck — verifica conexão com banco de dados."""
    try:
        with agente.get_connection() as conn:
            n = conn.execute('SELECT COUNT(*) FROM vendas').fetchone()[0]
        return jsonify({
            'ok': True,
            'modelo': LLM_MODEL,
            'vendas': n,
            'db_path': agente.DB_PATH,
        })
    except Exception as e:
        return jsonify({'ok': False, 'erro': str(e)}), 500


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_registro_db()
    print(f'✅ Banco de dados : {agente.DB_PATH}')
    print(f'✅ Modelo LLM     : {LLM_MODEL}')
    print(f'✅ Registro req.  : {REG_DB}')
    print(f'🚀 Iniciando Flask em http://localhost:5000')
    app.run(debug=True, port=5000, use_reloader=False)

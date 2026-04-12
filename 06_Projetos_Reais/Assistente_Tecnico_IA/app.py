"""
app.py
======
Servidor Flask do Assistente Técnico IA.

Rotas:
    GET  /          → interface de chat
    POST /chat      → processa pergunta, retorna JSON
    POST /limpar    → apaga histórico global
    GET  /status    → saúde do servidor e índice RAG
"""

from flask import Flask, render_template, request, jsonify
from assistente import responder, limpar_historico, _get_indice, _pronto

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data     = request.get_json()
    pergunta = (data or {}).get('pergunta', '').strip()

    if not pergunta:
        return jsonify({'erro': 'Pergunta vazia.'}), 400

    try:
        resultado = responder(pergunta)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'erro': str(e)}), 500


@app.route('/limpar', methods=['POST'])
def limpar():
    return jsonify(limpar_historico())


@app.route('/status')
def status():
    indice = _get_indice() if _pronto else None
    return jsonify({
        'status' : 'ok',
        'pronto' : _pronto,
        'rag'    : 'carregado' if indice else 'carregando...',
        'chunks' : indice['faiss'].ntotal if indice else 0,
    })


if __name__ == '__main__':
    print('\n' + '='*55)
    print('  Assistente Técnico IA')
    print('  http://localhost:5000')
    print('='*55 + '\n')
    app.run(debug=True, port=5000)

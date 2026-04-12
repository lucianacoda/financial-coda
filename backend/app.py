import logging
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, request, render_template

# ==========================================
# CONFIGURAÇÕES E CAMINHOS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = BASE_DIR / 'models' / 'modelo_Naive_Bayes_inadimplencia.pkl'
CAMINHO_TEMPLATES = BASE_DIR / 'frontend'

app = Flask(__name__, template_folder=str(CAMINHO_TEMPLATES))
logging.basicConfig(level=logging.INFO)

# ==========================================
# VARIÁVEIS EM MEMÓRIA (HISTÓRICO)
# ==========================================
historico_consultas = []  # Lista que guardará as últimas 10 consultas

# ==========================================
# CARGA DO MODELO
# ==========================================
try:
    modelo = joblib.load(CAMINHO_MODELO)
    logging.info(f"Modelo carregado com sucesso de: {CAMINHO_MODELO}")
except FileNotFoundError:
    logging.error(f"Modelo não encontrado em {CAMINHO_MODELO}")
    modelo = None

# ==========================================
# DE-PARA DE VARIÁVEIS
# ==========================================
FEATURES_MODELO = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

FEATURES_TELA = {
    'LIMIT_BAL': 'Limite de Crédito Disponível (R$)',
    'AGE': 'Idade do Cliente (Anos)',
    'PAY_0': 'Status do Último Pagamento (0=Em dia, 1=Atrasado)',
    'BILL_AMT1': 'Valor da Última Fatura (R$)',
    'PAY_AMT1': 'Valor Pago no Último Mês (R$)'
}


# ==========================================
# ROTAS DA APLICAÇÃO
# ==========================================
@app.route('/')
def home():
    """Renderiza a página inicial com o formulário e o histórico."""
    return render_template('index.html', features=FEATURES_TELA, predicao=None, historico=historico_consultas)


@app.route('/predict', methods=['POST'])
def predict():
    """Processa o formulário, faz a inferência e salva no histórico."""
    global historico_consultas

    if not modelo:
        return render_template('index.html', features=FEATURES_TELA, predicao="Erro interno: Modelo não carregado.",
                               cor="danger", historico=historico_consultas)

    try:
        dados_completos = {f: 0.0 for f in FEATURES_MODELO}

        for chave in FEATURES_TELA.keys():
            valor_digitado = request.form.get(chave)
            if valor_digitado:
                dados_completos[chave] = float(valor_digitado)

        dados_df = pd.DataFrame([dados_completos], columns=FEATURES_MODELO)
        predicao = modelo.predict(dados_df)[0]

        resultado_texto = "Alto Risco" if predicao == 1 else "Bom Pagador"
        cor = "danger" if predicao == 1 else "success"

        # ------------------------------------------
        # GRAVAÇÃO NO HISTÓRICO EM MEMÓRIA
        # ------------------------------------------
        registro = {
            'hora': datetime.now().strftime("%H:%M:%S"),
            'idade': int(dados_completos['AGE']),
            'limite': f"R$ {dados_completos['LIMIT_BAL']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
            'resultado': resultado_texto,
            'cor': cor
        }

        # Adiciona no começo da lista (posição 0)
        historico_consultas.insert(0, registro)
        # Mantém apenas os 10 primeiros
        historico_consultas = historico_consultas[:10]

        resultado_banner = "ALTO RISCO DE INADIMPLÊNCIA (Atenção)" if predicao == 1 else "BOM PAGADOR (Crédito Aprovado)"

        return render_template('index.html', features=FEATURES_TELA, predicao=resultado_banner, cor=cor,
                               historico=historico_consultas)

    except ValueError as e:
        return render_template('index.html', features=FEATURES_TELA,
                               predicao="Erro: Por favor, insira apenas números válidos.", cor="warning",
                               historico=historico_consultas)
    except Exception as e:
        return render_template('index.html', features=FEATURES_TELA, predicao=f"Erro inesperado: {str(e)}",
                               cor="warning", historico=historico_consultas)


if __name__ == '__main__':
    app.run(debug=True)
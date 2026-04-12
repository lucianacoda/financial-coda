from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# CONFIGURAÇÕES E LIMITES (THRESHOLDS)
# ==========================================
MIN_ACURACIA = 0.70
MIN_F1_SCORE = 0.35

# Resolve o caminho a partir da localização deste arquivo de teste para a raiz do projeto
# Ajuste o `.parents[X]` de acordo com a profundidade da pasta de testes
BASE_DIR = Path(__file__).resolve().parents[2]

CAMINHO_MODELO = BASE_DIR / 'models' / 'modelo_Naive_Bayes_inadimplencia.pkl'
CAMINHO_DADOS = BASE_DIR / 'notebooks' / 'data' / 'UCI_Credit_Card.csv'


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture(scope="module")
def ambiente_teste():
    """
    Fixture: Carrega o modelo e uma amostra de 2000 dados reais.
    O scope="module" garante que o modelo e o CSV sejam carregados apenas uma vez
    para todos os testes, deixando a execução bem mais rápida.
    """
    assert CAMINHO_MODELO.exists(), f"Erro: Modelo não encontrado em {CAMINHO_MODELO}"
    assert CAMINHO_DADOS.exists(), f"Erro: Base de dados não encontrada em {CAMINHO_DADOS}"

    modelo = joblib.load(CAMINHO_MODELO)
    df = pd.read_csv(CAMINHO_DADOS)

    # Padronização e Limpeza conforme diretrizes de anonimização
    if 'default.payment.next.month' in df.columns:
        df = df.rename(columns={'default.payment.next.month': 'target'})

    # Garantindo que dados suprimidos no treino também sejam removidos no teste
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Amostragem para teste rápido
    df_amostra = df.tail(2000)

    X_test = df_amostra.drop(columns=['target'])
    y_test = df_amostra['target']

    return modelo, X_test, y_test


# ==========================================
# TESTES
# ==========================================
def test_desempenho_minimo_modelo(ambiente_teste):
    """Garante que as métricas de negócio não caiam abaixo do limiar aceitável."""
    # Arrange
    modelo, X_test, y_test = ambiente_teste

    # Act
    y_pred = modelo.predict(X_test)
    acuracia_atual = accuracy_score(y_test, y_pred)
    f1_atual = f1_score(y_test, y_pred)

    # Assert
    assert acuracia_atual >= MIN_ACURACIA, f"Acurácia ({acuracia_atual:.2f}) abaixo de {MIN_ACURACIA}."
    assert f1_atual >= MIN_F1_SCORE, f"F1-Score ({f1_atual:.2f}) abaixo de {MIN_F1_SCORE}."


def test_formato_saida_modelo(ambiente_teste):
    """Garante que a saída do modelo será sempre um binário (0 ou 1)."""
    modelo, X_test, _ = ambiente_teste

    y_pred = modelo.predict(X_test.head(10))

    for predicao in y_pred:
        assert predicao in [0, 1], f"Saída inesperada: {predicao}."


def test_quantidade_features(ambiente_teste):
    """Garante que o modelo está recebendo o número correto de features."""
    modelo, X_test, _ = ambiente_teste

    qtd_esperada = modelo.n_features_in_
    qtd_enviada = X_test.shape[1]

    assert qtd_esperada == qtd_enviada, f"Espera {qtd_esperada} features, recebeu {qtd_enviada}."
# 🏦 Sistema Inteligente de Análise de Risco de Crédito

**Pontifícia Universidade Católica do Rio de Janeiro - PUC-Rio**  
**Pós-Graduação Lato Sensu em Engenharia de Software**  
**Projeto da Disciplina:** Sprint – Qualidade de Software, Segurança e Sistemas Inteligentes  
**Aluna:** Luciana Coda Sant'Anna Gomes  
**Tecnologias:** Python 3.13, Flask, Scikit-Learn, PyTest, Bootstrap 5
---

## 📌 Visão Geral
Este projeto é uma solução **Full Stack** voltada para a predição de inadimplência em cartões de crédito. A solução utiliza modelos clássicos de Machine Learning (como Naive Bayes e KNN) treinados com o dataset *Credit Card Defaults* da UCI. O sistema permite que um analista de crédito insira dados simplificados de um cliente e obtenha, em tempo real, uma classificação de risco baseada em inteligência artificial.

## 🗂️ Estrutura do Projeto
A organização do repositório segue as melhores práticas de modularização, separando os artefatos de dados, modelos, notebooks de pesquisa e a aplicação servidora:

```text
financial-coda/
├── .venv/                                      # Ambiente virtual do projeto
├── backend/                                    # Camada de Back-End (Flask)
│   └── app.py                                  # Lógica do servidor e carregamento do modelo
├── frontend/                                   # Camada de Front-End (HTML/CSS/JS)
│   └── index.html                              # Front-End (Interface Web)
├── notebooks/                                  # Camada de Ciência de Dados
│   ├── data/                                   # Diretório de datasets utilizados
│   └── credit-card-delinquency.ipynb           # Pipeline de treinamento e validação
├── models/                                     # Camada de Modelos e seus testes
│   ├── tests/                                  # Diretório de testes automatizados para os modelos
│   │   └── test_models.py                      # Testes automatizados (PyTest)
│   └── modelo_Naive_Bayes_inadimplencia.pkl    # Modelo serializado
├── requirements.txt                            # Lista de dependências do sistema
└── README.md                                   # Documentação do projeto
```

## 🔒 Desenvolvimento de Software Seguro e Anonimização
O projeto foi desenvolvido sob a ótica da proteção de dados pessoais e privacidade, equilibrando a utilidade do modelo preditivo com a segurança dos titulares. Foram aplicadas as seguintes técnicas de **Anonimização** baseadas no material oficial da disciplina:

### 1. Supressão de Identificadores (Notebook)
Durante o processo de preparação dos dados (`credit-card-delinquency.ipynb`), foi aplicada a técnica de **Supressão** na coluna `ID`. Identificadores permitem a identificação direta do titular; ao suprimi-los, reduzimos o risco de inferência explícita. Como o `ID` não possui utilidade estatística para os algoritmos, sua remoção garante a privacidade sem impactar a precisão do modelo.

### 2. Supressão de Atributos e Minimização (Aplicação)
Na interface da aplicação (`app.py`), implementamos a técnica de **Supressão** de *quase-identificadores* e atributos sensíveis. Embora o modelo interno processe 23 variáveis para manter sua clareza e precisão, o formulário de entrada solicita apenas 5 variáveis fundamentais. Ao ocultar dados históricos e secundários da interface, evitamos que operadores realizem inferências implícitas por meio do cruzamento de variáveis, garantindo o princípio da **Minimização de Dados**.

## 🧪 Testes de Qualidade e Desempenho
A aplicação conta com uma suíte de testes automatizados via **PyTest** para assegurar que o modelo atenda aos requisitos de desempenho estabelecidos antes de ser servido:

* **Validação de Thresholds:** O teste garante que o modelo em produção mantenha uma Acurácia mínima de 70% e um F1-Score mínimo de 35% sobre dados reais.
* **Integridade de Saída:** Verifica se o modelo retorna estritamente as classes binárias (0 ou 1) para evitar quebras no Front-End.
* **Controle de Versão:** Impede a implantação de um novo modelo caso ele apresente regressão de desempenho comparado aos limites definidos.

## 🚀 Como Executar

**1. Configuração do Ambiente Virtual e Instalação de Dependências:**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

pip install --upgrade pip && pip install -r requirements.txt
```

**2. Execução dos Testes:**
```bash
pytest models/tests/test_models.py -v
```

**3. Inicialização da Aplicação:**
```bash
python backend/app.py
```

**Acesse a aplicação no navegador em: `http://127.0.0.1:5000`**
# Spam Detector — UNIMAR

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

Detecção de e-mails spam com Machine Learning, usando o dataset **Spambase (UCI)**. Projeto da disciplina **Fábrica de Projetos** do curso de Inteligência Artificial da UNIMAR (2026).

### 🚀 [Teste o app online](https://spam-detector-unimar.streamlit.app/)

---

## Resultado

Foram comparados modelos de classificação, e o **SVM** foi escolhido como modelo final, com **~92% de acurácia**.

## Estrutura

| Pasta / arquivo | Conteúdo |
|---|---|
| `notebooks/` | Análise exploratória, treino e comparação dos modelos |
| `model/` | Modelo treinado usado pelo app |
| `reports/` | Relatório técnico do projeto |
| `app.py` | App em Streamlit para classificar e-mails |
| `requirements.txt` | Dependências do projeto |

## Como rodar localmente

```bash
git clone https://github.com/luisbobsled/spam-detector-unimar.git
cd spam-detector-unimar
pip install -r requirements.txt
streamlit run app.py
```

## Autor

**Luis Filipe Seixas de Souza**: [LinkedIn](https://www.linkedin.com/in/luisseixasdesouza/) · [GitHub](https://github.com/luisbobsled)

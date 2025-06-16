# BIOFTIR

**BIOFTIR** é uma aplicação web interativa desenvolvida em Python com [Streamlit](https://streamlit.io/), voltada para análise de dados espectrais (especialmente FTIR) com foco em aplicações biomédicas e explicabilidade de modelos de machine learning.

---

## Funcionalidades

* Pipeline completo: pré-processamento, redução de dimensionalidade (PCA/MNF), seleção de atributos, classificação (ML/DL), explicabilidade.
* Visualização de espectros, gráficos interativos, matrizes de confusão, métricas detalhadas.
* Download de resultados, modelos e explicações.
* Ajustável para diferentes conjuntos de dados espectrais biomédicos.

---

## Instalação

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/pdodonto/AG_FTIR.git
   cd AG_FTIR
   ```

2. **Crie um ambiente virtual (recomendado):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Execução do App

Execute o comando abaixo no terminal dentro da pasta do projeto:

```bash
streamlit run app.py
```


Após a execução, o navegador será aberto automaticamente com a interface da aplicação.
Se não abrir, acesse manualmente: [http://localhost:8501](http://localhost:8501)

---

## Organização dos Arquivos

* `app.py`           – Arquivo principal Streamlit da aplicação.
* `utils/`           – Funções auxiliares, pré-processamento, modelos, etc.
* `requirements.txt` – Lista de dependências Python.
* `README.md`        – Este guia de uso.
* `data/`            – Pasta (opcional) para seus dados espectrais de entrada.

---


## Requisitos

* Python 3.8 ou superior
* Navegador moderno (Chrome, Firefox, Edge...)

---


## Citação

Se utilizar BIOFTIR em trabalhos científicos, cite o repositório e entre em contato para parcerias ou suporte!

---


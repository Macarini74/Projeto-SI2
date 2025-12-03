# 🧠 Sistemas Inteligentes 2: Análise de Dados Varejistas com IA

## 📜 Visão Geral do Projeto

Este projeto, desenvolvido para a disciplina de **Sistemas Inteligentes 2**, foca na aplicação de algoritmos de **Inteligência Artificial (IA)** e **Machine Learning (ML)** para analisar um conjunto de dados de um comércio varejista.

O objetivo é extrair *insights* valiosos, como **padrões de compra**, **segmentação de clientes**, e possivelmente **previsões de vendas**, utilizando técnicas como *clustering*, *classificação* e/ou *regressão*.

A interface do projeto é construída utilizando o **Streamlit**, permitindo uma visualização **interativa** e acessível dos resultados e modelos de IA.

---

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar o ambiente e rodar o projeto localmente.

### 1. Clonar o Repositório

Primeiro, clone este repositório para a sua máquina local:

```bash
git clone [URL_DO_SEU_REPOSITORIO]
cd [NOME_DO_SEU_REPOSITORIO]
````

### 2. Configurar o Ambiente Virtual (venv)

É **altamente recomendado** utilizar um ambiente virtual (`venv`) para isolar as dependências do projeto.

**Criação e Ativação:**

```bash
# Cria o ambiente virtual
python -m venv venv
````

# Ativa o ambiente virtual (Linux/macOS)
source venv/bin/activate

# Ativa o ambiente virtual (Windows)
.\venv\Scripts\activate
````

### 3. Instalar as Dependências

Com o ambiente virtual ativado, utilize o arquivo `requirements.txt` para instalar todas as dependências necessárias:

```bash
pip install -r requirements.txt
````

### 4. Rodar a Aplicação

Para iniciar o projeto, execute o comando do Streamlit na raiz do diretório (onde o arquivo `main.py` está localizado):

```bash
streamlit run main.py
````

A aplicação será aberta automaticamente no seu navegador. Caso isso não ocorra, acesse a URL exibida no seu terminal (geralmente http://localhost:8501).

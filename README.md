# Tech Challenge 1 – Sistema de IA para Suporte ao Diagnóstico

Este projeto implementa um sistema inteligente de suporte ao diagnóstico para um hospital universitário, utilizando **Python**, **Machine Learning** e **Visão Computacional**.

O objetivo é construir uma base de IA capaz de:
- Classificar exames e dados clínicos em “tem ou não tem a doença”.
- Analisar imagens médicas com redes neurais convolucionais (CNN).
- Gerar métricas, gráficos e interpretações (feature importance, SHAP) para apoiar o médico.

---

## Estrutura do Projeto

No diretório `Tech Challenge 1/` estão os principais componentes:

- `config.py`  
  Centraliza os caminhos para todos os datasets e diretórios de saída.

- `tabular_pipeline.py`  
  Implementa os pipelines de ML para dados **tabulares**:
  - Diagnóstico de **câncer de mama** (maligno x benigno).  
  - Diagnóstico de **diabetes**.  
  - Análise de **social media** (viral x não viral).

- `vision_pipeline.py`  
  Implementa os pipelines de **visão computacional** com CNN:
  - Detecção de **pneumonia** em radiografias de tórax.  
  - Infraestrutura para detecção de **câncer de mama** em mamografias (depende das imagens do CBIS-DDSM estarem presentes).

- `main.py`  
  Script principal com interface de linha de comando para executar as tarefas.

- `requirements.txt`  
  Lista das principais dependências Python.

- `Dockerfile`  
  Definição de imagem Docker para executar todos os experimentos.

- `results/` (criado em tempo de execução)  
  Contém gráficos, relatórios de métricas e modelos salvos.

Os datasets são lidos diretamente das pastas:

- `Diagnostico Cancer Mama Dataset/diagnosticoCancerMama.csv`
- `Diagnostico Diabetes Dataset/diagnosticoDiabetes.csv`
- `Social Media/social_media_viral_content_dataset.csv`
- `RaioX Pneumonia/chest_xray/...`
- `Imagens Cancer Mama/csv/...` (metadados da CBIS-DDSM)

---

## Requisitos

- Python 3.11+ instalado.
- Pip disponível no PATH.

Principais bibliotecas (todas listadas em `Tech Challenge 1/requirements.txt`):

- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `opencv-python`
- `tensorflow`
- `shap`

---

## Instalação (ambiente local)

No diretório raiz do projeto (`Challenge 1`):

```bash
cd "c:\Users\f.eduardo.de.souza\Desktop\POS TECH\Challenge 1"
pip install -r "Tech Challenge 1/requirements.txt"
```

Se preferir, crie antes um ambiente virtual (`venv` ou similar) e execute o comando dentro dele.

---

## Execução com Python

No diretório raiz:

```bash
cd "c:\Users\f.eduardo.de.souza\Desktop\POS TECH\Challenge 1"
python "Tech Challenge 1/main.py" --task <task>
```

Valores possíveis para `--task`:

- `all`  
  Executa **todas** as tarefas:
  - Modelos tabulares: câncer de mama, diabetes, social media.  
  - Modelos de visão: pneumonia e pipeline de mamografia.

- `tabular`  
  Executa **apenas dados tabulares**:
  - Câncer de mama.  
  - Diabetes.  
  - Social media.

- `vision`  
  Executa **apenas visão computacional**:
  - CNN para pneumonia.  
  - Pipeline de mamografia (treina apenas se as imagens estiverem disponíveis).

- `diabetes`  
  Executa somente o pipeline completo de **diabetes**.

Exemplos:

```bash
# Executar tudo (tabular + visão)
python "Tech Challenge 1/main.py" --task all

# Executar apenas diagnósticos tabulares
python "Tech Challenge 1/main.py" --task tabular

# Executar apenas visão computacional
python "Tech Challenge 1/main.py" --task vision

# Executar apenas diabetes
python "Tech Challenge 1/main.py" --task diabetes
```

---

## Execução com Docker

No diretório raiz:

```bash
cd "c:\Users\f.eduardo.de.souza\Desktop\POS TECH\Challenge 1"

docker build -t tech-challenge-1 -f "Tech Challenge 1/Dockerfile" .
docker run --rm tech-challenge-1
```

O comando acima executa automaticamente:

```bash
python "Tech Challenge 1/main.py" --task all
```

Se desejar mapear os resultados para fora do container:

```bash
docker run --rm -v "%cd%/Tech Challenge 1/results:/app/Tech Challenge 1/results" tech-challenge-1
```

(Em Linux/Mac, ajuste o caminho `-v` conforme necessário.)

---

## Saída de Resultados

Todos os resultados são salvos em:

```text
Tech Challenge 1/results/
```

Principais subpastas:

- `results/tabular/cancer_mama/`
  - `correlacao_cancer_mama.png`
  - `relatorio_classificacao_cancer_mama.txt`
  - `matriz_confusao_cancer_mama.png`
  - `importancia_features_cancer_mama.png`
  - `shap_cancer_mama.png` (se SHAP estiver instalado)

- `results/tabular/diabetes/`
  - `correlacao_diabetes.png`
  - `relatorio_classificacao_diabetes.txt`
  - `matriz_confusao_diabetes.png`
  - `importancia_features_diabetes.png`
  - `shap_diabetes.png`

- `results/tabular/social_media/`
  - `correlacao_social_media.png`
  - `relatorio_classificacao_social_media.txt`
  - `matriz_confusao_social_media.png`
  - `importancia_features_social_media.png`
  - `shap_social_media.png`

- `results/visao_computacional/pneumonia/`
  - `melhor_modelo_pneumonia.keras`
  - `metricas_pneumonia.txt`

- `results/visao_computacional/cancer_mama/`
  - `mamografia_nao_treinada.txt` (quando as imagens não estão disponíveis).  
  - Quando as imagens forem adicionadas corretamente, o pipeline passa a gerar também `metricas_mamografia.txt`.

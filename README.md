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

---

# Tech Challenge 2 – Rotas (Otimização de Rotas Médicas)

Este segundo projeto adiciona um sistema de **otimização de rotas** para distribuição de medicamentos e insumos, com foco em um cenário hospitalar com **restrições realistas** (prioridade, capacidade, autonomia e múltiplos veículos). A solução combina:

- **Algoritmos Genéticos (GA)** para resolver um problema de roteamento inspirado no TSP/VRP.
- **Baselines/heurísticas** para comparação de desempenho.
- **LLMs (opcional)** para gerar relatórios e instruções em linguagem natural via Ollama local (ver `llm.py`).

O código principal do Tech Challenge 2 está no diretório:

- `Tech Challenge 2 - Rotas/`

## Estrutura do Projeto (Tech Challenge 2)

No diretório `Tech Challenge 2 - Rotas/`:

- `main.py`  
  Interface de linha de comando (CLI) para gerar instância, otimizar rotas, visualizar e gerar relatórios.

- `routes.py`  
  Núcleo do problema: modelagem dos dados (Local, Entrega, Veículo), avaliação (fitness), restrições (capacidade/autonomia/prioridade), baselines e solver GA.

- `visualization.py`  
  Visualização das rotas em PNG usando matplotlib (opcional).

- `llm.py`  
  Geração de texto (relatório e perguntas/respostas) e integração opcional com LLM via HTTP (Ollama: `/api/generate`). Provider suportado neste build: Ollama.

- `config.py`  
  Caminhos padrão e diretório de resultados.

- `requirements.txt`  
  Dependências mínimas do TC2 (atualmente, `matplotlib` para visualização).

Saídas são geradas em:

- `Tech Challenge 2 - Rotas/results/`

## Modelo do Problema (Otimização de Rotas)

O projeto trabalha com uma instância composta por:

- **Depósito (DEPOT)**: ponto de partida e retorno.
- **Locais**: pontos (x, y) representando unidades/endereços.
- **Entregas**: associadas a um local, com:
  - `demand` (demanda/carga)
  - `priority` (`critical` ou `regular`)
- **Veículos**: cada um com:
  - `capacity` (capacidade máxima)
  - `max_distance` (autonomia máxima da rota)

### Restrições implementadas

- **Múltiplos veículos (VRP)**: uma solução gera uma rota por veículo.
- **Capacidade**: a soma das demandas atendidas por um veículo não pode exceder `capacity`.
- **Autonomia**: a distância total da rota do veículo não pode exceder `max_distance`.
- **Prioridade**: entregas `critical` são penalizadas se ocorrerem tarde no sequenciamento (incentiva atendimento mais cedo).
- **Entregas não alocadas**: se faltar veículo/capacidade/autonomia para atender tudo, o fitness recebe penalidade alta.

## Código Base de TSP (Reuso no TC2)

O repositório contém um código base de **TSP com Algoritmo Genético** em:

- `genetic_algorithm_tsp-main/genetic_algorithm_tsp-main/`

Esse código base é utilizado diretamente no Tech Challenge 2 para:

- Gerar população inicial (`generate_random_population`)
- Ordenar população por fitness (`sort_population`)
- Crossover OX (`order_crossover`)
- Mutação (`mutate`)

O relatório do TC2 inclui uma seção “Base Utilizada (TSP/GA)” com o caminho do repositório e a licença.

## Requisitos (Tech Challenge 2)

- Python 3.11+ instalado
- Pip disponível no PATH

Dependências:

- Para rodar otimização/relatório sem plot: apenas Python padrão.
- Para gerar a imagem de rotas (`routes.png`): instalar `matplotlib` via `Tech Challenge 2 - Rotas/requirements.txt`.

## Instalação (Tech Challenge 2)

No diretório raiz do repositório:

```bash
pip install -r "Tech Challenge 2 - Rotas/requirements.txt"
```

Observação: se você não quiser visualização, o projeto ainda gera `solution.json` e `report.txt`. O comando `--task all` tenta visualizar, mas ignora a etapa caso as dependências estejam ausentes.

## Execução com Python (Tech Challenge 2)

No diretório raiz:

```bash
python "Tech Challenge 2 - Rotas/main.py" --task <task>
```

### Tasks disponíveis

- `all`  
  Executa: `generate` + `optimize` + `visualize` + `report`.  
  Se a visualização não puder ser carregada (matplotlib ausente), ela é ignorada e o relatório ainda é gerado.

- `generate`  
  Gera uma instância sintética e salva em `results/instance.json`.

- `optimize`  
  Executa o algoritmo genético para roteamento e salva em `results/solution.json`.

- `visualize`  
  Gera `results/routes.png` (se houver dependências).

- `report`  
  Gera `results/report.txt`, contendo:
  - Base utilizada (TSP/GA)
  - Comparativo de desempenho (GA vs baselines)
  - Resumo da solução e instruções por veículo

- `ask`  
  Responde uma pergunta em linguagem natural sobre a instância/rotas e salva em `results/answer.txt` (pode usar LLM se habilitada).

### Parâmetros principais

- `--population` (default: 80): tamanho da população do GA  
- `--generations` (default: 250): número de gerações  
- `--mutation` (default: 0.3): probabilidade de mutação  
- `--benchmark-random` (default: 200): quantidade de rotas aleatórias para o baseline “melhor de N aleatórias”  
- `--use-llm`: força geração do relatório via LLM (se configurada)  
- `--instance`, `--solution`, `--plot`: caminhos de entrada/saída  
- `--seed`: semente de aleatoriedade (reprodutibilidade)  

## Comparativo de Desempenho (Baselines)

O relatório do TC2 calcula e imprime um comparativo entre:

- GA (solução final)
- Heurística: Nearest Neighbor
- Heurística: Prioridade Primeiro (critical antes de regular, com nearest neighbor em cada grupo)
- Heurística: Melhor de N rotas aleatórias (N configurável)

Cada linha inclui:

- `objetivo` = distância base + penalidades (fitness total)
- `base` = distância total percorrida (sem penalidades)
- `penalidades` = soma das penalidades (capacidade/autonomia/prioridade/unassigned)
- `viavel` = indicador booleano

## Integração com LLM (Opcional)

O projeto suporta geração de relatórios e respostas via LLM por variáveis de ambiente, usando Ollama local.

### Ollama (local)

Defina:

- `LLM_ENABLE=1`
- `LLM_PROVIDER=ollama`
- `LLM_MODEL=deepseek-r1:8b` (recomendado) ou `llama3.1:8b` (alternativa leve)

Opcional:

- `OLLAMA_HOST=http://localhost:11434`
- `LLM_TEMPERATURE=0.2`
- `LLM_NUM_PREDICT=900`
- `LLM_HTTP_TIMEOUT=1200` (default recomendado; aumente se houver “timed out”)

Então rode:

```bash
python "Tech Challenge 2 - Rotas/main.py" --task report --use-llm
```

Notas:
- Em caso de falha, é gerado `Tech Challenge 2 - Rotas/results/report_llm_error.txt` com a causa (timeout, modelo ausente, resposta vazia/“thinking”).
- Modelos muito grandes podem falhar por memória; prefira `deepseek-r1:8b` ou `llama3.1:8b`.
- Este build suporta apenas Ollama.

---

## Glossário (TC2)

- TSP (Traveling Salesman Problem / Problema do Caixeiro Viajante)  
  Problema clássico de encontrar a menor rota que visita todos os pontos e retorna ao início.  
  Tipicamente um único “veículo”, sem restrições como capacidade/autonomia.

- VRP (Vehicle Routing Problem / Problema de Roteamento de Veículos)  
  Extensão prática do TSP: vários veículos saindo de um depósito, com restrições reais  
  (capacidade, autonomia, prioridades, janelas de tempo). Objetivo pode incluir penalidades.

- GA (Genetic Algorithm / Algoritmo Genético)  
  Técnica de otimização baseada em evolução. Mantém uma população de soluções; melhora por seleção,  
  crossover (recombinação) e mutação. Pode usar elitismo e semente para reprodutibilidade.

- Fitness (Função objetivo / Aptidão)  
  Métrica que o algoritmo minimiza. No TC2 é `distância + penalidades`, onde penalidades cobrem:  
  `capacity` (carga acima do limite), `autonomy` (distância acima da autonomia),  
  `priority` (entregas críticas tardias/ausentes) e `unassigned` (entregas não alocadas).

- Baseline (Linha de base)  
  Métodos simples para comparação de desempenho:  
  Nearest Neighbor, Prioridade Primeiro, Melhor de N Aleatórias.  
  Servem para validar que o GA agrega valor frente a heurísticas.

- LLM (Large Language Model)  
  Modelo de linguagem usado para sintetizar relatórios e respostas. No projeto, via **Ollama** local  
  (ex.: `LLM_MODEL="deepseek-r1:8b"` ou `LLM_MODEL="llama3.1:8b"`). Controlado por variáveis de ambiente (`LLM_ENABLE`, `LLM_PROVIDER`, `LLM_MODEL`, `LLM_HTTP_TIMEOUT`).

- Ollama  
  Servidor local de modelos. Endpoint `/api/generate`. Configurável por `OLLAMA_HOST`.  
  Permite rodar modelos sem depender de serviços externos.

- Seed (Semente)  
  Valor que fixa aleatoriedade para reprodutibilidade (`--seed`).  
  Garante execuções determinísticas em demonstrações/avaliações.

- Depósito (DEPOT)  
  Ponto inicial/final das rotas. Todas as rotas saem e retornam a este ponto.

- Instância  
  Conjunto com depósito, locais, entregas e veículos. Serializada em `results/instance.json`.

- Plano de rotas  
  Rotas por veículo e métricas associadas (custo total, penalidades, viabilidade).  
  Serializado em `results/solution.json` e detalhado em `results/report.txt`.
  Quando LLM está habilitado, arquivos adicionais são gerados:
  - `results/report_llm.txt` (relatório textual via LLM)
  - `results/report_llm_error.txt` (diagnóstico quando falhar)
  - `results/answer_llm.txt` (resposta via LLM para `--task ask`, se configurada)
  - `results/answer_llm_error.txt` (diagnóstico quando falhar)

---

# Tech Challenge 3 – Assistente Virtual Médico (LangChain + LangGraph)

Este projeto cria um assistente clínico interno com:

- **Fine-tuning** (LoRA/PEFT) de um LLM com dados internos/sintéticos (scripts de pipeline);
- **LangChain** para RAG (protocolos internos e evidência externa opcional) e contextualização com dados estruturados do paciente;
- **LangGraph** para orquestrar fluxos seguros (policy → checagens → alertas → resposta final);
- **Segurança** (não prescrição), **logging/auditoria** e **explainability** (fontes na resposta);
- **Frontend** web em estilo chat (lembrando WhatsApp), servido pelo backend.

O código do TC3 fica em:

- `Tech Challenge 3 - Assistente/`

## Instalação (Tech Challenge 3)

No diretório raiz do repositório:

```bash
pip install -r "Tech Challenge 3 - Assistente/requirements.txt"
```

## Executar (local)

1) Subir o backend (FastAPI) e a UI:

```bash
cd "Tech Challenge 3 - Assistente/backend"
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

2) Abrir no navegador:

- `http://127.0.0.1:8000/`

## Configurar LLM (Claude / Anthropic)

Para usar Claude (Anthropic) como LLM do assistente:
- Defina `ANTHROPIC_API_KEY` no ambiente
- Defina `TC3_LLM_PROVIDER=anthropic`
- Defina `TC3_ANTHROPIC_MODEL=claude-sonnet-4-6` (ou o nome exato do modelo habilitado na sua conta)

Exemplo (PowerShell):

```bash
set ANTHROPIC_API_KEY=SEU_TOKEN
set TC3_LLM_PROVIDER=anthropic
set TC3_ANTHROPIC_MODEL=claude-sonnet-4-6
```

O sistema cria automaticamente (se não existir):
- Banco SQLite sintético em `Tech Challenge 3 - Assistente/data/patients.db`
- Vetorstore persistido em `Tech Challenge 3 - Assistente/data/vectorstore/`
- Log de auditoria em `Tech Challenge 3 - Assistente/logs/audit.jsonl`

## Executar com Docker

No diretório `Tech Challenge 3 - Assistente/`:

```bash
docker compose up --build
```

Depois, acessar:

- `http://localhost:8000/`

## Fine-tuning (LoRA) e avaliação

Gerar dados sintéticos (se quiser recriar):

```bash
python "Tech Challenge 3 - Assistente/scripts/make_synthetic_dataset.py"
```

Treinar adapter LoRA:

```bash
python "Tech Challenge 3 - Assistente/scripts/finetune_lora.py" --train_jsonl "Tech Challenge 3 - Assistente/data/synthetic/train.jsonl"
```

Avaliar geração (smoke test):

```bash
python "Tech Challenge 3 - Assistente/scripts/evaluate.py" --adapter "Tech Challenge 3 - Assistente/artifacts/lora_adapter"
```

Avaliar o serviço rodando (end-to-end, via API):

```bash
python "Tech Challenge 3 - Assistente/scripts/evaluate_assistant.py" --base_url "http://127.0.0.1:8000"
```

Para usar o modelo fine-tunado no backend (inference via HuggingFace):
- `TC3_LLM_PROVIDER=hf`
- `TC3_HF_MODEL_ID=<modelo_base>`
- `TC3_HF_ADAPTER_PATH=Tech Challenge 3 - Assistente/artifacts/lora_adapter`

## RAG externo (PubMedQA) – opcional

Conversão do PubMedQA (`ori_pqal.json`) para uso no assistente:

```bash
python "Tech Challenge 3 - Assistente/scripts/ingest_pubmedqa.py" --src_json "c:\Users\felip\Downloads\pubmedqa-master\pubmedqa-master\data\ori_pqal.json"
```

Isso gera:
- `Tech Challenge 3 - Assistente/data/protocols_external/pubmedqa_pqal.jsonl` (RAG externo)
- `Tech Challenge 3 - Assistente/data/synthetic/pubmedqa_train.jsonl` (SFT opcional)

Para ativar no backend:
- `TC3_PROTOCOL_EXTERNAL_DIR=Tech Challenge 3 - Assistente/data/protocols_external`

Comportamento de segurança:
- O retrieval prioriza o índice interno e só consulta o PubMedQA quando não houver match interno suficiente (limiares ajustáveis por `TC3_RAG_INTERNAL_DISTANCE_MAX` e `TC3_RAG_INTERNAL_SIMILARITY_MIN`).

Relatório técnico do TC3:

- `Tech Challenge 3 - Assistente/RELATORIO_TECNICO.md`

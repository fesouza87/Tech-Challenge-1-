# Plano de Execucao - Tech Challenge 4

## 1. Leitura dos requisitos

O desafio exige uma solucao de **monitoramento preventivo multimodal** para o contexto hospitalar, cobrindo:

- **video**: analise de cirurgias e sessoes de fisioterapia;
- **audio**: transcricao e deteccao de alteracoes vocais;
- **texto**: prescricoes, evolucao clinica e laudos;
- **series temporais**: sinais vitais e eventos de internacao;
- **nuvem**: integracao com Azure Cognitive Services;
- **tempo real**: geracao de alertas acionaveis para a equipe medica.

## 2. Interpretacao tecnica

O desafio nao pede apenas classificacao por modalidade. Ele pede uma **plataforma integrada** que:

1. recebe dados heterogeneos;
2. transforma cada modalidade em eventos estruturados;
3. detecta anomalias por contexto clinico;
4. consolida evidencias em um score de risco;
5. dispara alertas explicaveis e auditaveis.

Por isso, a melhor abordagem nao e construir um modelo unico, e sim uma arquitetura modular com **fusao tardia de evidencias**.

## 3. Melhor estrategia de execucao

### 3.1 Principio de implementacao

A estrategia mais segura para entrega e dividir a solucao em **quatro camadas**:

1. **ingestao**: upload, leitura e padronizacao;
2. **analise por modalidade**: video, audio, texto e vitais;
3. **deteccao de anomalias**: regras + modelos estatisticos/ML;
4. **orquestracao de alertas**: consolidacao, priorizacao e resposta.

### 3.2 Reaproveitamento do Tech Challenge 3

O `Tech Challenge 3 - Assistente` ja oferece padroes uteis:

- configuracao centralizada;
- backend em FastAPI;
- pipeline orientado a etapas;
- logging/auditoria;
- integracao com servicos de IA;
- camada final de resposta explicavel.

Para o Tech Challenge 4, a recomendacao e evoluir esse desenho para um **motor multimodal**.

## 4. Arquitetura proposta

```text
Entrada multimodal
  -> API/Fila de ingestao
  -> Normalizacao por modalidade
  -> Analise especializada
  -> Eventos estruturados
  -> Deteccao de anomalias
  -> Fusao multimodal
  -> Alertas e dashboard
  -> Auditoria e historico clinico
```

## 5. Analise por requisito

### 5.1 Analise de video

**Objetivo**
- detectar movimentos fora do esperado;
- identificar eventos ou objetos de risco;
- gerar relatorios automáticos.

**Abordagem recomendada**
- usar `YOLOv8` para detectar objetos, instrumentos, pessoas e zonas de interesse;
- usar `OpenPose` ou alternativa equivalente para keypoints corporais e postura;
- extrair features por frame/janela:
  - angulos articulares;
  - amplitude de movimento;
  - velocidade e assimetria;
  - permanencia em areas criticas;
- comparar a execucao observada contra:
  - padroes basais;
  - protocolos de procedimento;
  - faixas esperadas por exercicio.

**Saidas sugeridas**
- score de anomalia por trecho;
- timeline de eventos;
- frames destacados;
- relatorio final com "desvios detectados".

### 5.2 Analise de audio

**Objetivo**
- transcrever consultas;
- identificar sinais de fadiga, dispneia, disartria ou stress;
- detectar termos clinicos criticos.

**Abordagem recomendada**
- usar `Azure Speech to Text` para transcricao;
- usar `Azure Text Analytics` para extrair entidades, frases criticas e sentimento;
- extrair features acusticas locais com `librosa`:
  - energia;
  - pitch;
  - jitter/shimmer aproximados;
  - pausas longas;
  - taxa de fala;
- opcionalmente segmentar falantes para separar medico/paciente.

**Saidas sugeridas**
- texto transcrito;
- lista de termos/sintomas criticos;
- score de alteracao vocal;
- alerta semantico + alerta acustico.

### 5.3 Deteccao de anomalias

**Objetivo**
- detectar alteracoes em sinais vitais, prescricoes e movimentacao;
- alertar a equipe medica em tempo real.

**Abordagem recomendada**
- sinais vitais:
  - z-score robusto;
  - EWMA;
  - Isolation Forest;
  - deteccao por janela deslizante;
- prescricoes:
  - regras clinicas e inconsistencias temporais;
  - mudancas abruptas de dose/classe/frequencia;
  - comparacao com protocolo e historico;
- movimentacao:
  - tempo excessivo parado;
  - padroes anormais de deslocamento;
  - risco de queda ou retorno motor insuficiente.

**Saidas sugeridas**
- severidade do alerta;
- motivo;
- modalidade geradora;
- evidencias usadas;
- recomendacao operacional para triagem humana.

## 6. Fusao multimodal

### 6.1 Por que fusao tardia

A fusao tardia e a melhor escolha para esse contexto porque:

- cada modalidade tem latencia e qualidade diferentes;
- nem todos os pacientes terao todas as modalidades disponiveis;
- a explicabilidade fica melhor ao manter evidencias separadas;
- o sistema continua funcional mesmo se um servico externo falhar.

### 6.2 Logica de fusao sugerida

Cada pipeline gera um evento padronizado:

```json
{
  "patient_id": "P001",
  "timestamp": "2026-06-24T14:10:00Z",
  "modality": "audio",
  "anomaly_score": 0.81,
  "severity": "high",
  "signal": "fadiga_vocal",
  "evidence": ["pausas longas", "queda de energia", "termo: falta de ar"]
}
```

Depois, o motor de fusao consolida:

- score por modalidade;
- score agregado;
- persistencia temporal;
- correlacao entre eventos;
- contexto do paciente e prioridade clinica.

## 7. Mapeamento para Azure

### Servicos recomendados

- **Azure Speech to Text**: transcricao de audio clinico;
- **Azure Text Analytics / Language**: sentimentos, entidades e frases-chave;
- **Azure Blob Storage**: armazenamento de audio e video;
- **Azure Functions** ou **Azure Container Apps**: execucao de pipelines;
- **Azure Event Hubs** ou fila equivalente: ingestao near real-time;
- **Azure Monitor / Application Insights**: observabilidade;
- **Azure AI Vision**: opcional para enriquecer analise visual, sem substituir o pipeline medico dedicado.

### Estrategia pratica

- manter inferencias sensiveis de video em container Python proprio;
- usar Azure para servicos prontos de transcricao e NLP;
- armazenar metadados e alertas localmente ou em banco gerenciado;
- desacoplar servicos externos por adaptadores.

## 8. Estrutura de projeto recomendada

```text
src/
  api/
    main.py
    schemas.py
    routes_ingestion.py
    routes_alerts.py
  alerts/
    engine.py
    severity.py
    notifier.py
  fusion/
    aggregator.py
    risk_scoring.py
  ingestion/
    queue.py
    storage.py
    normalizers.py
  pipelines/
    audio/
      azure_stt.py
      acoustic_features.py
      anomaly_audio.py
    text/
      azure_text_analytics.py
      prescriptions.py
      clinical_notes.py
    video/
      yolo_detector.py
      pose_estimation.py
      anomaly_video.py
    vitals/
      timeseries_features.py
      anomaly_vitals.py
  shared/
    config.py
    models.py
    audit.py
    utils.py
tests/
docs/
```

## 9. Roadmap por fases

### Fase 1 - Fundacao

- criar estrutura do projeto;
- definir schemas de eventos multimodais;
- criar configuracao e adaptadores Azure;
- implementar auditoria e modelo de alertas.

### Fase 2 - MVP funcional

- pipeline de audio com Azure STT + Text Analytics;
- pipeline de texto para prescricoes e evolucao clinica;
- pipeline de vitais com anomalias em series temporais;
- API para upload e consulta de alertas.

### Fase 3 - Video

- deteccao com YOLOv8;
- estimacao de pose;
- anomalias em fisioterapia ou procedimento simulado;
- geracao de relatorio automatico por video.

### Fase 4 - Fusao e tempo real

- agregacao multimodal;
- score unico de risco;
- notificacao em tempo real;
- dashboard consolidado.

### Fase 5 - Entrega academica

- consolidar relatorio tecnico;
- gerar casos demonstrativos;
- preparar roteiro do video de 15 minutos.

## 10. MVP recomendado

Para maximizar chance de entrega com qualidade, o MVP deve priorizar:

1. `audio + texto + vitais` antes de `video`;
2. relatorios claros e alertas auditaveis;
3. pelo menos um fluxo Azure funcionando de ponta a ponta;
4. video com caso controlado de fisioterapia, nao cirurgia real;
5. dados sinteticos ou publicos anonimizados.

Motivo: video e a parte mais custosa computacionalmente e a mais dificil de validar rapidamente.

## 11. Datasets e dados de demonstracao

### Fontes sugeridas

- `https://physionet.org/` para sinais vitais e series temporais;
- `https://research.google.com/audioset/` para audio;
- bases publicas de pose/acao humana para video;
- dados sinteticos internos para prescricoes e evolucao clinica.

### Recomendacao pratica

- usar dados reais publicos para vitais;
- usar poucos samples curados para audio;
- criar videos curtos controlados para postura/fisioterapia;
- sintetizar prescricoes e evolucao clinica com casos plausiveis.

## 12. Riscos principais

- **custo computacional do video**: resolver com amostragem por frames e janelas;
- **dependencia de servicos Azure**: resolver com adaptadores e fallback local;
- **falta de dados clinicos integrados**: resolver com esquema comum de eventos;
- **latencia em tempo real**: resolver com processamento assincrono e severidade por fila;
- **explicabilidade**: resolver com armazenamento de evidencias e justificativas.

## 13. Definicoes de sucesso

O projeto deve demonstrar, no minimo:

- ingestao de texto, audio e video;
- deteccao de anomalias em ao menos 3 fluxos distintos;
- uso de Azure em audio/NLP;
- geracao de alertas estruturados;
- relatorio tecnico com exemplos e resultados;
- demonstracao em video com fluxo completo.

## 14. Proximos passos recomendados

1. criar o esqueleto tecnico do projeto em `src/`;
2. definir contratos de evento multimodal e alerta;
3. implementar primeiro o pipeline de `audio`;
4. implementar `vitals` e `texto`;
5. adicionar `video` com foco em fisioterapia;
6. integrar motor de fusao e dashboard.

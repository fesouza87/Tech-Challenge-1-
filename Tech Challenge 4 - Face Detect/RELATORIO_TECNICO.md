# Relatorio Tecnico - Tech Challenge 4

## 1. Visao geral

Esta solucao implementa uma base de monitoramento clinico multimodal para:

- processar audio, video, texto e sinais vitais;
- detectar anomalias clinicas precoces;
- gerar alertas estruturados;
- manter trilha de auditoria;
- integrar servicos Azure de forma opcional.

## 2. Fluxo multimodal

```text
Entrada de dados
  -> API FastAPI / dashboard web
  -> Pipeline especializado por modalidade
  -> Conversao para evento multimodal padronizado
  -> Calculo de score e severidade
  -> Geracao de alerta
  -> Resumo de risco do paciente
  -> Persistencia em auditoria e relatorios
```

### Arquitetura em camadas

- **Entrada e exposicao**
  - A solucao expoe endpoints HTTP em `src/api/` e um dashboard web em `src/static/`.
  - O frontend consome a propria API para upload de audio, importacao de sinais vitais, consulta de pacientes e leitura de alertas.

- **Processamento por modalidade**
  - Cada modalidade possui pipeline dedicado em `src/pipelines/`.
  - O objetivo e transformar entradas heterogeneas em um evento clinico padronizado contendo paciente, timestamp, sinal detectado, score, severidade, evidencias e metadados.

- **Fusao e risco**
  - Os eventos gerados alimentam a camada de fusao em `src/fusion/`, que consolida risco por paciente, modalidades ativas, severidade maxima e sinal mais recente.

- **Alertas e observabilidade**
  - A camada de `src/alerts/` e `src/ingestion/` decide se um evento deve gerar alerta acionavel, atualiza o estado global e registra auditoria.
  - No pipeline de video, essa camada tambem persiste relatorios tecnicos em `reports/video/`.

- **Apresentacao**
  - O dashboard consolida risco, evidencias, sinais vitais e alertas em tempo quase real.
  - A atualizacao ao vivo utiliza `Server-Sent Events` via `GET /api/alerts/stream`.

## 3. Modelos e tecnicas por modalidade

### Audio

- Heuristicas sobre transcript clinico
- Metricas acusticas informadas no payload
- Azure Speech to Text opcional
- Azure Text Analytics opcional

### Texto

- Regras clinicas sobre termos criticos
- Identificacao de alteracoes de prescricao
- Sinais de piora clinica

### Sinais vitais

- Regras de limiar clinico
- Tendencia temporal simplificada sobre janelas recentes

### Video

- Inferencia opcional com YOLOv8
- Pose estimation opcional com OpenPose quando configurado, inclusive via pacote oficial com `OpenPoseDemo.exe`
- Fallback para MediaPipe Pose
- Relatorio automatico persistido por analise

## 4. Saidas produzidas

- evento multimodal estruturado
- alerta com severidade e recomendacao
- resumo de risco do paciente
- auditoria em `logs/audit.jsonl`
- relatorio de video em `reports/video/*.json` e `reports/video/*.txt`

## 5. Componentes implementados

- `src/api/`
  - rotas de dashboard, pipelines, alertas, ingestao e healthcheck.
- `src/pipelines/audio/`
  - integracao com Azure Speech, Azure Text e heuristicas clinicas sobre fala/transcricao.
- `src/pipelines/text/`
  - analise de texto clinico e deteccao de termos criticos.
- `src/pipelines/video/`
  - leitura de frames com OpenCV, objetos com YOLOv8, pose com OpenPose e fallback para MediaPipe.
- `src/pipelines/vitals/`
  - processamento de sinais vitais normalizados e importacao de `.vital` com `vitaldb`.
- `src/fusion/`
  - consolidacao de eventos e resumo de risco por paciente.
- `src/shared/`
  - contratos, configuracao via `.env` e estado em memoria da aplicacao.
- `src/static/`
  - dashboard HTML/CSS/JS para demonstracao e consumo da API.

## 6. Exemplo de anomalias detectadas

- audio: fadiga vocal, sintoma respiratorio, alteracao de articulacao
- texto: termos criticos em evolucao, alteracao inesperada de prescricao
- vitals: dessaturacao, instabilidade hemodinamica
- video: intrusao em area critica, desvio postural, movimento fora do padrao

## 7. Integracao Azure

Variaveis utilizadas:

- `AZURE_SPEECH_ENDPOINT`
- `AZURE_SPEECH_KEY`
- `AZURE_TEXT_ENDPOINT`
- `AZURE_TEXT_KEY`

Comportamento:

- sem configuracao: fallback local/heuristico
- com configuracao: uso real dos SDKs Azure quando possivel

Validacoes reais executadas:

- `Azure Speech` validado com audio sintetico `speech_demo_en.wav`, retornando transcricao com sucesso
- `Azure Text Analytics` validado com analise de sentimento, frases-chave e entidades
- fluxo fim a fim do endpoint `POST /api/pipelines/audio` validado com `speech_success=true`, `azure_text_success=true` e geracao de alerta para o paciente `PAZ01`

## 8. Operacao local para demonstracao

Scripts adicionados para simplificar a preparacao da apresentacao:

- `scripts/start_tc4.ps1`
  - usa a venv `.venv_tc4`;
  - le host e porta a partir do `.env`;
  - encerra a instancia anterior se a API ja estiver em execucao;
  - sobe novamente a aplicacao;
  - espera resposta positiva do endpoint `/health`;
  - pode abrir automaticamente o dashboard no navegador.

- `scripts/stop_tc4.ps1`
  - encerra a API usando o PID salvo na ultima inicializacao;
  - remove listeners residuais na porta configurada quando necessario.

Esses scripts foram criados para reduzir risco operacional antes da gravacao e da demonstracao ao vivo.

### Passo-a-passo de uso da API

O `README.md` possui um passo-a-passo completo com exemplos em PowerShell/curl para:

- `POST /api/pipelines/audio` (JSON)
- `POST /api/pipelines/audio/upload` (multipart)
- `POST /api/pipelines/text`
- `POST /api/pipelines/vitals/vitaldb`
- `POST /api/pipelines/video`
- consultas de dashboard e alertas (`/api/dashboard/*`, `/api/alerts/*`, `/api/alerts/stream`)

Para avaliacao e apresentacao, recomenda-se utilizar o Swagger em `/docs` e validar o healthcheck em `/health` antes de iniciar a demonstracao.

## 9. Resultados de validacao sintetica

- script gerador de midia demo em `scripts/generate_demo_media.py`
- audio sintetico criado em `data/synthetic/media/consulta_demo.wav`
- audio de fala sintetica validado em `data/synthetic/media/speech_demo_en.wav`
- video sintetico criado em `data/synthetic/media/fisioterapia_demo.mp4`
- pipeline de video validado com arquivo real, processando `12` frames na API
- pose via `MediaPipe` validada localmente
- YOLOv8 validado localmente com inferencia real de objetos no evento `PV05`
- pipeline `POST /api/pipelines/video` validado com o video clinico `data/raw/video/rehab_demo_lifting_object.mp4`, gerando os relatorios `PVITALSHOW_video-7c10d2f6-9b04-458d-a146-acbe25e77b72.json/.txt`
- na validacao final, o provider configurado foi `openpose`, com `pose_enabled=true`, `pose_error=null`, `runtime_pose_deviation_score=0.0129` e `Frames JSON OpenPose: 6`
- endpoint `POST /api/pipelines/audio` validado com Azure Speech + Azure Text no evento `PAZ01`
- relatorios persistidos gerados em `reports/video/`

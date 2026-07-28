# Tech Challenge 4 - Monitoramento Clinico Multimodal

## Visao geral

**Tech Challenge 4**, com foco em **monitoramento clinico multimodal** usando:

- texto clinico e prescricoes;
- audio de consultas e interacoes com pacientes;
- video de cirurgias, fisioterapia e monitoramento assistido;
- deteccao de anomalias em tempo real;
- integracao com servicos gerenciados na nuvem, com prioridade para **Azure Cognitive Services**.


## Objetivo da fase

Construir uma base de projeto capaz de:

- ingerir dados multimodais;
- extrair sinais clinicamente relevantes;
- detectar anomalias precoces;
- consolidar alertas acionaveis para a equipe medica;
- manter rastreabilidade, seguranca e auditabilidade.


## Arquitetura

```text
Entradas clinicas
  -> API FastAPI / Dashboard HTML+JS
  -> pipeline por modalidade (audio, texto, video, vitals)
  -> extracao de sinais e enriquecimento por IA
  -> normalizacao em evento clinico unico
  -> score de risco e geracao de alerta
  -> persistencia em memoria + trilha de auditoria + relatorios
  -> consulta por API / dashboard / stream SSE
```

### Visao por camadas

1. **Camada de entrada**
   - O sistema recebe dados por endpoints FastAPI e pelo dashboard web.
   - O frontend em `src/static/` consome a propria API do projeto para disparar upload de audio, processamento de sinais vitais e consulta de pacientes, eventos e alertas.
   - As rotas HTTP ficam concentradas em `src/api/`, separadas por responsabilidade: dashboard, pipelines, alertas, ingestao e healthcheck.

2. **Camada de processamento multimodal**
   - Cada modalidade possui um pipeline proprio em `src/pipelines/`.
   - O objetivo dessa camada e transformar uma entrada heterogenea em uma estrutura padronizada com:
     - identificacao do paciente;
     - timestamp clinico;
     - sinal detectado;
     - score de anomalia;
     - severidade;
     - evidencias e metadados tecnicos.
   - Isso permite que audio, video, texto e sinais vitais sigam para as proximas etapas no mesmo formato logico.

3. **Camada de fusao e avaliacao de risco**
   - Depois que um pipeline gera um evento, o sistema calcula risco agregado do paciente em `src/fusion/`.
   - Essa etapa resume:
     - quantidade de eventos;
     - quantidade de alertas;
     - maior severidade observada;
     - modalidades ativas;
     - ultimo sinal clinico relevante.
   - O resultado alimenta o dashboard e sustenta a leitura consolidada do estado do paciente.

4. **Camada de alerta e observabilidade**
   - Os eventos processados passam por regras em `src/alerts/` e `src/ingestion/`.
   - Quando o score e a severidade justificam, o sistema gera um alerta estruturado com titulo, mensagem, evidencias e acao recomendada.
   - Em paralelo, o processamento atualiza o estado em memoria da aplicacao, escreve auditoria em `logs/audit.jsonl` e, no caso do video, gera relatorios tecnicos em `reports/video/`.

5. **Camada de apresentacao**
   - O dashboard consolida pacientes, resumo de risco, evidencias, sinais vitais e alertas em tempo quase real.
   - A atualizacao ao vivo ocorre por `Server-Sent Events` no endpoint `GET /api/alerts/stream`.
   - Essa camada foi pensada para demonstracao tecnica e para leitura rapida de contexto clinico.

### Fluxo operacional

```text
1. Usuario ou sistema envia uma entrada clinica
2. A rota correspondente chama o pipeline da modalidade
3. O pipeline executa preprocessamento e inferencia
4. O resultado e convertido em Event
5. O servico de ingestao registra o evento
6. O motor de alertas decide se ha alerta acionavel
7. O estado global do paciente e atualizado
8. Dashboard, APIs de consulta e stream SSE refletem o novo estado
```

### Arquitetura por modalidade

- `audio`
  - Entrada por transcript direto, `audio_file_path` ou upload multipart.
  - Integracao opcional com `Azure Speech` para transcricao.
  - Integracao opcional com `Azure Text Analytics` para sentimento, frases-chave e entidades.
  - Saida: evento clinico com evidencias acusticas e semanticas.

- `text`
  - Processa prescricoes, evolucoes e trechos clinicos estruturados ou livres.
  - Aplica heuristicas e enriquecimento textual para identificar termos criticos e contexto de risco.
  - Saida: evento textual normalizado para fusao multimodal.

- `video`
  - Le frames com OpenCV.
  - Usa `YOLOv8` para objetos e `OpenPose` como provedor principal de pose quando configurado, com fallback para `MediaPipe` ou heuristica.
  - Gera relatorio `.json` e `.txt` com detalhes tecnicos da inferencia.
  - Saida: evento de video com contagem de objetos, anomalias e metadados de pose.

- `vitals`
  - Processa amostras normalizadas ou importa arquivo `.vital` por `vitaldb`.
  - Construi janela temporal de sinais como SpO2, FC, PA, FR e temperatura.
  - Detecta desvios como dessaturacao e exibe tendencia no dashboard.
  - Saida: evento fisiologico e serie resumida para monitor visual.

### Mapeamento dos modulos principais

- `ingestion/`: upload, fila, batch e streaming.
- `pipelines/video/`: OpenPose, YOLOv8 e regras de eventos.
- `pipelines/audio/`: Azure Speech to Text, features acusticas e NLP.
- `pipelines/text/`: prescricoes, evolucoes, laudos e sinais criticos.
- `pipelines/vitals/`: series temporais e deteccao de desvios.
- `fusion/`: agregacao de evidencias e score de risco.
- `alerts/`: regras, severidade, roteamento e notificacao.
- `api/`: endpoints para upload, consulta e monitoramento.
- `shared/`: contratos, configuracao via `.env` e estado global da aplicacao.
- `docs/`: fluxo tecnico, resultados e roteiro de demonstracao.

### Estado e persistencia

- **Estado transiente em memoria**
  - A instancia `AppState` mantem pacientes, eventos e alertas correntes para resposta rapida do dashboard.
  - Esse modelo foi suficiente para a entrega e facilita a demonstracao local sem dependencia de banco externo.

- **Auditoria**
  - Toda execucao relevante pode ser registrada em `logs/audit.jsonl`.
  - Isso preserva rastreabilidade de entradas, eventos processados e alertas gerados.

- **Relatorios de video**
  - Cada analise de video gera artefatos em `reports/video/`, o que ajuda na validacao tecnica e na apresentacao.

### Decisoes arquiteturais da entrega

- **Separacao por modalidade** para permitir evolucao independente dos pipelines.
- **Padronizacao de evento clinico** para unificar a leitura multimodal.
- **Fallbacks controlados** para manter o sistema funcional mesmo sem todos os provedores externos ativos.
- **Dashboard acoplado a mesma API** para simplificar execucao local e demonstracao.
- **Observabilidade simples e objetiva** com SSE, auditoria em arquivo e relatorios persistidos.

## Documentos desta pasta

- `RELATORIO_TECNICO.md`: consolidacao tecnica da solucao implementada.
- `MATRIZ_ADERENCIA.md`: matriz requisito x status x evidencia.

## Integracao Azure no audio

O pipeline de `audio` ja suporta integracao real com Azure em modo opcional:

- `Azure Speech` para transcricao quando `audio_file_path` for informado;
- `Azure Text Analytics` para sentimento, frases-chave e entidades sobre o transcript final;
- fallback automatico para modo heuristico quando SDK, credenciais ou arquivo nao estiverem disponiveis.

### Variaveis de ambiente

Defina em `.env`:

```text
AZURE_SPEECH_ENDPOINT=<endpoint-do-recurso-speech>
AZURE_SPEECH_KEY=<chave-do-recurso-speech>
AZURE_SPEECH_REGION=<regiao-opcional-como-brazilsouth>
AZURE_TEXT_ENDPOINT=<endpoint-do-recurso-language>
AZURE_TEXT_KEY=<chave-do-recurso-language>
```

Observacao:

- se `AZURE_SPEECH_REGION` estiver preenchida, o projeto usa autenticacao por regiao;
- se ela estiver vazia e o endpoint for regional, como `https://brazilsouth.api.cognitive.microsoft.com/`, o projeto tenta derivar a regiao automaticamente;
- para endpoints customizados do recurso, o projeto continua aceitando `AZURE_SPEECH_ENDPOINT`.

### Comportamento do pipeline

- Se `transcript` vier no request, ele e usado diretamente;
- Se `audio_file_path` vier e o Azure Speech estiver configurado, o sistema tenta transcrever o audio;
- Se a transcricao Azure funcionar, ela passa a ser a fonte principal do texto;
- Se o Azure Text estiver configurado, o transcript final e enriquecido com sentimento, frases-chave e entidades;
- Se algo falhar, a API continua respondendo e devolve os detalhes de fallback em `details`.

### Endpoint principal

- `POST /api/pipelines/audio`

Payload minimo por transcript:

```json
{
  "patient_id": "P300",
  "timestamp": "2026-06-24T16:00:00Z",
  "transcript": "Paciente com cansaco, falta de ar e fala arrastada.",
  "language": "pt-BR"
}
```

Payload com tentativa de Azure Speech:

```json
{
  "patient_id": "P301",
  "timestamp": "2026-06-24T16:05:00Z",
  "audio_file_path": "c:\\temp\\consulta.wav",
  "language": "pt-BR"
}
```

## Integracao de video

O pipeline de `video` aceita arquivo de video local e tenta executar inferencia real em modo opcional:

- `YOLOv8` via `ultralytics`;
- `OpenPose` quando o ambiente local estiver configurado;
- fallback para `MediaPipe Pose` quando OpenPose nao estiver disponivel;
- fallback heuristico quando nenhum provider estiver acessivel.

### Pre-requisitos para OpenPose no Windows

Para reproduzir a validacao do provider `openpose` no ambiente Windows, foi necessario preparar previamente:

- binarios do OpenPose com `OpenPoseDemo.exe` acessivel em um diretorio local;
- `CMake`, utilizado na preparacao do ambiente nativo;
- `Visual Studio Build Tools` com toolchain MSVC compatível (`v143`);
- modelos do OpenPose presentes dentro da pasta configurada;
- variavel `TC4_OPENPOSE_DIR` apontando para a raiz dessa instalacao.

Observacao:

- quando esses pre-requisitos nao estao disponiveis, o projeto continua funcional com fallback para `MediaPipe`, conforme o provider configurado em `TC4_VIDEO_POSE_PROVIDER`.

### Variaveis de ambiente de video

```text
TC4_VIDEO_REPORT_DIR=reports/video
TC4_YOLO_MODEL_PATH=
TC4_OPENPOSE_DIR=
TC4_VIDEO_POSE_PROVIDER=auto
```

### Endpoint principal

- `POST /api/pipelines/video`

Exemplo:

```json
{
  "patient_id": "PV01",
  "timestamp": "2026-06-24T17:00:00Z",
  "procedure_type": "fisioterapia",
  "video_file_path": "c:\\temp\\sessao.mp4",
  "expected_objects": ["person"],
  "expected_people": 1,
  "frame_stride": 10,
  "max_frames": 24
}
```

### Relatorios gerados

Cada analise de video gera artefatos persistidos em `reports/video/`:

- relatorio `.json` com evento e detalhes tecnicos;
- relatorio `.txt` com resumo legivel para avaliacao.

### Midia demo para validacao local

O projeto possui um gerador de midia sintetica para acelerar testes locais:

```bash
.venv_tc4\Scripts\python scripts\generate_demo_media.py
```

Arquivos gerados:

- `data/synthetic/media/consulta_demo.wav`
- `data/synthetic/media/fisioterapia_demo.mp4`

Essa midia ja foi usada para validar o pipeline de `video` com arquivo real e `MediaPipe` ativo no ambiente local.

Para validar `YOLOv8`, foi utilizado um ambiente isolado de pacotes em:

- `c:\Users\felip\source\FIAP\TechChallenge1\tc4_yolo_pkgs`

Com a API executada com `PYTHONPATH` apontando para esse diretorio, o pipeline de `video` validou inferencia real de objetos no video sintetico.

## Inicializacao rapida

Para reiniciar a aplicacao inteira do TC4 com um comando so, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_tc4.ps1
```

O script:

- usa a venv `.venv_tc4`;
- le `TC4_API_HOST` e `TC4_API_PORT` do `.env`;
- encerra a instancia anterior se a porta ja estiver em uso;
- sobe a API FastAPI/Frontend novamente;
- aguarda o healthcheck responder;
- abre o dashboard no navegador.

Opcoes uteis:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_tc4.ps1 -NoBrowser
powershell -ExecutionPolicy Bypass -File .\scripts\start_tc4.ps1 -Foreground
```

Para parar a aplicacao:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_tc4.ps1
```

## Como usar a API (passo-a-passo)

### 1) Subir a aplicacao

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_tc4.ps1 -NoBrowser
```

URLs principais:

- Dashboard: `http://127.0.0.1:8010/`
- Swagger/OpenAPI: `http://127.0.0.1:8010/docs`
- Healthcheck: `http://127.0.0.1:8010/health`

### 2) Verificar se a API esta no ar

```powershell
Invoke-RestMethod http://127.0.0.1:8010/health
```

### 3) Rodar audio (JSON, sem upload)

Endpoint:

- `POST /api/pipelines/audio`

Exemplo (transcript direto):

```powershell
$payload = @{
  patient_id = "PAZMANUAL"
  timestamp = "2026-07-28T20:00:00Z"
  transcript = "Paciente com cansaco, falta de ar e fala arrastada."
  language = "pt-BR"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8010/api/pipelines/audio" `
  -ContentType "application/json" `
  -Body $payload
```

### 4) Rodar audio (upload multipart)

Endpoint:

- `POST /api/pipelines/audio/upload`

Exemplo (usando o demo do projeto):

```powershell
curl.exe -sS -X POST "http://127.0.0.1:8010/api/pipelines/audio/upload" ^
  -F "patient_id=PAZUPLOAD" ^
  -F "timestamp=2026-07-28T20:05:00Z" ^
  -F "language=en-US" ^
  -F "audio_file=@data/synthetic/media/speech_demo_en.wav"
```

### 5) Rodar texto

Endpoint:

- `POST /api/pipelines/text`

Exemplo:

```powershell
$payload = @{
  patient_id = "PTXT01"
  timestamp = "2026-07-28T20:10:00Z"
  text = "Evolucao: paciente com piora de dispneia. Ajuste de oxigenoterapia."
  language = "pt-BR"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8010/api/pipelines/text" `
  -ContentType "application/json" `
  -Body $payload
```

### 6) Rodar sinais vitais (importacao VitalDB)

Endpoint:

- `POST /api/pipelines/vitals/vitaldb`

Exemplo (deixe `vital_file_path` em branco para usar `vital/0001.vital`):

```powershell
$payload = @{
  patient_id = "PVITALSHOW"
  interval_seconds = 60
  max_samples = 24
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8010/api/pipelines/vitals/vitaldb" `
  -ContentType "application/json" `
  -Body $payload
```

### 7) Rodar video (arquivo real)

Endpoint:

- `POST /api/pipelines/video`

Exemplo (video de fisioterapia ja presente no projeto):

```powershell
$payload = @{
  patient_id = "PVITALSHOW"
  timestamp = "2026-07-28T20:20:00Z"
  procedure_type = "fisioterapia"
  video_file_path = "C:\Users\felip\source\FIAP\TechChallenge1\Tech-Challenge-1-\Tech Challenge 4 - Face Detect\data\raw\video\rehab_demo_lifting_object.mp4"
  expected_objects = @("person")
  expected_people = 1
  frame_stride = 10
  max_frames = 6
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8010/api/pipelines/video" `
  -ContentType "application/json" `
  -Body $payload
```

Resultado esperado:

- `details.report_json_path` e `details.report_txt_path` apontando para `reports/video/`
- `details.pose_provider` indicando `openpose` quando configurado

### 8) Consultar resultados no dashboard (overview/paciente)

- `GET /api/dashboard/overview`
- `GET /api/dashboard/patient/{patient_id}`

Exemplo:

```powershell
Invoke-RestMethod "http://127.0.0.1:8010/api/dashboard/overview"
Invoke-RestMethod "http://127.0.0.1:8010/api/dashboard/patient/PVITALSHOW"
```

### 9) Consultar alertas e stream ao vivo (SSE)

- Lista:
  - `GET /api/alerts`
  - `GET /api/alerts/patient/{patient_id}`
- Stream SSE:
  - `GET /api/alerts/stream`

Exemplo de stream no terminal:

```powershell
curl.exe -N "http://127.0.0.1:8010/api/alerts/stream"
```

## Monitoramento continuo

Para o requisito de acompanhamento em tempo real, a API expoe:

- `GET /api/alerts`
- `GET /api/alerts/patient/{patient_id}`
- `GET /api/alerts/stream`

O endpoint `stream` utiliza SSE (`text/event-stream`) para publicar alertas conforme eles entram no sistema.

## Estrutura inicial

```text
Tech Challenge 4 - Face Detect/
  README.md
  data/
    raw/
    processed/
    synthetic/
  docs/
  notebooks/
  src/
    api/
    alerts/
    fusion/
    ingestion/
    pipelines/
      audio/
      text/
      video/
      vitals/
  tests/
```

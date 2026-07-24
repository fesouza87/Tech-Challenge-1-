# Tech Challenge 4 - Face Detect

## Visao geral

**Tech Challenge 4**, com foco em **monitoramento clinico multimodal** usando:

- texto clinico e prescricoes;
- audio de consultas e interacoes com pacientes;
- video de cirurgias, fisioterapia e monitoramento assistido;
- deteccao de anomalias em tempo real;
- integracao com servicos gerenciados na nuvem, com prioridade para **Azure Cognitive Services**.

O nome da pasta foi mantido como solicitado, mas o escopo da solucao vai alem de "face detect": trata-se de uma plataforma multimodal de vigilancia clinica e apoio preventivo.

## Objetivo da fase

Construir uma base de projeto capaz de:

- ingerir dados multimodais;
- extrair sinais clinicamente relevantes;
- detectar anomalias precoces;
- consolidar alertas acionaveis para a equipe medica;
- manter rastreabilidade, seguranca e auditabilidade.

## Direcao recomendada

A melhor forma de execucao e **evolutiva**, reaproveitando a arquitetura do `Tech Challenge 3 - Assistente`:

1. manter um backend central em Python;
2. separar pipelines por modalidade (`texto`, `audio`, `video`, `vitals`);
3. unificar os resultados em um motor de regras/anomalias;
4. usar Azure para transcricao, NLP clinico e servicos gerenciados;
5. expor alertas e justificativas em API e dashboard.

## Arquitetura sugerida

```text
Fontes de dados
  -> ingestao multimodal
  -> preprocessamento por modalidade
  -> extracao de features e eventos
  -> deteccao de anomalias
  -> fusao multimodal
  -> motor de alertas
  -> API / dashboard / auditoria
```

### Modulos principais

- `ingestion/`: upload, fila, batch e streaming.
- `pipelines/video/`: OpenPose, YOLOv8 e regras de eventos.
- `pipelines/audio/`: Azure Speech to Text, features acusticas e NLP.
- `pipelines/text/`: prescricoes, evolucoes, laudos e sinais criticos.
- `pipelines/vitals/`: series temporais e deteccao de desvios.
- `fusion/`: agregacao de evidencias e score de risco.
- `alerts/`: regras, severidade, roteamento e notificacao.
- `api/`: endpoints para upload, consulta e monitoramento.
- `docs/`: fluxo tecnico, resultados e roteiro de demonstracao.

## Proposta de stack

- **Backend**: FastAPI
- **Orquestracao**: Python + filas assicronas
- **Video**: OpenCV, YOLOv8, OpenPose ou alternativa equivalente de pose
- **Audio**: Azure Speech to Text, librosa, pyannote opcional
- **Texto**: Azure Text Analytics / Language, regras clinicas e embeddings
- **Anomalias**: Isolation Forest, autoencoder, z-score robusto, regras clinicas
- **Persistencia**: SQLite/PostgreSQL para metadados, Blob Storage para midia
- **Observabilidade**: logs estruturados, auditoria e trilha de alertas
- **Nuvem**: Azure AI Services, Azure Blob Storage, Azure Functions ou Container Apps

## Entregaveis esperados

- codigo-fonte da solucao multimodal;
- documentacao tecnica da arquitetura e fluxo de dados;
- exemplos de processamento de audio, video e texto;
- deteccao de anomalias com alertas;
- integracao demonstravel com servicos Azure;
- roteiro para video de apresentacao de ate 15 minutos.

## Documentos desta pasta

- `PLANO_EXECUCAO.md`: analise dos requisitos e plano de implementacao.
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

O pipeline de `video` agora aceita arquivo de video local e tenta executar inferencia real em modo opcional:

- `YOLOv8` via `ultralytics`;
- `OpenPose` quando o ambiente local estiver configurado;
- fallback para `MediaPipe Pose` quando OpenPose nao estiver disponivel;
- fallback heuristico quando nenhum provider estiver acessivel.

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

Para validar `YOLOv8` nesta maquina, foi utilizado um ambiente isolado de pacotes em:

- `c:\Users\felip\source\FIAP\TechChallenge1\tc4_yolo_pkgs`

Com a API executada com `PYTHONPATH` apontando para esse diretorio, o pipeline de `video` validou inferencia real de objetos no video sintetico.

## Monitoramento continuo

Para aproximar o requisito de acompanhamento em tempo real, a API expoe:

- `GET /api/alerts`
- `GET /api/alerts/patient/{patient_id}`
- `GET /api/alerts/stream`

O endpoint `stream` utiliza SSE (`text/event-stream`) para publicar alertas conforme eles entram no sistema.

## Estrutura inicial sugerida

```text
Tech Challenge 4 - Face Detect/
  README.md
  PLANO_EXECUCAO.md
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

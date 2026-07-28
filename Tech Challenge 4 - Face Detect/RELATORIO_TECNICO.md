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
  -> Pipeline especializado por modalidade
  -> Conversao para evento multimodal padronizado
  -> Calculo de score e severidade
  -> Geracao de alerta
  -> Resumo de risco do paciente
  -> Persistencia em auditoria e relatorios
```

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

## 5. Exemplo de anomalias detectadas

- audio: fadiga vocal, sintoma respiratorio, alteracao de articulacao
- texto: termos criticos em evolucao, alteracao inesperada de prescricao
- vitals: dessaturacao, instabilidade hemodinamica
- video: intrusao em area critica, desvio postural, movimento fora do padrao

## 6. Integracao Azure

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

## 7. Limitacoes atuais

- a validacao ponta a ponta com Azure depende de credenciais reais
- o monitoramento continuo ainda e leve, via stream SSE
- o video final de demonstracao ainda precisa ser produzido

## 8. Resultados de validacao sintetica

- script gerador de midia demo em `scripts/generate_demo_media.py`
- audio sintetico criado em `data/synthetic/media/consulta_demo.wav`
- audio de fala sintetica validado em `data/synthetic/media/speech_demo_en.wav`
- video sintetico criado em `data/synthetic/media/fisioterapia_demo.mp4`
- pipeline de video validado com arquivo real, processando `12` frames na API
- pose via `MediaPipe` validada localmente
- YOLOv8 validado localmente com inferencia real de objetos no evento `PV05`
- pipeline `POST /api/pipelines/video` validado em `27/07/2026` com o video clinico `data/raw/video/rehab_demo_lifting_object.mp4`, gerando os relatorios `PVITALSHOW_video-7c10d2f6-9b04-458d-a146-acbe25e77b72.json/.txt`
- na validacao final, o provider configurado foi `openpose`, com `pose_enabled=true`, `pose_error=null`, `runtime_pose_deviation_score=0.0129` e `Frames JSON OpenPose: 6`
- a causa raiz do timeout anterior estava na amostragem: o pipeline limitava `frame_last`, mas nao passava `--frame_step` para o OpenPose CLI, fazendo o CPU processar frames demais
- endpoint `POST /api/pipelines/audio` validado com Azure Speech + Azure Text no evento `PAZ01`
- relatorios persistidos gerados em `reports/video/`

## 9. Proximos passos

- capturar relatorios e screenshots para a apresentacao final
- produzir o video de demonstracao de ate 15 minutos

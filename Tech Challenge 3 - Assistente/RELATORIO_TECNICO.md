# Tech Challenge 3 – Assistente Virtual Médico (LangChain + LangGraph)

## 1) Visão geral
Este projeto implementa um assistente clínico para uso interno em hospital, com:
- Base de conhecimento (protocolos internos) indexada para RAG;
- Consulta a dados estruturados (prontuário sintético em SQLite);
- Orquestração de fluxos com LangGraph (checagens + alertas + resposta final);
- Limites de atuação (não prescrever), logging para auditoria e respostas com fontes.

O foco é demonstrar uma arquitetura segura e extensível para personalização com dados do hospital.

## 2) Dados: preprocessing, anonimização e curadoria
### 2.1 Tipos de dados esperados
- Protocolos internos (textos e fluxos);
- Perguntas frequentes de médicos (FAQ);
- Modelos internos (laudos, receitas e procedimentos).

### 2.2 Curadoria e padronização
Padronização sugerida (JSONL):
- Protocolos: `{id, title, source, text}`
- Treino (chat): `{messages: [{role, content}, ...]}`

### 2.3 Anonimização
Foi incluído um pipeline de anonimização simples por regex em `scripts/preprocess_dataset.py`, que remove padrões típicos:
- CPF, telefone, e-mail, datas.

Para produção, recomenda-se:
- Pseudonimização determinística (hash com salt seguro) para reidentificação controlada;
- Dicionários e modelos de NER para PHI (nome, endereço, ID interno);
- Revisão humana (amostragem) e validação automatizada (detecção de PHI remanescente).

## 3) Fine-tuning (LoRA/PEFT)
### 3.1 Abordagem
O fine-tuning é feito com LoRA (PEFT) usando TRL (`SFTTrainer`) em dataset chat (`messages`), gerando um adapter (artefato leve) sobre um modelo base (ex.: TinyLlama).

Script: `scripts/finetune_lora.py`
- Entrada: `data/synthetic/train.jsonl` (ou dataset interno anonimizado)
- Saída: `artifacts/lora_adapter/` (adapter + tokenizer)

### 3.2 Observações de segurança
Mesmo com fine-tuning, a solução mantém:
- Política de não prescrição (bloqueio de solicitações por padrões);
- Respostas sempre com linguagem de apoio e validação humana.

## 4) Assistente com LangChain (RAG + dados estruturados)
### 4.1 RAG com protocolos internos
Arquivo exemplo: `data/protocols/protocols.jsonl`.
Indexação: feita automaticamente na inicialização, persistindo em `data/vectorstore/`.

Mecanismo:
- Chunking com `RecursiveCharacterTextSplitter`
- Vetorização preferencial via `sentence-transformers` (fallback para embeddings determinísticos)
- Busca por similaridade e retorno de trechos com metadados (fonte, título, id)

### 4.1.1 RAG externo (PubMedQA) com prioridade do interno
Além dos protocolos internos, o assistente pode usar evidência externa (ex.: PubMedQA) como base complementar, mantendo o protocolo interno como fonte primária.

Pipeline de ingestão (conversão PubMedQA → JSONL para RAG e SFT):
- Script: `scripts/ingest_pubmedqa.py`
- Entrada típica: `ori_pqal.json`
- Saídas:
  - RAG externo: `data/protocols_external/pubmedqa_pqal.jsonl`
  - SFT (chat): `data/synthetic/pubmedqa_train.jsonl`

Ativação no backend:
- `TC3_PROTOCOL_EXTERNAL_DIR` apontando para a pasta `data/protocols_external/` (ou outra)

Estratégia de decisão (segurança):
- Recuperação primeiro no índice interno.
- Só consulta/mescla o externo quando o melhor match interno não atingir o limiar mínimo (heurística baseada em score).
- Limiar configurável por variáveis de ambiente:
  - `TC3_RAG_INTERNAL_DISTANCE_MAX` (default 0.35)
  - `TC3_RAG_INTERNAL_SIMILARITY_MIN` (default 0.75)
O `source` retornado nas fontes passa a indicar o tipo: `interno:<arquivo>` ou `externo_pubmedqa:<arquivo>`.

### 4.2 Base estruturada (SQLite)
Banco: `data/patients.db` (sintético) com:
- `patients`, `visits`, `pending_exams`, `labs`

O snapshot do paciente é usado para contextualizar a resposta e acionar alertas.

## 5) LangGraph: fluxo de decisão automatizado
Fluxo implementado:
1. policy → bloqueia pedidos de prescrição/dose
2. patient → carrega snapshot do paciente
3. pending_exams → gera alertas de exames pendentes
4. retrieval → recupera trechos dos protocolos (RAG)
5. answer → compõe prompt com contexto + fontes + alertas e gera resposta

Diagrama (alto nível):
```
policy ──┬──> answer (se bloqueado)
         └──> patient -> pending_exams -> retrieval -> answer -> END
```

## 6) Segurança, validação e auditabilidade
### 6.1 Limites de atuação
- Não prescreve: sem dose/posologia e sem orientação de controlados
- Respostas condicionadas a validação humana
- Proíbe invenção de dados do paciente e de protocolos inexistentes

### 6.2 Logging/auditoria
Cada interação gera um evento em `logs/audit.jsonl` com:
- request_id, paciente, usuário, modelo, fontes recuperadas, flags de política e texto final

### 6.4 Provedores de LLM
O assistente suporta múltiplos provedores de LLM, configurados por variáveis de ambiente:
- Anthropic (Claude): `TC3_LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` + `TC3_ANTHROPIC_MODEL` (ex.: `claude-sonnet-4-6`)
- Ollama (local): `TC3_LLM_PROVIDER=ollama` + `TC3_LLM_MODEL` + `OLLAMA_HOST`
- HuggingFace (local): `TC3_LLM_PROVIDER=hf` + `TC3_HF_MODEL_ID` (+ `TC3_HF_ADAPTER_PATH` opcional)

### 6.3 Explainability (fontes)
O backend retorna `sources` contendo:
- `doc_id`, `title`, `source`, `excerpt`, `score`
O frontend mostra as fontes como chips e o texto orienta citações [1], [2] quando houver.

## 7) Frontend estilo “WhatsApp”
UI web servida pelo FastAPI (`backend/static`) com:
- Layout em duas colunas (lista/config à esquerda + chat à direita)
- Bolhas de conversa e seção de metadados (alertas + fontes + request_id)

## 8) Avaliação do modelo
Foi incluído `scripts/evaluate.py` para smoke test de geração:
- Avaliação qualitativa (aderência aos protocolos e linguagem segura)
- Para produção: incluir conjunto de validação e métricas (ex.: taxa de alucinação, cobertura de protocolos, robustez a prompts adversariais).

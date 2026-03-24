# Relatório Técnico: Otimização de Rotas Médicas (Tech Challenge 2)

Este relatório documenta a arquitetura, metodologia e resultados do projeto de Otimização de Rotas Médicas (TC2), que resolve um problema de roteamento inspirado em VRP com restrições realistas de logística hospitalar.

---

## 1. Objetivo e Escopo

- Otimizar rotas para distribuição de medicamentos e insumos em ambiente hospitalar.
- Considerar múltiplos veículos, capacidade de carga, autonomia (distância máxima) e prioridades de entregas.
- Comparar o algoritmo genético (GA) com heurísticas simples (baselines) e gerar relatório operacional.
- Opcionalmente sintetizar relatórios e respostas em linguagem natural via LLM (Ollama).

---

## 2. Arquitetura e Fluxo

- Diretório principal: `Tech Challenge 2 - Rotas/`
- Componentes:
  - `routes.py`: modelagem de entidades, função objetivo (fitness), restrições e solver GA.
  - `main.py`: CLI para gerar instância, otimizar, visualizar e produzir relatório.
  - `visualization.py`: renderização das rotas em PNG.
  - `llm.py`: geração de relatórios e respostas via LLM (Ollama).
  - `config.py`: caminhos padrão e diretório de resultados.
- Fluxo típico:
  - Gerar instância sintética (`generate`).
  - Otimizar com GA (`optimize`).
  - Visualizar rotas (`visualize`) — opcional.
  - Gerar relatório (`report`) — com ou sem LLM.

---

## 3. Modelagem do Problema

- Entidades (definidas em `routes.py`):
  - `Location`: ponto no plano (x, y) com identificação e nome amigável.
  - `Delivery`: entrega associada a um local, com demanda e prioridade (`critical`/`regular`).
  - `Vehicle`: veículo com `capacity` e `max_distance` (autonomia).
  - `Instance`: cenário completo com depósito, locais, entregas e veículos.
  - `RoutePlan`: rotas por veículo, custo total (distância + penalidades), indicador de viabilidade e mapa de penalidades.
- Representação da solução:
  - Tour “gigante” sobre IDs de entregas, posteriormente dividido em rotas por veículo (`split_giant_tour`).

---

## 4. Restrições e Penalidades

- Capacidade (`capacity`): soma das demandas por veículo não deve exceder sua capacidade.
- Autonomia (`max_distance`): distância total de cada rota não deve exceder a autonomia do veículo.
- Prioridade (`priority`): entregas críticas recebem penalidade se atendidas muito tarde ou ausentes.
- Não alocadas (`unassigned`): entregas não alocadas geram penalidade alta.
- Função objetivo (fitness):
  - `total_distance = base_distance + penalties`
  - `base_distance` é a soma das distâncias depósito -> entregas -> depósito por rota.
  - `penalties` agregam violações de capacidade, autonomia, prioridade e não alocação.

---

## 5. Algoritmo Genético (GA) e Código Base de TSP

- Integração com código base de TSP (CC0) em `genetic_algorithm_tsp-main/genetic_algorithm_tsp-main/`.
- Reuso de:
  - `generate_random_population` (população inicial)
  - `sort_population` (ordenação por fitness)
  - `order_crossover` (crossover OX)
  - `mutate` (mutação)
- Adaptação no TC2:
  - Avaliação (`evaluate`) considera restrições VRP e penalidades.
  - Seleção adicional e elitismo para manter os melhores indivíduos.
- Parâmetros padrão (configuráveis via `main.py`):
  - `--population=80`
  - `--generations=250`
  - `--mutation=0.3`
  - `--seed=42` (reprodutibilidade)

---

## 6. Baselines e Comparativo de Desempenho

- Baselines implementados:
  - `Nearest Neighbor`: escolhe sempre a próxima entrega mais próxima do ponto atual.
  - `Prioridade Primeiro`: atende críticas primeiro (via NN), depois regulares (via NN).
  - `Melhor de N Aleatórias`: sorteia N tours e escolhe o melhor segundo `evaluate`.
- Métricas por solução:
  - `objetivo`: distância base + penalidades.
  - `base`: distância total sem penalidades.
  - `penalidades`: soma de violações.
  - `viavel`: booleano (atende todas as restrições?).
- Exemplo real (extraído de `results/report.txt`):
  - GA (solução): objetivo=739.29 | base=639.29 | penalidades=100.00 | viavel=True
  - Heurística - Nearest Neighbor: objetivo=1120.72 | base=520.72 | penalidades=600.00 | viavel=True
  - Heurística - Prioridade Primeiro: objetivo=624.33 | base=624.33 | penalidades=0.00 | viavel=True
  - Heurística - Melhor de 200 Aleatórias: objetivo=14991.93 | base=641.93 | penalidades=14350.00 | viavel=False

---

## 7. Visualização

- `visualization.py` gera `results/routes.png`:
  - Depósito em destaque.
  - Rotas por veículo com cores distintas.
  - Útil para validar ordem de atendimento e distribuição espacial.

---

## 8. Integração com LLM (Opcional)

- Controle por variáveis de ambiente:
  - `LLM_ENABLE=1`
  - `LLM_PROVIDER=ollama`
  - `LLM_MODEL=deepseek-r1:8b` (exemplo em Ollama)
  - `OLLAMA_HOST=http://localhost:11434`
  - `LLM_TEMPERATURE`, `LLM_NUM_PREDICT`, `LLM_HTTP_TIMEOUT=1200` (parâmetros de geração/tempo; 1200 recomendado)
- `llm.py`:
  - `generate_llm_report`: agrega “Base Utilizada”, comparativo e contexto JSON para produzir relatório textual.
  - `answer_question`: Q&A sobre plano/instância com fallback sem LLM.
- Segurança:
  - Chaves nunca são versionadas; use variáveis de ambiente.
 - Provider suportado neste build: Ollama.
 - Diagnóstico de falhas:
   - Em caso de erro, é gerado `results/report_llm_error.txt` com a causa (ex.: timeout, apenas “thinking”, modelo ausente).
   - Ajuste `LLM_NUM_PREDICT` e `LLM_HTTP_TIMEOUT` (parta de 1200) para modelos que demoram a concluir (ex.: `deepseek-r1:8b`).
   - Se houver limitações de memória ou respostas vazias persistentes, usar um modelo menor (ex.: `llama3.1:8b`).

---

## 9. Execução e Artefatos

- Execução típica:
  - `python "Tech Challenge 2 - Rotas/main.py" --task all --seed 42`
- Artefatos gerados:
  - `results/instance.json` (instância)
  - `results/solution.json` (plano de rotas)
  - `results/report.txt` (comparativo e instruções operacionais)
  - `results/report_llm.txt` (relatório textual via LLM, se `LLM_ENABLE=1`)
  - `results/routes.png` (visualização, se `matplotlib` instalado)

---

## 10. Validação e Reprodutibilidade

- Reprodutibilidade garantida via `--seed`.
- Comparativo contra baselines valida que o GA agrega valor.
- Visualização e relatório operacional facilitam auditoria e comunicação com equipes.

---

## 11. Limitações e Próximos Passos

- Simplificações:
  - Sem janelas de tempo ou tempos de serviço.
  - Priorização crítica modelada por penalidade simples de atraso.
- Próximos passos:
  - Adicionar janelas de tempo (VRPTW).
  - Ajustar pesos de penalidades por dados históricos.
  - Integrar custos reais (combustível, tempo, SLA) e novos objetivos.

---

## 12. Referências de Código

- Núcleo do solver e fitness: `Tech Challenge 2 - Rotas/routes.py`
- CLI e relatório: `Tech Challenge 2 - Rotas/main.py`
- Visualização: `Tech Challenge 2 - Rotas/visualization.py`
- LLM: `Tech Challenge 2 - Rotas/llm.py`

# Relatório Técnico: Pipeline de Classificação Tabular

Este relatório detalha as estratégias adotadas, modelos utilizados e os resultados obtidos nos experimentos de classificação supervisionada para três conjuntos de dados distintos: **Diagnóstico de Câncer de Mama**, **Diagnóstico de Diabetes** e **Previsão de Viralização em Redes Sociais(Escolhido livremente de acordo com o desafio proposto)**.

---

## 1. Estratégias de Pré-processamento

Para garantir a qualidade dos dados e a compatibilidade com os algoritmos de machine learning, foi implementado um pipeline utilizando `ColumnTransformer` e `Pipeline` do Scikit-Learn.

### Estratégia Geral
O pré-processamento foi dividido por tipo de variável:

*   **Variáveis Numéricas**:
    *   **Imputação de Dados Faltantes**: Utilizou-se a **mediana** (`SimpleImputer(strategy="median")`). A mediana é preferível à média por ser mais robusta a outliers.
    *   **Padronização**: Aplicou-se o `StandardScaler` (z-score normalization) para colocar todas as variáveis na mesma escala (média 0 e desvio padrão 1) para a Regressão Logística.

*   **Variáveis Categóricas(Strings)**:
    *   **Imputação**: Valores faltantes foram preenchidos com a **moda** (valor mais frequente).
    *   **Codificação**: Utilizou-se `OneHotEncoder` para transformar categorias em vetores binários.

### Estratégia por Dataset

1.  **Câncer de Mama**:
    *   Conversão da variável alvo (`diagnosis`) de categórica ('M'/'B') para binária (1/0).
    *   Remoção de colunas identificadoras (`id`) que não possuem valor preditivo.

2.  **Diabetes**:
    *   **Tratamento de Zeros Fisiologicamente Impossíveis**: Em colunas como *Glucose*, *BloodPressure*, *SkinThickness*, *Insulin* e *BMI*, valores iguais a 0 foram tratados como dados faltantes (`NaN`) antes da imputação, pois representam ausência de medição e não um valor real.

3.  **Social Media**:
    *   **Engenharia de Features (Feature Engineering)**:
        *   Extração de componentes temporais: Hora da postagem, Dia da semana e Mês.
        *   Criação da feature `hashtags_count`: Contagem do número de hashtags na postagem.
    *   Remoção de texto bruto (`hashtags`) e identificadores (`post_id`) após a extração de informações relevantes.

---

## 2. Modelos Usados e Justificativa

Para cada dataset, foram treinados e comparados dois tipos de modelos distintos. O modelo final foi selecionado automaticamente com base no melhor **F1-Score** no conjunto de validação.

### Modelos Candidatos

1.  **Logistic Regression (Regressão Logística)**:
    *   **Por que usar?**: Serve como um *baseline* linear robusto e interpretável. É eficiente computacionalmente e funciona bem quando as classes são linearmente separáveis.
    *   **Configuração**: `max_iter=1000` 

2.  **Random Forest Classifier (Floresta Aleatória)**:
    *   Modelo de *ensemble* baseado em árvores de decisão. 
    *   **Configuração**: `n_estimators=200`, `n_jobs=-1` (paralelismo).

### Critério de Avaliação
Utilizei o **F1-Score** da classe positiva como métrica principal de decisão pois mostrou bom equilíbrio entre *Precision* e *Recall* .

---

## 3. Resultados e Interpretação

Abaixo estão os resultados obtidos no conjunto de teste (dados nunca vistos durante o treinamento).

### 3.1. Diagnóstico de Câncer de Mama
*   **Acurácia**: 98.25%
*   **F1-Score (Maligno)**: 0.9762
*   **Recall (Maligno)**: 0.9762

**Interpretação**:
O modelo obteve um bom desempenho. Um Recall de ~97.6% indica que o modelo detecta quase todos os casos malignos, o que é crítico para um diagnóstico médico. A alta acurácia sugere que as features extraídas das imagens de células são altamente discriminativas para o problema.

### 3.2. Diagnóstico de Diabetes
*   **Acurácia**: 70.78%
*   **F1-Score (Positivo)**: 0.5455
*   **Recall (Positivo)**: 0.5000

**Interpretação**:
O desempenho foi moderado. O Recall de 0.50 indica que o modelo identifica corretamente apenas 50% dos pacientes com diabetes no conjunto de teste. A precisão também não é alta (60%). Nesse caso, é necessário considerar outras métricas, como a curva ROC-AUC, para avaliar o balanceamento entre sensibilidade e especificidade.

### 3.3. Social Media (Viralização)
*   **Acurácia**: 100.00%
*   **F1-Score (Viral)**: 1.0000
*   **Recall (Viral)**: 1.0000

**Interpretação**:
O modelo atingiu a acurácia máxima no conjunto de teste.
Em um cenário real provavelmente não conseguiremos esse resultado.

---

## Conclusão Geral

O pipeline automatizado se mostrou eficaz para testar e selecionar o melhor modelo para diferentes tipos de problemas. O pré-processamento tratou dados numéricos e categóricos, e a inclusão de modelos lineares e não-lineares permitiu flexibilidade. Os resultados variaram da perfeição (Social Media) a desafios moderados (Diabetes), refletindo a complexidade de cada domínio de dados.

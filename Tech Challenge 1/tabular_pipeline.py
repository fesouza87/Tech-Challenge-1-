from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from config import CANCER_MAMA_CSV, DIABETES_CSV, SOCIAL_MEDIA_CSV, RESULTS_DIR


# Funções auxiliares para treinar e avaliar modelos de classificação com dados tabulares
def ensure_dir(path: Path) -> None:
    # Cria o diretório informado, incluindo diretórios pais, caso ainda não existam
    path.mkdir(parents=True, exist_ok=True)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path, title: str) -> None:
    # Gera e salva o mapa de calor de correlação entre as variáveis numéricas do dataset
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_classification_report(y_true, y_pred, output_path: Path, positive_label: int) -> None:
    # Calcula métricas principais (accuracy, recall, F1) e salva o relatório de classificação em texto
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=positive_label)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label)
    report = classification_report(y_true, y_pred)
    text = []
    text.append(f"Accuracy: {acc:.4f}")
    text.append(f"Recall ({positive_label}): {rec:.4f}")
    text.append(f"F1-score ({positive_label}): {f1:.4f}")
    text.append("")
    text.append(report)
    output_path.write_text("\n".join(text), encoding="utf-8")


def plot_confusion_matrix(y_true, y_pred, output_path: Path, labels) -> None:
    # Plota e salva a matriz de confusão para os rótulos informados
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(model, feature_names, output_path: Path, top_n: int = 15) -> None:
    # Gera gráfico de barras com as principais features, se o modelo expuser feature_importances_
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    scores = importances[order]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=names, orient="h")
    plt.xlabel("Importância")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def try_compute_shap_values(model, X_sample, feature_names, output_path: Path) -> None:
    # Tenta gerar gráfico de resumo SHAP; se a biblioteca não estiver instalada, apenas retorna
    try:
        import shap
    except ImportError:
        return
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_preprocessor(df: pd.DataFrame, target_column: str, drop_columns=None):
    # Monta o pré-processador, separando colunas numéricas e categóricas e aplicando imputação/escalação
    if drop_columns is None:
        drop_columns = []
    X = df.drop(columns=[target_column] + drop_columns)
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, X, numeric_features, categorical_features


def get_feature_names(preprocessor) -> list[str]:
    # Reconstrói a lista de nomes de features após o pré-processamento (incluindo colunas one-hot)
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            if not cols:
                continue
            encoder = transformer.named_steps["onehot"]
            encoded = encoder.get_feature_names_out(cols)
            feature_names.extend(encoded.tolist())
        else:
            feature_names.extend(list(cols))
    return feature_names


def run_breast_cancer_tabular(output_dir: Path) -> None:
    # Executa pipeline completo para diagnóstico de câncer de mama (benigno x maligno)
    ensure_dir(output_dir)
    df = pd.read_csv(CANCER_MAMA_CSV)
    # Converte rótulos M/B em valores binários 1/0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    df = df.drop(columns=["id"])
    plot_correlation_heatmap(df, output_dir / "correlacao_cancer_mama.png", "Correlação Câncer de Mama")
    preprocessor, X, _, _ = build_preprocessor(df, target_column="diagnosis")
    y = df["diagnosis"]
    # Separa em conjuntos de treino, validação e teste de forma estratificada
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
    )
    # Define modelos candidatos para comparação
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
    }
    best_f1 = -np.inf
    best_model = None
    fitted_models = {}
    for name, base_model in models.items():
        # Monta pipeline com pré-processamento e modelo escolhido
        clf = Pipeline(steps=[("preprocessor", preprocessor), (name, base_model)])
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_val_pred, pos_label=1)
        fitted_models[name] = clf
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf
    if best_model is None:
        return
    # Avalia o melhor modelo no conjunto de teste e salva métricas e gráficos
    y_test_pred = best_model.predict(X_test)
    save_classification_report(
        y_test,
        y_test_pred,
        output_dir / "relatorio_classificacao_cancer_mama.txt",
        positive_label=1,
    )
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        output_dir / "matriz_confusao_cancer_mama.png",
        labels=[0, 1],
    )
    rf_model = fitted_models.get("random_forest")
    if rf_model is not None:
        fitted_preprocessor = rf_model.named_steps["preprocessor"]
        feature_names = get_feature_names(fitted_preprocessor)
        estimator = rf_model.named_steps["random_forest"]
        # Gera gráficos de importância de features e explicabilidade via SHAP para o RandomForest
        plot_feature_importance(
            estimator,
            feature_names,
            output_dir / "importancia_features_cancer_mama.png",
        )
        X_sample = fitted_preprocessor.transform(X_test.iloc[:200])
        try_compute_shap_values(
            estimator,
            X_sample,
            feature_names,
            output_dir / "shap_cancer_mama.png",
        )


def run_diabetes_tabular(output_dir: Path) -> None:
    # Executa pipeline completo para diagnóstico de diabetes
    ensure_dir(output_dir)
    df = pd.read_csv(DIABETES_CSV)
    cols_with_zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    # Substitui zeros por NaN em colunas onde zero representa dado faltante
    for col in cols_with_zero_as_missing:
        df[col] = df[col].replace(0, np.nan)
    plot_correlation_heatmap(df, output_dir / "correlacao_diabetes.png", "Correlação Diabetes")
    preprocessor, X, _, _ = build_preprocessor(df, target_column="Outcome")
    y = df["Outcome"]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
    )
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
    }
    best_f1 = -np.inf
    best_model = None
    fitted_models = {}
    for name, base_model in models.items():
        clf = Pipeline(steps=[("preprocessor", preprocessor), (name, base_model)])
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_val_pred, pos_label=1)
        fitted_models[name] = clf
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf
    if best_model is None:
        return
    y_test_pred = best_model.predict(X_test)
    save_classification_report(
        y_test,
        y_test_pred,
        output_dir / "relatorio_classificacao_diabetes.txt",
        positive_label=1,
    )
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        output_dir / "matriz_confusao_diabetes.png",
        labels=[0, 1],
    )
    rf_model = fitted_models.get("random_forest")
    if rf_model is not None:
        fitted_preprocessor = rf_model.named_steps["preprocessor"]
        feature_names = get_feature_names(fitted_preprocessor)
        estimator = rf_model.named_steps["random_forest"]
        # Gera gráficos de importância de features e SHAP para o modelo de referência
        plot_feature_importance(
            estimator,
            feature_names,
            output_dir / "importancia_features_diabetes.png",
        )
        X_sample = fitted_preprocessor.transform(X_test.iloc[:200])
        try_compute_shap_values(
            estimator,
            X_sample,
            feature_names,
            output_dir / "shap_diabetes.png",
        )


def engineer_social_media_features(df: pd.DataFrame) -> pd.DataFrame:
    # Cria features derivadas de data/hora e contagem de hashtags para melhorar o modelo
    df = df.copy()
    df["post_datetime"] = pd.to_datetime(df["post_datetime"])
    df["post_hour"] = df["post_datetime"].dt.hour
    df["post_dayofweek"] = df["post_datetime"].dt.dayofweek
    df["post_month"] = df["post_datetime"].dt.month
    df["hashtags_count"] = df["hashtags"].fillna("").apply(
        lambda x: len([h for h in str(x).split() if h.startswith("#")])
    )
    df = df.drop(columns=["post_datetime"])
    return df


def run_social_media_tabular(output_dir: Path) -> None:
    # Executa pipeline para prever se um post de redes sociais será viral ou não
    ensure_dir(output_dir)
    df = pd.read_csv(SOCIAL_MEDIA_CSV)
    df = engineer_social_media_features(df)
    plot_correlation_heatmap(df, output_dir / "correlacao_social_media.png", "Correlação Social Media")
    target_column = "is_viral"
    drop_columns = ["post_id", "hashtags"]
    preprocessor, X, _, _ = build_preprocessor(df, target_column=target_column, drop_columns=drop_columns)
    y = df[target_column]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
    )
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
    }
    best_f1 = -np.inf
    best_model = None
    fitted_models = {}
    for name, base_model in models.items():
        clf = Pipeline(steps=[("preprocessor", preprocessor), (name, base_model)])
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_val_pred, pos_label=1)
        fitted_models[name] = clf
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf
    if best_model is None:
        return
    y_test_pred = best_model.predict(X_test)
    save_classification_report(
        y_test,
        y_test_pred,
        output_dir / "relatorio_classificacao_social_media.txt",
        positive_label=1,
    )
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        output_dir / "matriz_confusao_social_media.png",
        labels=[0, 1],
    )
    rf_model = fitted_models.get("random_forest")
    if rf_model is not None:
        fitted_preprocessor = rf_model.named_steps["preprocessor"]
        feature_names = get_feature_names(fitted_preprocessor)
        estimator = rf_model.named_steps["random_forest"]
        plot_feature_importance(
            estimator,
            feature_names,
            output_dir / "importancia_features_social_media.png",
        )
        X_sample = fitted_preprocessor.transform(X_test.iloc[:200])
        try_compute_shap_values(
            estimator,
            X_sample,
            feature_names,
            output_dir / "shap_social_media.png",
        )


def run_all_tabular_experiments() -> None:
    base_output = RESULTS_DIR / "tabular"
    run_breast_cancer_tabular(base_output / "cancer_mama")
    run_diabetes_tabular(base_output / "diabetes")
    run_social_media_tabular(base_output / "social_media")

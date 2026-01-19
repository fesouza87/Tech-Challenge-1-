from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import PNEUMONIA_ROOT, MAMMO_DICOM_INFO, MAMMO_MASS_TRAIN, MAMMO_JPEG_ROOT, RESULTS_DIR


# Funções auxiliares para carregar datasets de imagem e treinar CNNs de diagnóstico
def ensure_dir(path: Path) -> None:
    # Cria o diretório informado, caso ainda não exista
    path.mkdir(parents=True, exist_ok=True)


def load_images_from_directory(base_dir: Path, image_size=(224, 224)) -> tuple[np.ndarray, np.ndarray]:
    # Lê imagens JPEG em tons de cinza, redimensiona e normaliza para treino da CNN
    images = []
    labels = []
    for label_name, label_value in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        class_dir = base_dir / label_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*.jpeg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, image_size)
            # Duplica o canal de cinza para três canais, compatível com camadas Conv2D padrão
            img = np.stack([img, img, img], axis=-1)
            images.append(img)
            labels.append(label_value)
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)
    return X, y


def load_pneumonia_dataset(image_size=(224, 224)):
    # Carrega imagens de treino, teste e validação do dataset de pneumonia
    train_dir = PNEUMONIA_ROOT / "train"
    test_dir = PNEUMONIA_ROOT / "test"
    val_dir = PNEUMONIA_ROOT / "val"
    X_train, y_train = load_images_from_directory(train_dir, image_size=image_size)
    X_test, y_test = load_images_from_directory(test_dir, image_size=image_size)
    X_val, y_val = load_images_from_directory(val_dir, image_size=image_size)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_cnn_model(input_shape, num_classes: int = 2):
    # Define uma CNN simples para classificação binária ou multi-classe de imagens médicas
    from tensorflow.keras import layers, models

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    return model


def train_pneumonia_cnn(output_dir: Path, image_size=(224, 224), epochs: int = 3, batch_size: int = 32) -> None:
    # Treina a CNN para detecção de pneumonia em radiografias de tórax
    ensure_dir(output_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_pneumonia_dataset(image_size=image_size)
    if X_train.size == 0 or X_test.size == 0:
        return
    model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=2)
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    checkpoint_path = output_dir / "melhor_modelo_pneumonia.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    metrics_text = []
    metrics_text.append(f"Test loss: {test_loss:.4f}")
    metrics_text.append(f"Test accuracy: {test_acc:.4f}")
    (output_dir / "metricas_pneumonia.txt").write_text("\n".join(metrics_text), encoding="utf-8")


def load_mammography_metadata():
    # Carrega metadados de mamografia e determina rótulos binários (maligno x benigno)
    if not MAMMO_MASS_TRAIN.exists():
        return None, None
    meta_train = pd.read_csv(MAMMO_MASS_TRAIN)
    if "patient_id" not in meta_train.columns and "PatientID" in meta_train.columns:
        meta_train = meta_train.rename(columns={"PatientID": "patient_id"})
    dicom_info = pd.read_csv(MAMMO_DICOM_INFO)
    if "image_path" in dicom_info.columns:
        prefix = "CBIS-DDSM/jpeg/"
        dicom_info["image_path"] = dicom_info["image_path"].astype(str).apply(
            lambda p: p.split(prefix, 1)[-1] if prefix in p else p
        )
    if "PatientID" in dicom_info.columns:
        dicom_info["patient_id"] = dicom_info["PatientID"].astype(str).str.extract(
            r"(P_\d+)", expand=False
        )
        dicom_info = dicom_info.dropna(subset=["patient_id"])
    dicom_info = dicom_info[["patient_id", "image_path"]]
    merged = pd.merge(meta_train, dicom_info, on="patient_id", how="inner")
    label_col = None
    for col in ["pathology", "classification", "abnormality_type"]:
        if col in merged.columns:
            label_col = col
            break
    if label_col is None:
        return None, None
    merged = merged.dropna(subset=[label_col])
    merged["label"] = merged[label_col].str.contains("MALIGNANT", case=False).astype(int)
    image_paths = merged["image_path"].tolist()
    labels = merged["label"].values
    return image_paths, labels


def load_mammography_images(dataset_root: Path, image_paths, labels, image_size=(224, 224)):
    # Carrega imagens de mamografia em escala de cinza e gera tensores normalizados
    images = []
    y = []
    for img_rel_path, label in zip(image_paths, labels):
        img_path = dataset_root / img_rel_path
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, image_size)
        img = np.stack([img, img, img], axis=-1)
        images.append(img)
        y.append(label)
    if not images:
        return None, None
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)
    return X, y


def train_mammography_cnn(output_dir: Path, dataset_root: Path, image_size=(224, 224), epochs: int = 3) -> None:
    # Treina a CNN para detecção de câncer de mama em mamografias, se os dados estiverem disponíveis
    ensure_dir(output_dir)
    image_paths, labels = load_mammography_metadata()
    if image_paths is None:
        info_path = output_dir / "mamografia_nao_treinada.txt"
        info_path.write_text(
            "Metadados ou rótulos de mamografia indisponíveis. Verifique se os arquivos de imagem estão presentes.",
            encoding="utf-8",
        )
        return
    X, y = load_mammography_images(dataset_root, image_paths, labels, image_size=image_size)
    if X is None:
        info_path = output_dir / "mamografia_nao_treinada.txt"
        info_path.write_text(
            "Imagens de mamografia não encontradas no caminho esperado. Adicione as imagens para treinar o modelo.",
            encoding="utf-8",
        )
        return
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=2)
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=16,
        verbose=1,
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    metrics_text = []
    metrics_text.append(f"Test loss: {test_loss:.4f}")
    metrics_text.append(f"Test accuracy: {test_acc:.4f}")
    (output_dir / "metricas_mamografia.txt").write_text("\n".join(metrics_text), encoding="utf-8")


def run_all_vision_experiments() -> None:
    # Executa todos os experimentos de visão computacional (pneumonia e câncer de mama)
    base_output = RESULTS_DIR / "visao_computacional"
    pneumonia_output = base_output / "pneumonia"
    train_pneumonia_cnn(pneumonia_output)
    mammography_output = base_output / "cancer_mama"
    dataset_root = MAMMO_JPEG_ROOT
    train_mammography_cnn(mammography_output, dataset_root=dataset_root)

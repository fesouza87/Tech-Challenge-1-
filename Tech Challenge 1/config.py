from pathlib import Path

# Diretório onde está o código do Tech Challenge 1
ROOT_DIR = Path(__file__).resolve().parent
# Diretório raiz do repositório, usado como base para localizar datasets
BASE_DIR = ROOT_DIR.parent

# Caminhos para os datasets tabulares utilizados nos experimentos
CANCER_MAMA_CSV = BASE_DIR / "Diagnostico Cancer Mama Dataset" / "diagnosticoCancerMama.csv"
DIABETES_CSV = BASE_DIR / "Diagnostico Diabetes Dataset" / "diagnosticoDiabetes.csv"
SOCIAL_MEDIA_CSV = BASE_DIR / "Social Media" / "social_media_viral_content_dataset.csv"

# Diretório raiz das imagens de raio-x para detecção de pneumonia
PNEUMONIA_ROOT = BASE_DIR / "RaioX Pneumonia" / "chest_xray"

MAMMO_CSV_DIR = BASE_DIR / "Imagens Cancer Mama" / "csv"
MAMMO_DICOM_INFO = MAMMO_CSV_DIR / "dicom_info.csv"
MAMMO_MASS_TRAIN = MAMMO_CSV_DIR / "mass_case_description_train_set.csv"
MAMMO_MASS_TEST = MAMMO_CSV_DIR / "mass_case_description_test_set.csv"
MAMMO_JPEG_ROOT = BASE_DIR / "Imagens Cancer Mama" / "jpeg"

RESULTS_DIR = ROOT_DIR / "results"

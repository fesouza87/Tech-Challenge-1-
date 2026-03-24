"""
Configura caminhos padrão do Tech Challenge 2 - Rotas.
- ROOT_DIR: diretório do módulo TC2
- BASE_DIR: raiz do repositório
- RESULTS_DIR: saída padrão dos artefatos (instância, solução, relatório e plot)
- DEFAULT_*: caminhos de arquivos padrão utilizados pelo CLI
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT_DIR.parent

RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_INSTANCE_PATH = RESULTS_DIR / "instance.json"
DEFAULT_SOLUTION_PATH = RESULTS_DIR / "solution.json"
DEFAULT_PLOT_PATH = RESULTS_DIR / "routes.png"

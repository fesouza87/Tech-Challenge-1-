from argparse import ArgumentParser
from pathlib import Path
import sys


# Garante que o diretório do projeto esteja no sys.path para permitir imports internos
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RESULTS_DIR


def main():
    # Cria a interface de linha de comando para escolher qual tarefa executar
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["all", "tabular", "vision", "diabetes"],
        default="all",
        help="Qual parte do desafio executar",
    )
    args = parser.parse_args()
    # Executa a rotina correspondente ao valor informado em --task
    if args.task == "all":
        # Executa todos os experimentos tabulares e de visão computacional
        from tabular_pipeline import run_all_tabular_experiments
        from vision_pipeline import run_all_vision_experiments
        run_all_tabular_experiments()
        run_all_vision_experiments()
    elif args.task == "tabular":
        # Executa apenas os experimentos com dados tabulares
        from tabular_pipeline import run_all_tabular_experiments
        run_all_tabular_experiments()
    elif args.task == "vision":
        # Executa apenas os experimentos de visão computacional
        from vision_pipeline import run_all_vision_experiments
        run_all_vision_experiments()
    elif args.task == "diabetes":
        # Executa somente o experimento tabular de diabetes, salvando em uma pasta específica
        output_dir = RESULTS_DIR / "tabular" / "diabetes"
        from tabular_pipeline import run_diabetes_tabular
        run_diabetes_tabular(output_dir)


if __name__ == "__main__":
    main()

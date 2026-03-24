from argparse import ArgumentParser
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

"""
Tech Challenge 2 - Rotas
Script CLI responsável por:
- Gerar uma instância sintética de entregas/veículos.
- Otimizar rotas com Algoritmo Genético (GA) integrando código base de TSP.
- Visualizar rotas em PNG (opcional, requer matplotlib).
- Gerar relatório com comparativo de desempenho (GA vs heurísticas) e instruções operacionais.
- Responder perguntas sobre o plano (opcionalmente com LLM, via Ollama).

Uso:
python "Tech Challenge 2 - Rotas/main.py" --task <all|generate|optimize|visualize|report|ask> [opções]
"""
from config import DEFAULT_INSTANCE_PATH, DEFAULT_PLOT_PATH, DEFAULT_SOLUTION_PATH
from llm import (
    answer_question_classic,
    answer_question_llm,
    format_driver_instructions,
    format_summary_report,
    generate_llm_report,
    llm_enabled,
)
from routes import (
    best_of_random_tours,
    compute_plan_base_distance,
    evaluate,
    generate_synthetic_instance,
    load_instance,
    load_plan,
    nearest_neighbor_tour,
    priority_first_tour,
    save_instance,
    save_plan,
    solve_ga,
)


def main() -> None:
    """
    Ponto de entrada do CLI.
    - Define e interpreta argumentos de linha de comando.
    - Encadeia tarefas conforme o valor de --task.
    - Persiste resultados em 'Tech Challenge 2 - Rotas/results/'.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["all", "generate", "optimize", "visualize", "report", "ask"],
        default="all",
    )
    parser.add_argument("--instance", default=str(DEFAULT_INSTANCE_PATH))
    parser.add_argument("--solution", default=str(DEFAULT_SOLUTION_PATH))
    parser.add_argument("--plot", default=str(DEFAULT_PLOT_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", type=str, default="")

    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--mutation", type=float, default=0.3)
    parser.add_argument("--benchmark-random", type=int, default=200)
    parser.add_argument("--use-llm", action="store_true")

    args = parser.parse_args()

    instance_path = Path(args.instance)
    solution_path = Path(args.solution)
    plot_path = Path(args.plot)

    if args.task in {"all", "generate"}:
        instance = generate_synthetic_instance(seed=args.seed)
        save_instance(instance, instance_path)
    else:
        instance = load_instance(instance_path)

    if args.task in {"all", "optimize"}:
        plan = solve_ga(
            instance,
            population_size=args.population,
            generations=args.generations,
            mutation_probability=args.mutation,
            seed=args.seed,
        )
        save_plan(plan, solution_path)
    else:
        plan = load_plan(solution_path) if solution_path.exists() else None

    if args.task in {"all", "visualize"}:
        if plan is None:
            plan = load_plan(solution_path)
        try:
            from visualization import plot_routes
        except ImportError:
            sys.stderr.write("Visualizacao ignorada: dependencias ausentes (instale requirements.txt para gerar o plot).\n")
        else:
            plot_routes(instance, plan, plot_path=plot_path)

    if args.task in {"all", "report"}:
        if plan is None:
            plan = load_plan(solution_path)
        base_repo_dir = ROOT_DIR.parent / "genetic_algorithm_tsp-main" / "genetic_algorithm_tsp-main"
        base_license_path = base_repo_dir / "LICENSE"
        attribution_lines: list[str] = []
        attribution_lines.append("Base Utilizada (TSP/GA)")
        attribution_lines.append("")
        attribution_lines.append(f"- Repositorio: {base_repo_dir}")
        attribution_lines.append(f"- Licenca: CC0 1.0 Universal (ver {base_license_path})")
        attribution_lines.append("- Componentes reutilizados: generate_random_population, sort_population, order_crossover, mutate")
        attribution_lines.append("- Integracao no projeto: Tech Challenge 2 - Rotas/routes.py (solve_ga + fitness com restricoes)")
        attribution = "\n".join(attribution_lines).strip() + "\n\n"

        delivery_ids = list(instance.deliveries.keys())
        nn_tour = nearest_neighbor_tour(instance, delivery_ids)
        nn_plan = evaluate(instance, nn_tour)

        prio_tour = priority_first_tour(instance)
        prio_plan = evaluate(instance, prio_tour)

        rand_tour = best_of_random_tours(instance, n=args.benchmark_random, seed=args.seed)
        rand_plan = evaluate(instance, rand_tour)

        def fmt_row(name: str, p) -> str:
            penalty = sum(p.penalties.values()) if p.penalties else 0.0
            base = compute_plan_base_distance(instance, p)
            return f"- {name}: objetivo={p.total_distance:.2f} | base={base:.2f} | penalidades={penalty:.2f} | viavel={p.feasible}"

        comparison_lines: list[str] = []
        comparison_lines.append("Comparativo de Desempenho")
        comparison_lines.append("")
        comparison_lines.append(fmt_row("GA (solucao)", plan))
        comparison_lines.append(fmt_row("Heuristica - Nearest Neighbor", nn_plan))
        comparison_lines.append(fmt_row("Heuristica - Prioridade Primeiro", prio_plan))
        comparison_lines.append(fmt_row(f"Heuristica - Melhor de {args.benchmark_random} Aleatorias", rand_plan))
        comparison = "\n".join(comparison_lines).strip() + "\n\n"

        report_classic = attribution + comparison + format_summary_report(instance, plan) + "\n" + format_driver_instructions(instance, plan)
        report_dir = solution_path.parent
        (report_dir / "report.txt").write_text(report_classic, encoding="utf-8")

        if args.use_llm or llm_enabled():
            report_llm_path = report_dir / "report_llm.txt"
            report_llm_error_path = report_dir / "report_llm_error.txt"
            try:
                report_llm = generate_llm_report(attribution, comparison, instance, plan)
            except Exception as e:
                report_llm = ""
                report_llm_error_path.write_text(str(e), encoding="utf-8")
            if report_llm.strip():
                report_llm_path.write_text(report_llm, encoding="utf-8")
                report_llm_error_path.unlink(missing_ok=True)
            elif not report_llm_error_path.exists():
                report_llm_error_path.write_text("Resposta vazia do LLM (verifique o modelo configurado em LLM_MODEL).", encoding="utf-8")

    if args.task == "ask":
        if plan is None:
            plan = load_plan(solution_path)
        answer_dir = solution_path.parent
        response_classic = answer_question_classic(instance, plan, args.question)
        (answer_dir / "answer.txt").write_text(response_classic, encoding="utf-8")

        if args.use_llm or llm_enabled():
            answer_llm_path = answer_dir / "answer_llm.txt"
            answer_llm_error_path = answer_dir / "answer_llm_error.txt"
            try:
                response_llm = answer_question_llm(instance, plan, args.question)
            except Exception as e:
                response_llm = ""
                answer_llm_error_path.write_text(str(e), encoding="utf-8")
            if response_llm.strip():
                answer_llm_path.write_text(response_llm, encoding="utf-8")
                answer_llm_error_path.unlink(missing_ok=True)
            elif not answer_llm_error_path.exists():
                answer_llm_error_path.write_text("Resposta vazia do LLM (verifique o modelo configurado em LLM_MODEL).", encoding="utf-8")


if __name__ == "__main__":
    main()

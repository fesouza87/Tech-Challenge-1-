"""
Geração de visualização das rotas otimizadas (PNG) para o Tech Challenge 2.
- Plota o depósito (quadrado preto) e as rotas por veículo com cores distintas.
- Ajuda na validação rápida da solução: distribuição espacial e ordem de atendimento.
"""
from pathlib import Path

from matplotlib import pyplot as plt

from routes import Instance, RoutePlan


def plot_routes(instance: Instance, plan: RoutePlan, plot_path: Path) -> None:
    """
    Renderiza um gráfico das rotas por veículo e salva em plot_path.

    Parâmetros:
    - instance: cenário com depósito/locais/veículos
    - plan: rotas por veículo e dados de custo
    - plot_path: caminho do arquivo PNG de saída
    """
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    depot = instance.locations[instance.depot_id]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]

    plt.figure(figsize=(10, 8))
    plt.scatter([depot.x], [depot.y], c="black", s=120, marker="s")
    plt.text(depot.x, depot.y, depot.name, fontsize=9)

    for idx, vehicle in enumerate(instance.vehicles):
        route = plan.routes_by_vehicle.get(vehicle.id, [])
        if not route:
            continue
        xs = [depot.x]
        ys = [depot.y]
        for delivery_id in route:
            delivery = instance.deliveries[delivery_id]
            loc = instance.locations[delivery.location_id]
            xs.append(loc.x)
            ys.append(loc.y)
        xs.append(depot.x)
        ys.append(depot.y)

        color = colors[idx % len(colors)]
        plt.plot(xs, ys, c=color, linewidth=2, label=f"Veiculo {vehicle.id}")
        plt.scatter(xs[1:-1], ys[1:-1], c=color, s=40)

    plt.title("Rotas Otimizadas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

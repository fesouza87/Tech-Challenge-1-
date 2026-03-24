from __future__ import annotations
"""
Tech Challenge 2 - Rotas
Módulo principal de modelagem e otimização de rotas (VRP-like) com GA.

Descrição geral:
- Modela entidades: Location (ponto), Delivery (entrega), Vehicle (veículo) e Instance (cenário).
- Avalia soluções (fitness) combinando distância percorrida com penalidades de restrição:
  capacidade de carga, autonomia máxima, prioridade de entregas e itens não alocados.
- Integra funções de Algoritmo Genético (GA) provenientes de um código base de TSP (CC0),
  utilizando geração de população, crossover e mutação para evoluir um “tour gigante”,
  posteriormente dividido em rotas por veículo (split_giant_tour).

Uso típico:
- Gerar uma instância sintética (generate_synthetic_instance)
- Calcular uma solução via GA (solve_ga)
- Avaliar heurísticas baseline para comparação (nearest_neighbor_tour, priority_first_tour, best_of_random_tours)
"""

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    x: float
    y: float
    """Ponto no espaço (x, y) com identificação e nome amigável."""


@dataclass(frozen=True)
class Delivery:
    id: str
    location_id: str
    demand: float
    priority: str
    """Entrega associada a um local, com demanda e prioridade ('critical' ou 'regular')."""


@dataclass(frozen=True)
class Vehicle:
    id: str
    capacity: float
    max_distance: float
    """Veículo com limites de capacidade e autonomia (distância máxima)."""


@dataclass(frozen=True)
class Instance:
    depot_id: str
    locations: dict[str, Location]
    deliveries: dict[str, Delivery]
    vehicles: list[Vehicle]
    """Cenário completo: depósito, locais, entregas e veículos."""


@dataclass(frozen=True)
class RoutePlan:
    routes_by_vehicle: dict[str, list[str]]
    total_distance: float
    feasible: bool
    penalties: dict[str, float]
    """Plano de rotas: rotas por veículo, custo total (distância + penalidades) e viabilidade."""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def euclidean(a: Location, b: Location) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def load_base_tsp_ga():
    """Carrega o módulo GA de TSP (CC0) adicionando o diretório ao sys.path."""
    repo_root = Path(__file__).resolve().parents[1]
    tsp_dir = repo_root / "genetic_algorithm_tsp-main" / "genetic_algorithm_tsp-main"
    if not tsp_dir.exists():
        raise FileNotFoundError(f"Código base de TSP não encontrado em: {tsp_dir}")
    tsp_dir_str = str(tsp_dir)
    if tsp_dir_str not in sys.path:
        sys.path.insert(0, tsp_dir_str)
    import genetic_algorithm as tsp_ga

    return tsp_ga


def compute_route_distance(instance: Instance, delivery_ids: list[str]) -> float:
    """Calcula a distância total: depósito -> entregas (na ordem) -> depósito."""
    if not delivery_ids:
        return 0.0
    depot = instance.locations[instance.depot_id]
    cur = depot
    dist = 0.0
    for delivery_id in delivery_ids:
        loc = instance.locations[instance.deliveries[delivery_id].location_id]
        dist += euclidean(cur, loc)
        cur = loc
    dist += euclidean(cur, depot)
    return float(dist)


def compute_route_load(instance: Instance, delivery_ids: list[str]) -> float:
    return float(sum(instance.deliveries[d].demand for d in delivery_ids))


def compute_plan_base_distance(instance: Instance, plan: RoutePlan) -> float:
    """Soma das distâncias (sem penalidades) das rotas por veículo dentro do plano."""
    total = 0.0
    for vehicle in instance.vehicles:
        route = plan.routes_by_vehicle.get(vehicle.id, [])
        if route:
            total += compute_route_distance(instance, route)
    return float(total)


def split_giant_tour(instance: Instance, tour: list[str]) -> tuple[dict[str, list[str]], dict[str, float]]:
    """
    Divide um tour único em rotas por veículo, respeitando capacidade e autonomia.
    Sobras de entregas geram penalidade 'unassigned'.
    """
    remaining = list(tour)
    routes_by_vehicle: dict[str, list[str]] = {}
    penalties: dict[str, float] = {"unassigned": 0.0}

    for vehicle in instance.vehicles:
        if not remaining:
            routes_by_vehicle[vehicle.id] = []
            continue
        route: list[str] = []
        load = 0.0
        while remaining:
            next_id = remaining[0]
            next_demand = instance.deliveries[next_id].demand
            if load + next_demand > vehicle.capacity:
                break
            candidate = route + [next_id]
            candidate_dist = compute_route_distance(instance, candidate)
            if candidate_dist > vehicle.max_distance:
                break
            route = candidate
            load += next_demand
            remaining.pop(0)
        routes_by_vehicle[vehicle.id] = route

    if remaining:
        penalties["unassigned"] = 10000.0 + 1000.0 * float(len(remaining))
        for vehicle in instance.vehicles:
            routes_by_vehicle.setdefault(vehicle.id, [])

    return routes_by_vehicle, penalties


def priority_penalty(instance: Instance, routes_by_vehicle: dict[str, list[str]]) -> float:
    """
    Penaliza entregas críticas tardias/ausentes.
    - Ausente: penalidade forte
    - Tardia: penalidade proporcional após 1/3 do sequenciamento total
    """
    critical = [d.id for d in instance.deliveries.values() if d.priority.lower() == "critical"]
    if not critical:
        return 0.0
    visited_order: list[str] = []
    for vehicle in instance.vehicles:
        visited_order.extend(routes_by_vehicle.get(vehicle.id, []))
    if not visited_order:
        return 5000.0
    position = {delivery_id: idx for idx, delivery_id in enumerate(visited_order)}
    penalty = 0.0
    threshold = max(1, len(visited_order) // 3)
    for delivery_id in critical:
        idx = position.get(delivery_id)
        if idx is None:
            penalty += 5000.0
        elif idx >= threshold:
            penalty += 25.0 * float(idx - threshold + 1)
    return penalty


def evaluate(instance: Instance, tour: list[str]) -> RoutePlan:
    """
    Avalia um tour gerando rotas por veículo, calculando distância e penalidades:
    capacidade, autonomia, prioridade e não alocados.
    """
    routes_by_vehicle, penalties = split_giant_tour(instance, tour)
    total_distance = 0.0
    feasible = True
    capacity_penalty = 0.0
    autonomy_penalty = 0.0

    for vehicle in instance.vehicles:
        route = routes_by_vehicle.get(vehicle.id, [])
        load = compute_route_load(instance, route)
        dist = compute_route_distance(instance, route) if route else 0.0
        total_distance += dist
        if load > vehicle.capacity:
            feasible = False
            capacity_penalty += 1000.0 * float(load - vehicle.capacity)
        if dist > vehicle.max_distance:
            feasible = False
            autonomy_penalty += 1000.0 * float(dist - vehicle.max_distance)

    penalties["capacity"] = capacity_penalty
    penalties["autonomy"] = autonomy_penalty
    penalties["priority"] = priority_penalty(instance, routes_by_vehicle)

    feasible = feasible and penalties.get("unassigned", 0.0) == 0.0
    total_penalty = float(sum(penalties.values()))
    return RoutePlan(
        routes_by_vehicle=routes_by_vehicle,
        total_distance=total_distance + total_penalty,
        feasible=feasible,
        penalties=penalties,
    )


 


def nearest_neighbor_tour(instance: Instance, delivery_ids: list[str]) -> list[str]:
    """Heurística NN: sempre escolhe a próxima entrega mais próxima do ponto atual."""
    depot = instance.locations[instance.depot_id]
    remaining = set(delivery_ids)
    tour: list[str] = []
    cur_loc = depot
    while remaining:
        next_id = min(
            remaining,
            key=lambda d: euclidean(cur_loc, instance.locations[instance.deliveries[d].location_id]),
        )
        tour.append(next_id)
        remaining.remove(next_id)
        cur_loc = instance.locations[instance.deliveries[next_id].location_id]
    return tour


def priority_first_tour(instance: Instance) -> list[str]:
    """Heurística: atende críticos primeiro via NN, depois regulares via NN."""
    critical = [d.id for d in instance.deliveries.values() if d.priority.lower() == "critical"]
    regular = [d.id for d in instance.deliveries.values() if d.priority.lower() != "critical"]
    return nearest_neighbor_tour(instance, critical) + nearest_neighbor_tour(instance, regular)


def best_of_random_tours(instance: Instance, n: int, seed: int | None = 42) -> list[str]:
    """Baseline: sorteia N tours aleatórios e devolve o melhor segundo evaluate(...)."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    delivery_ids = list(instance.deliveries.keys())
    if not delivery_ids:
        return []
    best_tour = list(delivery_ids)
    rng.shuffle(best_tour)
    best_score = evaluate(instance, best_tour).total_distance
    for _ in range(max(1, n) - 1):
        tour = list(delivery_ids)
        rng.shuffle(tour)
        score = evaluate(instance, tour).total_distance
        if score < best_score:
            best_score = score
            best_tour = tour
    return best_tour


def solve_ga(
    instance: Instance,
    population_size: int = 80,
    generations: int = 250,
    crossover_rate: float = 0.9,
    mutation_probability: float = 0.3,
    seed: int | None = 42,
) -> RoutePlan:
    """
    Otimiza via GA reutilizando funções do código TSP:
    - generate_random_population, sort_population, order_crossover, mutate
    Mantém elitismo e seleção por pesos inversos ao custo.
    """
    if seed is not None:
        random.seed(seed)

    tsp_ga = load_base_tsp_ga()

    delivery_ids = list(instance.deliveries.keys())
    if not delivery_ids:
        return RoutePlan(routes_by_vehicle={v.id: [] for v in instance.vehicles}, total_distance=0.0, feasible=True, penalties={})

    population: list[list[str]] = tsp_ga.generate_random_population(delivery_ids, population_size)
    best_idx = min(range(len(population)), key=lambda i: evaluate(instance, population[i]).total_distance)
    best_plan = evaluate(instance, population[best_idx])

    for _ in range(generations):
        plans = [evaluate(instance, tour) for tour in population]
        scores = [p.total_distance for p in plans]
        population, scores = tsp_ga.sort_population(population, scores)

        gen_best = evaluate(instance, population[0])
        if gen_best.total_distance < best_plan.total_distance:
            best_plan = gen_best

        new_population: list[list[str]] = [list(population[0])]
        weights: list[float] = []
        for s in scores:
            if s <= 0:
                weights.append(1.0)
            else:
                weights.append(1.0 / float(s))

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, weights=weights, k=2)
            if random.random() < crossover_rate:
                child = tsp_ga.order_crossover(parent1, parent2)
            else:
                child = list(parent1)
            child = tsp_ga.mutate(child, mutation_probability)
            new_population.append(child)

        population = new_population

    return best_plan


def generate_synthetic_instance(
    num_locations: int = 24,
    num_deliveries: int = 20,
    num_vehicles: int = 3,
    seed: int = 42,
) -> Instance:
    """Gera cenário sintético reprodutível para demonstração/teste."""
    rng = random.Random(seed)
    depot = Location(id="DEPOT", name="Centro de Distribuicao", x=50.0, y=50.0)
    locations: dict[str, Location] = {depot.id: depot}
    for i in range(num_locations):
        loc_id = f"L{i:03d}"
        locations[loc_id] = Location(
            id=loc_id,
            name=f"Unidade {i + 1}",
            x=rng.uniform(0, 100),
            y=rng.uniform(0, 100),
        )

    deliveries: dict[str, Delivery] = {}
    loc_ids = [lid for lid in locations.keys() if lid != depot.id]
    rng.shuffle(loc_ids)
    for i in range(num_deliveries):
        d_id = f"D{i:03d}"
        loc_id = loc_ids[i % len(loc_ids)]
        priority = "critical" if rng.random() < 0.25 else "regular"
        demand = rng.uniform(1.0, 8.0) if priority == "regular" else rng.uniform(2.0, 10.0)
        deliveries[d_id] = Delivery(id=d_id, location_id=loc_id, demand=float(demand), priority=priority)

    vehicles: list[Vehicle] = []
    for i in range(num_vehicles):
        vehicles.append(
            Vehicle(
                id=f"V{i + 1}",
                capacity=45.0,
                max_distance=260.0,
            )
        )

    return Instance(depot_id=depot.id, locations=locations, deliveries=deliveries, vehicles=vehicles)


def instance_to_dict(instance: Instance) -> dict:
    """Serializa Instance para dict (compatível com JSON)."""
    return {
        "depot_id": instance.depot_id,
        "locations": {k: {"id": v.id, "name": v.name, "x": v.x, "y": v.y} for k, v in instance.locations.items()},
        "deliveries": {
            k: {"id": v.id, "location_id": v.location_id, "demand": v.demand, "priority": v.priority}
            for k, v in instance.deliveries.items()
        },
        "vehicles": [{"id": v.id, "capacity": v.capacity, "max_distance": v.max_distance} for v in instance.vehicles],
    }


def instance_from_dict(data: dict) -> Instance:
    """Desserializa dict em Instance."""
    locations = {k: Location(**v) for k, v in data["locations"].items()}
    deliveries = {k: Delivery(**v) for k, v in data["deliveries"].items()}
    vehicles = [Vehicle(**v) for v in data["vehicles"]]
    return Instance(depot_id=data["depot_id"], locations=locations, deliveries=deliveries, vehicles=vehicles)


def save_instance(instance: Instance, path: Path) -> None:
    """Salva Instance em JSON."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(instance_to_dict(instance), ensure_ascii=False, indent=2), encoding="utf-8")


def load_instance(path: Path) -> Instance:
    """Carrega Instance de JSON."""
    return instance_from_dict(json.loads(path.read_text(encoding="utf-8")))


def plan_to_dict(plan: RoutePlan) -> dict:
    """Serializa RoutePlan para dict (compatível com JSON)."""
    return {
        "routes_by_vehicle": plan.routes_by_vehicle,
        "total_distance": plan.total_distance,
        "feasible": plan.feasible,
        "penalties": plan.penalties,
    }


def plan_from_dict(data: dict) -> RoutePlan:
    """Desserializa dict em RoutePlan."""
    return RoutePlan(
        routes_by_vehicle={k: list(v) for k, v in data["routes_by_vehicle"].items()},
        total_distance=float(data["total_distance"]),
        feasible=bool(data["feasible"]),
        penalties={k: float(v) for k, v in data.get("penalties", {}).items()},
    )


def save_plan(plan: RoutePlan, path: Path) -> None:
    """Salva RoutePlan em JSON."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(plan_to_dict(plan), ensure_ascii=False, indent=2), encoding="utf-8")


def load_plan(path: Path) -> RoutePlan:
    """Carrega RoutePlan de JSON."""
    return plan_from_dict(json.loads(path.read_text(encoding="utf-8")))

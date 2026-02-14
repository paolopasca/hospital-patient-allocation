# Hospital Patient Allocation Optimization

A bi-objective optimization framework for assigning patients to hospital wards, comparing **exact methods (MILP)**, **metaheuristics (ILS, VNS)**, and a **matheuristic (hybrid)** approach that warm-starts the MILP solver with metaheuristic solutions.

> 📄 Academic project for *Introduction to Artificial Intelligence* — FEUP, University of Porto (Jan 2026)
> > 📖 For the full mathematical formulation, algorithm derivations, and detailed results, see the [technical report](docs/report.pdf).

---

## The Problem

Assign a set of patients to hospital wards on specific admission days, respecting bed capacity, operating theatre availability, specialization compatibility, and admission time windows — while simultaneously optimizing two competing objectives:

- **f₁ — Operational Cost:** Minimize admission delays, OR overtime, and OR undertime
- **f₂ — Workload Balance:** Minimize the maximum normalized workload across all ward-day pairs (min-max equity)

These objectives conflict: perfect workload balance may require delaying patients, while minimizing delays may overload specific wards. The problem is solved via weighted-sum scalarization: `min λ₁·f₁ + λ₂·f₂`.

---

## Solution Approaches

### 1. MILP (Exact Method)
Formulated as a Mixed Integer Linear Program solved with **CBC** (open-source, via PuLP) or **Gurobi**. Uses Branch & Bound with cutting planes. Guarantees optimality but can be slow on large instances.

### 2. Metaheuristics
A four-stage pipeline for fast, high-quality feasible solutions:

| Stage | Method | Role |
|-------|--------|------|
| 1 | **Greedy Construction** | Builds initial feasible solution by patient urgency |
| 2 | **Local Search** | Deterministic hill-climbing (day shifts + ward transfers) |
| 3 | **Iterated Local Search (ILS)** | Escapes local optima via perturbation + local refinement |
| 4 | **Variable Neighborhood Search (VNS)** | Systematic exploration across 4 neighborhood structures |

### 3. Matheuristic (Hybrid)
The key contribution — combines metaheuristics with exact optimization:

1. Run the metaheuristic pipeline → obtain a good feasible solution
2. **Warm-start** the MILP solver with this solution (set initial values for all binary variables)
3. The solver starts with an incumbent, enabling aggressive pruning from iteration one

This bridges the gap: metaheuristics provide speed, the MILP solver provides optimality guarantees. On our benchmarks, the hybrid approach achieves **median optimality gap of 0%** while remaining practical on medium-to-large instances.

---

## Key Results

Benchmarked on instances based on Smet (2023), with 20–400 patients, 4–8 wards, 7–21 day horizons:

| Instance Size | MILP Time | MH Gap | Hybrid Gap | Recommended |
|---------------|-----------|--------|------------|-------------|
| Small (20–50 patients) | 5–30s | 0.5–2% | 0% | MILP |
| Medium (100–250) | 60–300s | 3–8% | 1–3% | Hybrid |
| Large (400+) | 300s+ | 5–12% | 2–5% | Metaheuristics |

The local search phase proved critical: variants without it (NoOpt) showed substantially higher gaps and variance.

---

## Repository Structure

```
├── data_parser.py          # Parses .dat instance files
├── milp_model.py           # MILP model (Gurobi)
├── CBC_milp.py             # MILP model (CBC/PuLP, open-source)
├── metaheuristics.py       # Full metaheuristic pipeline (Greedy → LS → ILS → VNS)
├── hybrid_solver.py        # Matheuristic: metaheuristic warm-start + MILP
├── box_benchmark.py        # Benchmarking framework with box plots
├── pareto_front.py         # Multi-objective Pareto front analysis
├── data/                   # Problem instances (.dat files)
├── docs/
│   └── report.pdf          # Full report (formulation, algorithms, results)
└── README.md
```

---

## Getting Started

### Installation

```bash
pip install numpy pandas matplotlib seaborn pulp
```

For Gurobi (optional, requires [academic license](https://www.gurobi.com/academia/)):
```bash
pip install gurobipy
```

### Running the Metaheuristics

```bash
# Default mode (Greedy → Local Search → ILS → VNS)
python3 metaheuristics.py -f data/instance.dat

# Skip local search phase
python3 metaheuristics.py -f data/instance.dat -o no_opt

# Custom weights and time limits
python3 metaheuristics.py -f data/instance.dat --lambda1 0.6 --lambda2 0.4 --ils-time 10 --vns-time 10
```

### Running the MILP (CBC)

```python
from data_parser import PatientAllocationData
from CBC_milp import PatientAllocationMILP_CBC

data = PatientAllocationData("data/instance.dat")
milp = PatientAllocationMILP_CBC(data, lambda1=0.5, lambda2=0.5)
milp.build_model()
result = milp.solve(time_limit=300, verbose=True)
```

### Running the Hybrid Solver

```python
from data_parser import PatientAllocationData
from hybrid_solver import HybridSolverCBC

data = PatientAllocationData("data/instance.dat")
hybrid = HybridSolverCBC(data, lambda1=0.5, lambda2=0.5)
result = hybrid.solve(metaheuristic='ILS', mh_max_time_min=5, milp_time_limit=300, use_warm_start=True)
```

### Running Benchmarks

```bash
python3 box_benchmark.py -d data/instance1.dat data/instance2.dat --runs 10
```

---

## Technologies

`Python` · `PuLP` · `CBC` · `Gurobi` (optional) · `NumPy` · `pandas` · `Matplotlib` · `Seaborn`

## Authors

- **Paolo Pascarelli** — [GitHub](https://github.com/paolopasca)
- **João Filipe**
- **Diogo Teixeira**

## References

1. Smet, P. (2023) — *Generating balanced workload allocations in hospitals*, Operations Research for Health Care, Vol. 38
2. Lourenço, Martin & Stützle (2010) — *Iterated Local Search: Framework and Applications*
3. Hansen, Mladenović et al. (2019) — *Variable Neighborhood Search: Basics and Variants*

## License

This project was developed for educational purposes at FEUP, University of Porto.

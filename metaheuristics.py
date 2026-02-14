"""
metaheuristics.py

Fast metaheuristics for the Patient Allocation problem.

This module implements a complete heuristic pipeline:

    1. Greedy feasible construction
    2. Greedy local improvement (deterministic hill-climbing)
    3. Iterated Local Search (ILS) with fast local search and perturbations
    4. Variable Neighborhood Search (VNS) with multiple shaking operators

The goal is to produce high-quality FEASIBLE solutions on large instances
where an exact MILP approach may be too slow or fail to find feasibility.

All methods assume:
- A PatientAllocationData object with:
    * patients
    * wards
    * specialisms
    * capacity, workload, OR availability, etc.
- Compatibility constraints based on specialization
- Bed capacity constraints per ward/day
"""

import argparse
import sys
import random
import math
import time
import copy
from collections import defaultdict
from data_parser import PatientAllocationData


# =====================================================================
#                         UTILITY FUNCTIONS
# =====================================================================

def compatible_wards_for_patient(data: PatientAllocationData, patient):
    """
    Return the list of wards where the given patient can be admitted.

    A ward is compatible if:
    - the patient's specialization is equal to the ward's major specialization, OR
    - the patient's specialization is included in ward["minor_specializations"].

    This compatibility check is used everywhere: greedy, local search, ILS, VNS.
    """
    spec = patient["specialization"]
    wards = []
    for wname, ward in data.wards.items():
        if spec == ward["major_specialization"] or spec in ward["minor_specializations"]:
            wards.append(wname)
    return wards


def count_beds_used_on_day(data: PatientAllocationData, allocation, ward_name, d):
    """
    Count the number of beds used in ward 'ward_name' on day d.

    Beds are occupied by:
    - carryover patients from previous planning horizons (fixed)
    - all patients assigned to this ward whose length of stay overlaps day d
    """
    used = data.wards[ward_name]["carryover_patients"][d]
    for pid, a in allocation.items():
        if a["ward"] != ward_name:
            continue
        p = data.patients[pid]
        if a["day"] <= d < a["day"] + p["los"]:
            used += 1
    return used


def feasible_after_change_beds(data: PatientAllocationData, allocation, change=None):
    """
    Check bed feasibility of an allocation, optionally after applying a single change.

    Parameters
    ----------
    allocation : dict
        Current allocation {pid: {"ward": w, "day": d}}.
    change : tuple or None
        If not None, must be (pid, new_ward, new_day) and the allocation is
        temporarily modified before checking capacity violations.

    Returns
    -------
    bool
        True if for all wards and days, used beds <= bed capacity.
    """
    if change is not None:
        pid, new_ward, new_day = change
        allocation = allocation.copy()
        allocation[pid] = {"ward": new_ward, "day": new_day}

    for wname, ward in data.wards.items():
        cap = ward["bed_capacity"]
        for d in range(data.num_days):
            used = count_beds_used_on_day(data, allocation, wname, d)
            if used > cap:
                return False
    return True


def bed_capacity_violations(data: PatientAllocationData, allocation):
    """
    Compute the total amount of bed capacity violations (soft measure).

    Returns
    -------
    int
        Sum over all wards and days of (used beds - capacity) whenever positive.
        This is used to penalize infeasible solutions in metaheuristics.
    """
    vio = 0
    for wname, ward in data.wards.items():
        cap = ward["bed_capacity"]
        for d in range(data.num_days):
            used = count_beds_used_on_day(data, allocation, wname, d)
            if used > cap:
                vio += (used - cap)
    return vio


def compute_ot_used_by_spec_day(data: PatientAllocationData, allocation, spec, d):
    """
    Compute operating theatre time used by specialization 'spec' on day d.

    Sums the surgery_duration of all patients of specialization 'spec'
    whose surgery day equals d.
    """
    total = 0
    for pid, a in allocation.items():
        p = data.patients[pid]
        if p["specialization"] == spec and a["day"] == d:
            total += p["surgery_duration"]
    return total


def compute_workload_normalized_and_z(data: PatientAllocationData, allocation):
    """
    Compute normalized workload x[(ward, day)] and the maximum value z.

    For each ward/day:
    - total workload is computed from carryover workload + patients admitted
      whose stay covers that day
    - workload is normalized by ward capacity
    - z is the maximum normalized workload over all ward/day pairs

    Returns
    -------
    x : dict
        Mapping (ward_name, day) -> normalized workload.
    z : float
        Maximum normalized workload across all (ward, day).
    """
    x = {}
    z = 0.0
    for wname, ward in data.wards.items():
        cap = ward["workload_capacity"]
        for d in range(data.num_days):
            total_w = ward["carryover_workload"][d]
            for pid, a in allocation.items():
                if a["ward"] != wname:
                    continue
                p = data.patients[pid]
                admit = a["day"]
                if admit <= d < admit + p["los"]:
                    day_of_stay = d - admit
                    if 0 <= day_of_stay < len(p["workload_per_day"]):
                        base = p["workload_per_day"][day_of_stay]
                        spec = p["specialization"]
                        # If the patient spec is minor in this ward, apply workload factor.
                        if spec != ward["major_specialization"] and spec in ward["minor_specializations"]:
                            base *= data.specialisms[spec]["workload_factor"]
                        total_w += base
            x[(wname, d)] = total_w / cap if cap > 0 else 0.0
            z = max(z, x[(wname, d)])
    return x, z


def compute_objective_components(data: PatientAllocationData, allocation):
    """
    Compute the main objective components:
    - total delays (admission day - earliest)
    - total OR overtime and undertime
    - maximum normalized workload z
    - workload map and OR usage map (for analysis/diagnostics)

    Returns
    -------
    delays : float
    total_overtime : float
    total_undertime : float
    z : float
    x_map : dict
        Workload per (ward, day).
    ot_used_map : dict
        For each (spec, day): (used, avail, overtime, undertime).
    """
    # Admission delays
    delays = 0.0
    for pid, a in allocation.items():
        p = data.patients[pid]
        delays += max(0, a["day"] - p["earliest"])

    # OR overtime and undertime
    total_overtime = 0.0
    total_undertime = 0.0
    ot_used_map = {}
    for spec in data.specialisms.keys():
        for d in range(data.num_days):
            used = compute_ot_used_by_spec_day(data, allocation, spec, d)
            avail = data.specialisms[spec]["ot_time"][d]
            if used > avail:
                total_overtime += (used - avail)
                ot_used_map[(spec, d)] = (used, avail, used - avail, 0)
            else:
                total_undertime += (avail - used)
                ot_used_map[(spec, d)] = (used, avail, 0, avail - used)

    # Workload balancing (max normalized workload z)
    x_map, z = compute_workload_normalized_and_z(data, allocation)
    return delays, total_overtime, total_undertime, z, x_map, ot_used_map


def objective_value(data: PatientAllocationData, allocation, lambda1=0.5, lambda2=0.5):
    """
    Compute the scalarized objective: λ1 * f1 + λ2 * f2.

    f1 combines:
        - weighted OR overtime
        - weighted OR undertime
        - weighted admission delays
    f2 is:
        - maximum normalized workload (z)

    Parameters
    ----------
    lambda1, lambda2 : float
        Scalarization weights for the two objective components.
    """
    delays, ovt, undt, z, _, _ = compute_objective_components(data, allocation)
    f1 = data.weight_overtime * ovt + data.weight_undertime * undt + data.weight_delay * delays
    f2 = z
    return lambda1 * f1 + lambda2 * f2


def objective_value_penalized(
    data: PatientAllocationData,
    allocation,
    lambda1=0.5,
    lambda2=0.5,
    penalty_weight=1000.0
):
    """
    Penalized objective for metaheuristics.

    Uses the scalarized objective plus a large penalty for any bed capacity violations.
    This allows metaheuristics to explore slightly infeasible solutions but pushes them
    towards feasibility during search.

    Parameters
    ----------
    penalty_weight : float
        Penalty coefficient multiplied by total bed violations.
    """
    f_obj = objective_value(data, allocation, lambda1, lambda2)
    violations = bed_capacity_violations(data, allocation)
    return f_obj + penalty_weight * violations


# =====================================================================
#                          GREEDY CONSTRUCTION
# =====================================================================

def _score_trial(data, alloc, pid, w, d, lambda1, lambda2):
    """
    Helper for greedy phase: evaluate assigning pid to ward w at day d.

    Returns inf if the resulting allocation violates bed capacity;
    otherwise returns the objective value of the trial allocation.
    """
    trial = copy.deepcopy(alloc)
    trial[pid] = {"ward": w, "day": d}
    if not feasible_after_change_beds(data, alloc, (pid, w, d)):
        return float("inf")
    return objective_value(data, trial, lambda1, lambda2)


def _tiny_repair_shift_same_ward_day_forward(data, alloc, ward_name, congested_day, max_shifts=3):
    """
    Small, cheap repair heuristic:
    when a given ward/day is congested, try to move a few patients (up to max_shifts)
    one day forward (within their latest day) to free capacity.

    Returns
    -------
    bool
        True if at least one successful shift is applied, False otherwise.
    """
    candidates = []
    for pid, a in alloc.items():
        if a["ward"] != ward_name:
            continue
        p = data.patients[pid]
        if a["day"] <= congested_day < a["day"] + p["los"]:
            candidates.append(pid)

    random.shuffle(candidates)
    tried = 0
    for pid in candidates:
        if tried >= max_shifts:
            break
        p = data.patients[pid]
        a_day = alloc[pid]["day"]
        # Shift one day forward if still within latest and horizon
        if a_day + 1 <= min(p["latest"], data.num_days - 1):
            trial = copy.deepcopy(alloc)
            trial[pid]["day"] = a_day + 1
            if feasible_after_change_beds(data, trial):
                alloc[pid]["day"] = a_day + 1
                return True
        tried += 1
    return False


def greedy_feasible_by_window_strict(data: PatientAllocationData):
    """
    Construct an initial feasible allocation using strict greedy rules.

    Patients are ordered by:
    - smallest (latest - earliest) window first (tighter windows are more critical)
    - then by earliest admission day
    - then by decreasing length of stay

    For each patient:
    - try to place them in a compatible ward and day without violating capacity
    - attempt small repairs if necessary
    - as last resort, pick the position minimizing the objective among feasible moves

    If no placement is possible, raises a RuntimeError (instance is too saturated).
    """
    patients_order = sorted(
        data.patients.items(),
        key=lambda kv: (
            (kv[1]["latest"] - kv[1]["earliest"]),
            kv[1]["earliest"],
            -kv[1]["los"]
        )
    )

    alloc = {}

    for pid, p in patients_order:
        wards = compatible_wards_for_patient(data, p)
        los = p["los"]
        placed = False

        # 1) Try a direct feasible assignment without repairs
        for w in wards:
            ward = data.wards[w]
            cap = ward["bed_capacity"]

            for d in range(p["earliest"], min(p["latest"] + 1, data.num_days)):
                ok = True
                for day_check in range(d, min(d + los, data.num_days)):
                    used = count_beds_used_on_day(data, alloc, w, day_check)
                    if used >= cap:
                        ok = False
                        break
                if ok:
                    alloc[pid] = {"ward": w, "day": d}
                    placed = True
                    break
            if placed:
                break

        # 2) If not placed, try a tiny repair shifting other patients one day forward
        if not placed:
            for w in wards:
                for d in range(p["earliest"], min(p["latest"] + 1, data.num_days)):
                    for day_check in range(d, min(d + los, data.num_days)):
                        used = count_beds_used_on_day(data, alloc, w, day_check)
                        cap = data.wards[w]["bed_capacity"]
                        if used >= cap:
                            if _tiny_repair_shift_same_ward_day_forward(
                                data, alloc, w, day_check, max_shifts=3
                            ):
                                break

            # 3) After repair attempts, try again to place greedily
            for w in wards:
                cap = data.wards[w]["bed_capacity"]
                for d in range(p["earliest"], min(p["latest"] + 1, data.num_days)):
                    ok = True
                    for day_check in range(d, min(d + los, data.num_days)):
                        used = count_beds_used_on_day(data, alloc, w, day_check)
                        if used >= cap:
                            ok = False
                            break
                    if ok:
                        alloc[pid] = {"ward": w, "day": d}
                        placed = True
                        break
                if placed:
                    break

        # 4) As a fallback, choose the least damaging feasible position
        if not placed:
            lambda1, lambda2 = 0.5, 0.5
            best_score = float('inf')
            best_w, best_d = None, None

            for w in wards:
                for d in range(p["earliest"], min(p["latest"] + 1, data.num_days)):
                    score = _score_trial(data, alloc, pid, w, d, lambda1, lambda2)
                    if score < best_score:
                        best_score = score
                        best_w, best_d = w, d

            if best_w is not None:
                alloc[pid] = {"ward": best_w, "day": best_d}
                placed = True

        if not placed:
            raise RuntimeError(f"❌ Nessun posto disponibile per {pid} — istanza troppo satura.")

    return alloc


# =====================================================================
#                     GREEDY + LOCAL IMPROVEMENT
# =====================================================================

def greedy_local_improvement(
    data: PatientAllocationData,
    start_allocation,
    lambda1=0.5,
    lambda2=0.5,
    max_rounds=6
):
    """
    Apply a deterministic local search to improve the greedy solution.

    Strategy:
    - Identify patients contributing most to delay
    - For each such patient:
        * try moving them to different days (same ward)
        * try moving them to different compatible wards (same day)
    - Accept only moves that strictly improve the objective value

    Parameters
    ----------
    start_allocation : dict
        Initial feasible allocation, typically from greedy_feasible_by_window_strict.
    """
    current = copy.deepcopy(start_allocation)
    curr_val = objective_value(data, current, lambda1, lambda2)

    for _ in range(max_rounds):
        improved = False
        no_improve = True

        # Compute contribution to delay cost for each patient
        delays, ovt, undt, z, x_map, ot_map = compute_objective_components(data, current)
        patient_contributions = []
        for pid in data.patients.keys():
            p = data.patients[pid]
            a = current[pid]
            contrib = max(0, a["day"] - p["earliest"]) * data.weight_delay
            patient_contributions.append((contrib, pid))
        patient_contributions.sort(reverse=True)
        ordered_pids = [pid for _, pid in patient_contributions]

        # Try improving patients in order of their delay contribution
        for pid in ordered_pids:
            p = data.patients[pid]
            base = current[pid]
            best_move = base
            best_val = curr_val

            # 1) Try different days in the same ward
            for d in range(p["earliest"], min(p["latest"] + 1, data.num_days)):
                if d == base["day"]:
                    continue
                if not feasible_after_change_beds(data, current, (pid, base["ward"], d)):
                    continue
                trial = copy.deepcopy(current)
                trial[pid] = {"ward": base["ward"], "day": d}
                val = objective_value(data, trial, lambda1, lambda2)
                if val + 1e-9 < best_val:
                    best_val = val
                    best_move = {"ward": base["ward"], "day": d}

            # 2) Try different wards on the same day
            wards = compatible_wards_for_patient(data, p)
            for w in wards:
                if w == base["ward"]:
                    continue
                if not feasible_after_change_beds(data, current, (pid, w, base["day"])):
                    continue
                trial = copy.deepcopy(current)
                trial[pid] = {"ward": w, "day": base["day"]}
                val = objective_value(data, trial, lambda1, lambda2)
                if val + 1e-9 < best_val:
                    best_val = val
                    best_move = {"ward": w, "day": base["day"]}

            # Apply the best move if it improves the objective
            if best_move != base:
                current[pid] = best_move
                curr_val = best_val
                improved = True
                no_improve = False

        if no_improve:
            break

    return current, curr_val


# =====================================================================
#                      ITERATED LOCAL SEARCH (ILS)
# =====================================================================

class IteratedLocalSearch:
    """
    Iterated Local Search (ILS) for Patient Allocation.

    Features:
    - Fast local search operating on a subset of patients and days
    - Penalized objective to handle temporary infeasibility
    - Mixed perturbation (random + worst contributors)
    - Stagnation-based perturbation intensification
    """

    def __init__(
        self,
        data: PatientAllocationData,
        lambda1=0.5,
        lambda2=0.5,
        penalty_weight=1000.0
    ):
        self.data = data
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.penalty_weight = penalty_weight

        self.best_feasible_allocation = None
        self.best_feasible_value = float("inf")
        self.solve_time = None

        # Number of iterations without improvement before triggering perturbation
        self.STAGNATION_LIMIT = 20
        # Base and max ratio of patients perturbed when diversification is needed
        self.PERTURB_BASE_RATIO = 0.10
        self.PERTURB_MAX_RATIO = 0.25

    def _perturbation_mixed(self, allocation, intensity=0.10):
        """
        Apply a mixed perturbation to the allocation.

        - Selects up to 'intensity' * num_patients patients to modify.
        - Half of them are the worst contributors in terms of delay.
        - The other half are randomly selected.

        For each selected patient:
        - assign a random compatible ward
        - assign a random day within their [earliest, latest] window
        """
        perturbed = copy.deepcopy(allocation)
        num_to_perturb = max(1, int(len(self.data.patients) * intensity))

        # Identify worst contributors w.r.t delay
        worst_pids = []
        for pid in self.data.patients.keys():
            p = self.data.patients[pid]
            a = allocation[pid]
            delay_score = max(0, a["day"] - p["earliest"])
            worst_pids.append((delay_score, pid))
        worst_pids.sort(reverse=True)
        worst_contributors = [pid for _, pid in worst_pids[:num_to_perturb // 2]]

        # 50% random patients + 50% worst contributors
        num_random = num_to_perturb // 2
        patients_random = random.sample(list(self.data.patients.keys()), num_random)
        patients_to_perturb = patients_random + worst_contributors

        for pid in patients_to_perturb:
            p = self.data.patients[pid]
            wards = compatible_wards_for_patient(self.data, p)
            new_ward = random.choice(wards)
            new_day = random.randint(p["earliest"], min(p["latest"], self.data.num_days - 1))
            perturbed[pid] = {"ward": new_ward, "day": new_day}

        return perturbed

    def _local_search_fast(self, allocation, max_time_seconds=30):
        """
        Fast local search for ILS using a penalized objective.

        Optimizations:
        - No deep copy at each move (in-place changes + rollback)
        - Sample only a subset of patients (≈30%) per iteration
        - For each selected patient, try only a few random days
        - Stop when:
          * no improvements are found OR
          * time limit is reached OR
          * max_iterations is reached
        """
        t0 = time.time()
        current = allocation  # Intentionally no deepcopy here
        curr_val = objective_value_penalized(
            self.data, current, self.lambda1, self.lambda2, self.penalty_weight
        )

        iterations = 0
        max_iterations = 10
        num_patients_to_check = max(10, int(len(self.data.patients) * 0.3))

        while iterations < max_iterations:
            if time.time() - t0 > max_time_seconds:
                break

            improved = False
            iterations += 1

            # Sample a subset of patients
            pids_sample = random.sample(
                list(self.data.patients.keys()),
                min(num_patients_to_check, len(self.data.patients))
            )

            for pid in pids_sample:
                if time.time() - t0 > max_time_seconds:
                    break

                p = self.data.patients[pid]
                base = current[pid]

                # Try only a few random days in the allowed window
                possible_days = list(range(p["earliest"], min(p["latest"] + 1, self.data.num_days)))
                days_to_try = random.sample(possible_days, min(3, len(possible_days)))

                for d in days_to_try:
                    if d == base["day"]:
                        continue

                    # In-place modification + restore if no improvement
                    old_day = current[pid]["day"]
                    current[pid]["day"] = d
                    val = objective_value_penalized(
                        self.data, current, self.lambda1, self.lambda2, self.penalty_weight
                    )

                    if val + 1e-9 < curr_val:
                        curr_val = val
                        improved = True
                        break  # First-improvement strategy
                    else:
                        current[pid]["day"] = old_day

                if improved:
                    break

            if not improved:
                break

        return current, curr_val

    def solve(self, start_allocation, max_iterations=50, max_time_minutes=10, verbose=True):
        """
        Run the full Iterated Local Search (ILS) procedure.

        Steps
        -----
        1. Initialize with a feasible allocation (must be feasible).
        2. Repeatedly:
           - apply fast local search
           - if the solution is feasible and better than the current best,
             update the global best
           - if stagnation exceeds STAGNATION_LIMIT, apply perturbation
        3. Stop when:
           - time limit is reached, OR
           - max_iterations is reached, OR
           - extended stagnation is detected

        Returns
        -------
        dict with keys:
            "objective_value", "solve_time", "allocation"
        """
        t0 = time.time()
        max_time_seconds = max_time_minutes * 60

        current = copy.deepcopy(start_allocation)

        # Ensure starting solution is feasible with respect to bed capacity
        if feasible_after_change_beds(self.data, current):
            self.best_feasible_allocation = copy.deepcopy(current)
            self.best_feasible_value = objective_value(self.data, current, self.lambda1, self.lambda2)
        else:
            raise ValueError("Start allocation must be feasible!")

        no_improve_counter = 0

        if verbose:
            print("\n" + "=" * 60)
            print("ITERATED LOCAL SEARCH (ILS) - FAST")
            print("=" * 60)
            print(f"Initial f = {self.best_feasible_value:.2f}")
            print(f"Max time: {max_time_minutes} minutes")

        for iteration in range(max_iterations):
            if time.time() - t0 > max_time_seconds:
                if verbose:
                    print(f"\n⏱️  Timeout reached ({max_time_minutes} min)")
                break

            # Local search from current solution
            current, curr_val_pen = self._local_search_fast(current, max_time_seconds=30)

            # If the locally improved solution is feasible, evaluate non-penalized objective
            if feasible_after_change_beds(self.data, current):
                curr_val = objective_value(self.data, current, self.lambda1, self.lambda2)

                if curr_val + 1e-9 < self.best_feasible_value:
                    self.best_feasible_value = curr_val
                    self.best_feasible_allocation = copy.deepcopy(current)
                    no_improve_counter = 0

                    if verbose:
                        elapsed = time.time() - t0
                        print(f"Iter {iteration} ({elapsed:.0f}s): f = {curr_val:.2f} ✓ NEW BEST")
                else:
                    no_improve_counter += 1
            else:
                # Infeasible after local search → count as non-improvement
                no_improve_counter += 1

            # Stagnation-based perturbation
            if no_improve_counter >= self.STAGNATION_LIMIT:
                intensity = min(
                    self.PERTURB_MAX_RATIO,
                    self.PERTURB_BASE_RATIO * (1 + no_improve_counter / self.STAGNATION_LIMIT)
                )

                if verbose:
                    print(f"Iter {iteration}: Perturbation ({intensity * 100:.0f}% patient)")

                current = self._perturbation_mixed(self.best_feasible_allocation, intensity)
                no_improve_counter = 0

            # Extended stagnation: early stop
            if no_improve_counter >= self.STAGNATION_LIMIT * 2:
                if verbose:
                    print(f"\n⚠️  Stopping early: no improvement")
                break

        self.solve_time = time.time() - t0

        if verbose:
            print(f"\n✓ ILS done in {self.solve_time:.2f}s")
            print(f"Best feasible f: {self.best_feasible_value:.2f}")

        return {
            "objective_value": self.best_feasible_value,
            "solve_time": self.solve_time,
            "allocation": self.best_feasible_allocation
        }


# =====================================================================
#                 VARIABLE NEIGHBORHOOD SEARCH (VNS)
# =====================================================================

class VariableNeighborhoodSearch:
    """
    Variable Neighborhood Search (VNS) for Patient Allocation.

    The algorithm explores a sequence of neighborhoods N1..Nk:

    N1: small day shifts for a single patient
    N2: change ward for a single patient
    N3: multi-patient random modifications (days or wards)
    N4: swap assignments between two patients (if compatible)

    Workflow:
    - Shake (apply Nk)
    - Local search (fast, penalized)
    - If improved:
        * update best solution
        * restart from neighborhood N1
      Else:
        * move to next neighborhood
    """

    def __init__(
        self,
        data: PatientAllocationData,
        lambda1=0.5,
        lambda2=0.5,
        penalty_weight=1000.0
    ):
        self.data = data
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.penalty_weight = penalty_weight

        self.best_feasible_allocation = None
        self.best_feasible_value = float("inf")
        self.solve_time = None

        # Maximum neighborhood index (we have N1..N4)
        self.K_MAX = 4

    def _shake_N1(self, allocation):
        """
        Neighborhood N1: small day shifts for a single patient.

        Randomly picks a patient and shifts their admission day by
        -2, -1, +1, or +2 days, if still within their time window.
        """
        shaken = copy.deepcopy(allocation)
        pid = random.choice(list(self.data.patients.keys()))
        p = self.data.patients[pid]
        current_day = shaken[pid]["day"]
        shifts = [-2, -1, 1, 2]
        shift = random.choice(shifts)
        new_day = current_day + shift
        if p["earliest"] <= new_day <= min(p["latest"], self.data.num_days - 1):
            shaken[pid]["day"] = new_day
        return shaken

    def _shake_N2(self, allocation):
        """
        Neighborhood N2: change ward for a single patient.

        Picks a random patient and moves them to a different
        compatible ward, if there is more than one option.
        """
        shaken = copy.deepcopy(allocation)
        pid = random.choice(list(self.data.patients.keys()))
        p = self.data.patients[pid]
        wards = compatible_wards_for_patient(self.data, p)
        if len(wards) > 1:
            new_ward = random.choice([w for w in wards if w != shaken[pid]["ward"]])
            shaken[pid]["ward"] = new_ward
        return shaken

    def _shake_N3(self, allocation):
        """
        Neighborhood N3: multi-patient perturbation.

        Randomly selects up to 2 patients and either:
        - randomly change their day within their feasible window, OR
        - randomly change their ward within compatible wards.
        """
        shaken = copy.deepcopy(allocation)
        pids_to_move = random.sample(list(self.data.patients.keys()), min(2, len(self.data.patients)))
        for pid in pids_to_move:
            p = self.data.patients[pid]
            if random.random() < 0.5:
                new_day = random.randint(p["earliest"], min(p["latest"], self.data.num_days - 1))
                shaken[pid]["day"] = new_day
            else:
                wards = compatible_wards_for_patient(self.data, p)
                if len(wards) > 1:
                    new_ward = random.choice(wards)
                    shaken[pid]["ward"] = new_ward
        return shaken

    def _shake_N4(self, allocation):
        """
        Neighborhood N4: swap assignments between two patients.

        Randomly picks two patients and attempts to swap their ward/day.
        The swap is applied only if:
        - each new ward is compatible with the other patient
        - each new day is within the other patient's admissible window
        """
        shaken = copy.deepcopy(allocation)
        if len(self.data.patients) >= 2:
            pids = random.sample(list(self.data.patients.keys()), 2)
            p1_id, p2_id = pids
            p1 = self.data.patients[p1_id]
            p2 = self.data.patients[p2_id]
            a1 = shaken[p1_id]
            a2 = shaken[p2_id]
            p1_wards = compatible_wards_for_patient(self.data, p1)
            p2_wards = compatible_wards_for_patient(self.data, p2)
            if (
                a2["ward"] in p1_wards and
                a1["ward"] in p2_wards and
                p1["earliest"] <= a2["day"] <= min(p1["latest"], self.data.num_days - 1) and
                p2["earliest"] <= a1["day"] <= min(p2["latest"], self.data.num_days - 1)
            ):
                shaken[p1_id] = {"ward": a2["ward"], "day": a2["day"]}
                shaken[p2_id] = {"ward": a1["ward"], "day": a1["day"]}
        return shaken

    def _shake(self, allocation, k):
        """
        Apply the shaking operator corresponding to neighborhood k.
        """
        if k == 1:
            return self._shake_N1(allocation)
        elif k == 2:
            return self._shake_N2(allocation)
        elif k == 3:
            return self._shake_N3(allocation)
        else:
            return self._shake_N4(allocation)

    def _local_search_fast(self, allocation, max_time_seconds=15):
        """
        Fast local search for VNS using the penalized objective.

        Similar structure as in the ILS local search, but:
        - fewer iterations
        - smaller sample of patients
        - fewer days per patient
        """
        t0 = time.time()
        current = allocation
        curr_val = objective_value_penalized(
            self.data, current, self.lambda1, self.lambda2, self.penalty_weight
        )

        iterations = 0
        max_iterations = 5
        num_patients_to_check = max(10, int(len(self.data.patients) * 0.2))

        while iterations < max_iterations:
            if time.time() - t0 > max_time_seconds:
                break

            improved = False
            iterations += 1

            pids_sample = random.sample(
                list(self.data.patients.keys()),
                min(num_patients_to_check, len(self.data.patients))
            )

            for pid in pids_sample:
                if time.time() - t0 > max_time_seconds:
                    break

                p = self.data.patients[pid]
                base = current[pid]

                possible_days = list(range(p["earliest"], min(p["latest"] + 1, self.data.num_days)))
                days_to_try = random.sample(possible_days, min(2, len(possible_days)))

                for d in days_to_try:
                    if d == base["day"]:
                        continue

                    old_day = current[pid]["day"]
                    current[pid]["day"] = d
                    val = objective_value_penalized(
                        self.data, current, self.lambda1, self.lambda2, self.penalty_weight
                    )

                    if val + 1e-9 < curr_val:
                        curr_val = val
                        improved = True
                        break
                    else:
                        current[pid]["day"] = old_day

                if improved:
                    break

            if not improved:
                break

        return current, curr_val

    def solve(self, start_allocation, max_iterations=100, max_time_minutes=10, verbose=True):
        """
        Run the full Variable Neighborhood Search (VNS) procedure.

        Steps
        -----
        1. Initialize with a feasible allocation.
        2. For k = 1..K_MAX:
            - Apply shaking in neighborhood Nk
            - Run local search from the shaken solution
            - If feasible and improved:
                * update global best
                * restart k from 1
              Else:
                * increase k (move to next neighborhood)
        3. Stop when:
            - time limit is reached, OR
            - max_iterations is reached.

        Returns
        -------
        dict with keys:
            "objective_value", "solve_time", "allocation"
        """
        t0 = time.time()
        max_time_seconds = max_time_minutes * 60

        current = copy.deepcopy(start_allocation)

        # Ensure starting solution is feasible
        if feasible_after_change_beds(self.data, current):
            self.best_feasible_allocation = copy.deepcopy(current)
            self.best_feasible_value = objective_value(self.data, current, self.lambda1, self.lambda2)
        else:
            raise ValueError("Start allocation must be feasible!")

        if verbose:
            print("\n" + "=" * 60)
            print("VARIABLE NEIGHBORHOOD SEARCH (VNS) - FAST")
            print("=" * 60)
            print(f"Initial f = {self.best_feasible_value:.2f}")
            print(f"Max time: {max_time_minutes} minutes")

        k = 1
        iteration = 0

        while iteration < max_iterations:
            if time.time() - t0 > max_time_seconds:
                if verbose:
                    print(f"\n⏱️  Timeout reached ({max_time_minutes} min)")
                break

            iteration += 1

            # Shake current solution in neighborhood Nk
            shaken = self._shake(current, k)
            # Local search from shaken solution
            improved_solution, improved_val_pen = self._local_search_fast(
                shaken, max_time_seconds=15
            )

            # If locally improved solution is feasible, update best if needed
            if feasible_after_change_beds(self.data, improved_solution):
                improved_val = objective_value(self.data, improved_solution, self.lambda1, self.lambda2)

                if improved_val + 1e-9 < self.best_feasible_value:
                    self.best_feasible_value = improved_val
                    self.best_feasible_allocation = copy.deepcopy(improved_solution)
                    current = improved_solution
                    k = 1  # Intensification: restart from neighborhood N1

                    if verbose:
                        elapsed = time.time() - t0
                        print(f"Iter {iteration} ({elapsed:.0f}s): f = {improved_val:.2f} ✓ NEW BEST (N{k})")
                else:
                    k += 1
            else:
                # Infeasible solution → just move to next neighborhood
                k += 1

            # Cycle neighborhoods if we exceed K_MAX
            if k > self.K_MAX:
                k = 1

            if iteration % 20 == 0 and verbose:
                elapsed = time.time() - t0
                print(f"Iter {iteration} ({elapsed:.0f}s): exploring N{k}, best = {self.best_feasible_value:.2f}")

        self.solve_time = time.time() - t0

        if verbose:
            print(f"\n✓ VNS done in {self.solve_time:.2f}s")
            print(f"Best feasible f: {self.best_feasible_value:.2f}")

        return {
            "objective_value": self.best_feasible_value,
            "solve_time": self.solve_time,
            "allocation": self.best_feasible_allocation
        }


# =====================================================================
#                             RUNNER
# =====================================================================

def run_metaheuristics(
    data: PatientAllocationData,
    lambda1=0.5,
    lambda2=0.5,
    ils_penalty=1000.0,
    ils_max_iter=50,
    ils_max_time_min=5,
    vns_penalty=1000.0,
    vns_max_iter=100,
    vns_max_time_min=5,
    verbose=True
):
    """
    Run the full metaheuristic pipeline on a given instance:

        1. Greedy feasible construction
        2. Greedy + local improvement
        3. Iterated Local Search (ILS)
        4. Variable Neighborhood Search (VNS)

    Returns a dictionary with:
        - greedy_det: objective, time, allocation
        - local: objective, time, allocation
        - ils: objective, time, allocation
        - vns: objective, time, allocation
        - total_time: total runtime of the pipeline
    """
    t_all = time.time()

    if verbose:
        print("\n" + "=" * 78)
        print("PIPELINE: Greedy -> Local -> ILS (FAST) -> VNS (FAST)")
        print("=" * 78)

    # 1) Greedy feasible solution
    t0 = time.time()
    alloc0 = greedy_feasible_by_window_strict(data)
    t_construct = time.time() - t0
    f0 = objective_value(data, alloc0, lambda1, lambda2)
    if verbose:
        print(f"\nGreedy feasible: f={f0:.2f}  (t={t_construct:.2f}s)")

    # 2) Greedy + local improvement
    t0 = time.time()
    alloc1, f1 = greedy_local_improvement(data, alloc0, lambda1, lambda2, max_rounds=6)
    t_greedy = time.time() - t0
    if verbose:
        print(f"Greedy+LocalImprovement: f={f1:.2f}  (t={t_greedy:.2f}s)")

    # 3) ILS
    ILS = IteratedLocalSearch(data, lambda1, lambda2, penalty_weight=ils_penalty)
    ils_res = ILS.solve(alloc1, max_iterations=ils_max_iter, max_time_minutes=ils_max_time_min, verbose=verbose)

    # 4) VNS
    VNS = VariableNeighborhoodSearch(data, lambda1, lambda2, penalty_weight=vns_penalty)
    vns_res = VNS.solve(alloc1, max_iterations=vns_max_iter, max_time_minutes=vns_max_time_min, verbose=verbose)

    total_time = time.time() - t_all

    if verbose:
        print("\n" + "-" * 78)
        print("SUMMARY")
        print("-" * 78)
        print(f"Greedy  : f={f0:.2f}, t={t_construct:.2f}s")
        print(f"Local   : f={f1:.2f}, t={t_greedy:.2f}s")
        print(f"ILS     : f={ils_res['objective_value']:.2f}, t={ils_res['solve_time']:.2f}s")
        print(f"VNS     : f={vns_res['objective_value']:.2f}, t={vns_res['solve_time']:.2f}s")
        print(f"TOTAL   : {total_time:.2f}s")
        print("-" * 78)

    return {
        "greedy_det": {"f": f0, "t": t_construct, "allocation": alloc0},
        "local": {"f": f1, "t": t_greedy, "allocation": alloc1},
        "ils": ils_res,
        "vns": vns_res,
        "total_time": total_time
    }

def run_metaheuristics_no_opt(
    data: PatientAllocationData,
    lambda1=0.5,
    lambda2=0.5,
    ils_penalty=1000.0,
    ils_max_iter=50,
    ils_max_time_min=5,
    vns_penalty=1000.0,
    vns_max_iter=100,
    vns_max_time_min=5,
    verbose=True
):
    """
    Run the full metaheuristic pipeline on a given instance:

        1. Greedy constructive heuristic
        2. Iterated Local Search (ILS) starting from the greedy solution
        3. Variable Neighborhood Search (VNS) starting from the greedy solution
           (can optionally be started from ILS result as a variant)

    Returns
    -------
    dict
        {
            "greedy_det": {"f", "t", "allocation"},
            "ils": {...},
            "vns": {...},
            "total_time": total runtime in seconds
        }
    """
    t_all = time.time()

    if verbose:
        print("\n" + "=" * 78)
        print("PIPELINE: Greedy -> ILS (FAST) -> VNS (FAST)")
        print("=" * 78)

    # 1) Greedy constructive
    t0 = time.time()
    alloc0 = greedy_feasible_by_window_strict(data)
    t_construct = time.time() - t0
    f0 = objective_value(data, alloc0, lambda1, lambda2)
    if verbose:
        print(f"\nGreedy feasible: f={f0:.2f}  (t={t_construct:.2f}s)")

    # 2) ILS starting from greedy solution
    ILS = IteratedLocalSearch(data, lambda1, lambda2, penalty_weight=ils_penalty)
    ils_res = ILS.solve(
        alloc0,
        max_iterations=ils_max_iter,
        max_time_minutes=ils_max_time_min,
        verbose=verbose
    )

    # 3) VNS starting from greedy solution
    # (optionally could use ils_res["allocation"] instead of alloc0)
    VNS = VariableNeighborhoodSearch(data, lambda1, lambda2, penalty_weight=vns_penalty)
    vns_res = VNS.solve(
        alloc0,
        max_iterations=vns_max_iter,
        max_time_minutes=vns_max_time_min,
        verbose=verbose
    )

    total_time = time.time() - t_all

    if verbose:
        print("\n" + "-" * 78)
        print("SUMMARY")
        print("-" * 78)
        print(f"Greedy  : f={f0:.2f}, t={t_construct:.2f}s")
        print(f"ILS     : f={ils_res['objective_value']:.2f}, t={ils_res['solve_time']:.2f}s")
        print(f"VNS     : f={vns_res['objective_value']:.2f}, t={vns_res['solve_time']:.2f}s")
        print(f"TOTAL   : {total_time:.2f}s")
        print("-" * 78)

    return {
        "greedy_det": {"f": f0, "t": t_construct, "allocation": alloc0},
        "ils": ils_res,
        "vns": vns_res,
        "total_time": total_time
    }

# =====================================================================
#                             MAIN
# =====================================================================

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Metaheuristics for Patient Allocation Problem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default mode (with local improvement)
  python3 metaheuristics_cli.py -f flexible_large.dat
  
  # Run with no_opt mode (without local improvement)
  python3 metaheuristics_cli.py -f flexible_large.dat -o no_opt
  
  # Run with custom parameters
  python3 metaheuristics_cli.py -f data.dat --lambda1 0.6 --lambda2 0.4 --ils-time 10
  
  # Run in quiet mode
  python3 metaheuristics_cli.py -f data.dat -q

Modes:
  default : Greedy -> Local Improvement -> ILS -> VNS
  no_opt  : Greedy -> ILS -> VNS (skips local improvement)
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Path to the .dat file containing the problem instance"
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--optimization",
        type=str,
        choices=["default", "no_opt"],
        default="default",
        help="Optimization mode: 'default' (with local improvement) or 'no_opt' (without). Default: default"
    )
    
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.5,
        help="Weight for objective 1 (operational cost). Default: 0.5"
    )
    
    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.5,
        help="Weight for objective 2 (workload balance). Default: 0.5"
    )
    
    parser.add_argument(
        "--ils-penalty",
        type=float,
        default=1000.0,
        help="Penalty weight for ILS. Default: 1000.0"
    )
    
    parser.add_argument(
        "--ils-iter",
        type=int,
        default=50,
        help="Maximum iterations for ILS. Default: 50"
    )
    
    parser.add_argument(
        "--ils-time",
        type=int,
        default=5,
        help="Maximum time (minutes) for ILS. Default: 5"
    )
    
    parser.add_argument(
        "--vns-penalty",
        type=float,
        default=1000.0,
        help="Penalty weight for VNS. Default: 1000.0"
    )
    
    parser.add_argument(
        "--vns-iter",
        type=int,
        default=100,
        help="Maximum iterations for VNS. Default: 100"
    )
    
    parser.add_argument(
        "--vns-time",
        type=int,
        default=5,
        help="Maximum time (minutes) for VNS. Default: 5"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Run in quiet mode (minimal output)"
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the CLI.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Print header
    if not args.quiet:
        print("\n" + "=" * 78)
        print("METAHEURISTICS FOR PATIENT ALLOCATION")
        print("=" * 78)
        print(f"File: {args.file}")
        print(f"Mode: {args.optimization}")
        print(f"Weights: λ1={args.lambda1}, λ2={args.lambda2}")
        print("=" * 78)
    
    # Load data
    try:
        if not args.quiet:
            print(f"\n📁 Loading data from '{args.file}'...")
        data = PatientAllocationData(args.file)
        if not args.quiet:
            print(f"✓ Loaded {len(data.patients)} patients, {len(data.wards)} wards, {data.num_days} days")
    except FileNotFoundError:
        print(f"❌ ERROR: File '{args.file}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        sys.exit(1)
    
    # Run the appropriate metaheuristic pipeline
    try:
        if args.optimization == "no_opt":
            results = run_metaheuristics_no_opt(
                data=data,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                ils_penalty=args.ils_penalty,
                ils_max_iter=args.ils_iter,
                ils_max_time_min=args.ils_time,
                vns_penalty=args.vns_penalty,
                vns_max_iter=args.vns_iter,
                vns_max_time_min=args.vns_time,
                verbose=not args.quiet
            )
        else:  # default mode
            results = run_metaheuristics(
                data=data,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                ils_penalty=args.ils_penalty,
                ils_max_iter=args.ils_iter,
                ils_max_time_min=args.ils_time,
                vns_penalty=args.vns_penalty,
                vns_max_iter=args.vns_iter,
                vns_max_time_min=args.vns_time,
                verbose=not args.quiet
            )
        
        # Print final results
        if not args.quiet:
            print("\n" + "=" * 78)
            print("✓ EXECUTION COMPLETED SUCCESSFULLY")
            print("=" * 78)
            print(f"Total runtime: {results['total_time']:.2f}s ({results['total_time']/60:.2f} min)")
            print("=" * 78)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
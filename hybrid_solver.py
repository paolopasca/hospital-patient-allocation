"""
Optimized Hybrid Method: Combines metaheuristics + CBC (Open-Source MILP).
Uses the metaheuristic solution as a warm start for CBC through PuLP.

PIPELINE:
1. Greedy Feasible Construction
2. Local Improvement
3. ILS (Iterated Local Search) - Fast
4. CBC MILP with warm start from best metaheuristic solution
"""

import argparse
import sys
import pulp
import time
import copy
from data_parser import PatientAllocationData
from CBC_milp import PatientAllocationMILP_CBC
from metaheuristics import (
    greedy_feasible_by_window_strict,
    greedy_local_improvement,
    IteratedLocalSearch,
    VariableNeighborhoodSearch,
    objective_value,
    feasible_after_change_beds
)


class HybridSolverCBC:
    """
    Optimized hybrid solver that combines metaheuristics with CBC.
    
    Process:
    1. Greedy construction (seconds)
    2. Local improvement (seconds)
    3. ILS/VNS (minutes)
    4. CBC MILP with warm start (minutes)
    """
    
    def __init__(self, data: PatientAllocationData, lambda1=0.5, lambda2=0.5):
        """
        Initialize the hybrid solver.
        
        Args:
            data: PatientAllocationData object with problem data
            lambda1: Weight for objective 1 (operational cost)
            lambda2: Weight for objective 2 (workload balance)
        """
        self.data = data
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Results from each phase
        self.greedy_time = None
        self.greedy_obj = None
        self.greedy_solution = None
        
        self.local_time = None
        self.local_obj = None
        self.local_solution = None
        
        self.metaheuristic_time = None
        self.metaheuristic_obj = None
        self.metaheuristic_solution = None
        self.metaheuristic_name = None
        
        self.milp_time = None
        self.milp_obj = None
        self.milp_solution = None
        
        self.total_time = None
        self.best_solution = None
        self.best_obj = None
    
    def solve(self, 
              metaheuristic='ILS',
              mh_max_iter=50,
              mh_max_time_min=5,
              milp_time_limit=300,
              threads=4,
              use_warm_start=True,
              verbose=True):
        """
        Solve the problem using the optimized hybrid approach.
        
        Args:
            metaheuristic: 'ILS' or 'VNS'
            mh_max_iter: Maximum iterations for metaheuristic
            mh_max_time_min: Maximum time for metaheuristic (minutes)
            milp_time_limit: Time limit for CBC (seconds)
            threads: Number of threads for CBC
            use_warm_start: If True, use warm start in CBC
            verbose: If True, show progress
        
        Returns:
            Dict with complete results
        """
        if verbose:
            print("\n" + "="*78)
            print("OPTIMIZED HYBRID METHOD: METAHEURISTIC + CBC MILP")
            print("="*78)
            print(f"Pipeline: Greedy → Local → {metaheuristic} → CBC (warm start)")
            print("="*78)
        
        total_start = time.time()
        
        # =============================
        # PHASE 1: GREEDY CONSTRUCTION
        # =============================
        if verbose:
            print("\n🔹 PHASE 1: Greedy Feasible Construction")
            print("-" * 78)
        
        t0 = time.time()
        self.greedy_solution = greedy_feasible_by_window_strict(self.data)
        self.greedy_time = time.time() - t0
        self.greedy_obj = objective_value(self.data, self.greedy_solution, 
                                          self.lambda1, self.lambda2)
        
        if verbose:
            print(f"✓ Greedy construction completed")
            print(f"  - Time: {self.greedy_time:.2f}s")
            print(f"  - Objective: {self.greedy_obj:.2f}")
            print(f"  - Patients allocated: {len(self.greedy_solution)}")
        
        # =============================
        # PHASE 2: LOCAL IMPROVEMENT
        # =============================
        if verbose:
            print("\n🔹 PHASE 2: Greedy + Local Improvement")
            print("-" * 78)
        
        t0 = time.time()
        self.local_solution, self.local_obj = greedy_local_improvement(
            self.data, 
            self.greedy_solution, 
            self.lambda1, 
            self.lambda2,
            max_rounds=6
        )
        self.local_time = time.time() - t0
        
        improvement_local = ((self.greedy_obj - self.local_obj) / self.greedy_obj) * 100
        
        if verbose:
            print(f"✓ Local improvement completed")
            print(f"  - Time: {self.local_time:.2f}s")
            print(f"  - Objective: {self.local_obj:.2f}")
            print(f"  - Improvement: {improvement_local:.2f}%")
        
        # =============================
        # PHASE 3: METAHEURISTIC (ILS/VNS)
        # =============================
        if verbose:
            print(f"\n🔹 PHASE 3: {metaheuristic} (Metaheuristic)")
            print("-" * 78)
        
        self.metaheuristic_name = metaheuristic
        
        if metaheuristic == 'ILS':
            solver = IteratedLocalSearch(
                self.data, 
                self.lambda1, 
                self.lambda2, 
                penalty_weight=1000.0
            )
            mh_results = solver.solve(
                self.local_solution,
                max_iterations=mh_max_iter,
                max_time_minutes=mh_max_time_min,
                verbose=verbose
            )
        elif metaheuristic == 'VNS':
            solver = VariableNeighborhoodSearch(
                self.data, 
                self.lambda1, 
                self.lambda2, 
                penalty_weight=1000.0
            )
            mh_results = solver.solve(
                self.local_solution,
                max_iterations=mh_max_iter,
                max_time_minutes=mh_max_time_min,
                verbose=verbose
            )
        else:
            raise ValueError(f"Metaheuristic '{metaheuristic}' not recognized. Use 'ILS' or 'VNS'.")
        
        self.metaheuristic_time = mh_results['solve_time']
        self.metaheuristic_solution = mh_results['allocation']
        self.metaheuristic_obj = mh_results['objective_value']
        
        improvement_mh = ((self.local_obj - self.metaheuristic_obj) / self.local_obj) * 100
        
        if verbose:
            print(f"\n✓ {metaheuristic} completed")
            print(f"  - Time: {self.metaheuristic_time:.2f}s")
            print(f"  - Objective: {self.metaheuristic_obj:.2f}")
            print(f"  - Improvement over local: {improvement_mh:.2f}%")
        
        # =============================
        # PHASE 4: CBC MILP WITH WARM START
        # =============================
        if verbose:
            print(f"\n🔹 PHASE 4: CBC MILP (Branch & Bound)")
            print("-" * 78)
            if use_warm_start:
                print("  ⚡ Using warm start from best metaheuristic solution")
        
        # Create MILP model
        milp = PatientAllocationMILP_CBC(self.data, self.lambda1, self.lambda2)
        milp.build_model()
        
        # Apply warm start if requested
        if use_warm_start and feasible_after_change_beds(self.data, self.metaheuristic_solution):
            self._apply_warm_start(milp, self.metaheuristic_solution)
            if verbose:
                print("  ✓ Warm start applied successfully")
        elif use_warm_start:
            if verbose:
                print("  ⚠ Metaheuristic solution not feasible - no warm start")
        
        # Solve with CBC
        milp_start = time.time()
        final_results = milp.solve(
            time_limit=milp_time_limit, 
            threads=threads, 
            verbose=False
        )
        self.milp_time = time.time() - milp_start
        
        if final_results:
            self.milp_solution = final_results['solution']
            self.milp_obj = final_results['objective_value']
            
            if verbose:
                improvement_milp = ((self.metaheuristic_obj - self.milp_obj) / self.metaheuristic_obj) * 100
                print(f"\n✓ CBC MILP completed")
                print(f"  - Time: {self.milp_time:.2f}s")
                print(f"  - Objective: {self.milp_obj:.2f}")
                print(f"  - Improvement over {metaheuristic}: {improvement_milp:.2f}%")
                print(f"  - Status: {final_results['status']}")
        else:
            if verbose:
                print("\n⚠ CBC did not find a better solution")
                print(f"  - Using {metaheuristic} solution")
        
        self.total_time = time.time() - total_start
        
        # Determine best solution
        if self.milp_obj and self.milp_obj < self.metaheuristic_obj:
            self.best_solution = self.milp_solution
            self.best_obj = self.milp_obj
            best_method = "CBC MILP"
        else:
            self.best_solution = self.metaheuristic_solution
            self.best_obj = self.metaheuristic_obj
            best_method = metaheuristic
        
        # =============================
        # FINAL RESULTS
        # =============================
        if verbose:
            self._print_final_results(best_method)
        
        return self._get_results_dict(best_method)
    
    def _apply_warm_start(self, milp: PatientAllocationMILP_CBC, solution):
        """
        Apply the metaheuristic solution as an initial point in PuLP.
        
        Note: PuLP does not have native warm start support like Gurobi.
        
        Args:
            milp: PatientAllocationMILP_CBC object
            solution: Allocation dictionary {patient_id: {'ward': ..., 'day': ...}}
        """
        # PuLP/CBC doesn't support warm start in the same way as Gurobi
        # Set variable initial values 
        
        # For each patient in the solution
        for patient_id, alloc in solution.items():
            ward = alloc['ward']
            day = alloc['day']
            
            # If the variable exists in the model, set initial value
            if (patient_id, ward, day) in milp.y:
                try:
                    milp.y[patient_id, ward, day].setInitialValue(1)
                except:
                    pass  # Solver may not support it
                
    
    def _print_final_results(self, best_method):
        """Print final results summary."""
        print("\n" + "="*78)
        print("FINAL RESULTS - HYBRID METHOD")
        print("="*78)
        
        print(f"\n⏱️  EXECUTION TIMES:")
        print(f"  1. Greedy Construction:  {self.greedy_time:>8.2f}s")
        print(f"  2. Local Improvement:    {self.local_time:>8.2f}s")
        print(f"  3. {self.metaheuristic_name:<20} {self.metaheuristic_time:>8.2f}s")
        print(f"  4. CBC MILP:             {self.milp_time:>8.2f}s")
        print(f"  {'='*35}")
        print(f"  TOTAL:                   {self.total_time:>8.2f}s")
        
        print(f"\n📊 OBJECTIVE PROGRESSION:")
        print(f"  1. Greedy:               {self.greedy_obj:>12.2f}")
        
        improvement_local = ((self.greedy_obj - self.local_obj) / self.greedy_obj) * 100
        print(f"  2. Local:                {self.local_obj:>12.2f}  ({improvement_local:+.2f}%)")
        
        improvement_mh = ((self.local_obj - self.metaheuristic_obj) / self.local_obj) * 100
        print(f"  3. {self.metaheuristic_name:<20} {self.metaheuristic_obj:>12.2f}  ({improvement_mh:+.2f}%)")
        
        if self.milp_obj:
            improvement_milp = ((self.metaheuristic_obj - self.milp_obj) / self.metaheuristic_obj) * 100
            print(f"  4. CBC MILP:             {self.milp_obj:>12.2f}  ({improvement_milp:+.2f}%)")
        else:
            print(f"  4. CBC MILP:             {'N/A':>12}  (no improvement)")
        
        print(f"  {'='*45}")
        total_improvement = ((self.greedy_obj - self.best_obj) / self.greedy_obj) * 100
        print(f"  BEST (via {best_method:<8}): {self.best_obj:>12.2f}  ({total_improvement:+.2f}%)")
        
        print(f"\n💡 CONCLUSION:")
        print(f"  ✓ Best solution obtained via: {best_method}")
        print(f"  ✓ Total time: {self.total_time:.2f}s ({self.total_time/60:.1f} min)")
        print(f"  ✓ Total improvement: {total_improvement:.2f}% over initial greedy")
        print(f"  ✓ Patients allocated: {len(self.best_solution)}/{len(self.data.patients)}")
        
        print("="*78)
    
    def _get_results_dict(self, best_method):
        """Return complete dictionary with all results."""
        return {
            'best_method': best_method,
            'best_objective': self.best_obj,
            'best_solution': self.best_solution,
            'total_time': self.total_time,
            
            'greedy': {
                'objective': self.greedy_obj,
                'time': self.greedy_time,
                'solution': self.greedy_solution
            },
            
            'local': {
                'objective': self.local_obj,
                'time': self.local_time,
                'solution': self.local_solution,
                'improvement_pct': ((self.greedy_obj - self.local_obj) / self.greedy_obj) * 100
            },
            
            'metaheuristic': {
                'name': self.metaheuristic_name,
                'objective': self.metaheuristic_obj,
                'time': self.metaheuristic_time,
                'solution': self.metaheuristic_solution,
                'improvement_pct': ((self.local_obj - self.metaheuristic_obj) / self.local_obj) * 100
            },
            
            'milp': {
                'objective': self.milp_obj,
                'time': self.milp_time,
                'solution': self.milp_solution,
                'improvement_pct': ((self.metaheuristic_obj - self.milp_obj) / self.metaheuristic_obj) * 100 if self.milp_obj else 0
            },
            
            'total_improvement_pct': ((self.greedy_obj - self.best_obj) / self.greedy_obj) * 100,
            'num_patients_allocated': len(self.best_solution) if self.best_solution else 0
        }
    
    def print_solution_details(self, max_patients=10):
        """Print details of the best solution."""
        if not self.best_solution:
            print("No solution available.")
            return
        
        print("\n" + "="*78)
        print("BEST SOLUTION DETAILS")
        print("="*78)
        print(f"Method: {self._get_results_dict('')['best_method']}")
        print(f"Objective: {self.best_obj:.2f}")
        print(f"Total time: {self.total_time:.2f}s")
        
        print(f"\nFirst {max_patients} allocated patients:")
        print("-" * 78)
        
        for i, (patient_id, alloc) in enumerate(list(self.best_solution.items())[:max_patients]):
            patient = self.data.patients[patient_id]
            print(f"\n{patient_id}:")
            print(f"  Specialization: {patient['specialization']}")
            print(f"  Ward: {alloc['ward']}")
            print(f"  Admission day: {alloc['day']}")
            print(f"  Allowed window: [{patient['earliest']}, {patient['latest']}]")
            print(f"  Length of stay: {patient['los']} days")
            
            # Calculate delay
            delay = max(0, alloc['day'] - patient['earliest'])
            if delay > 0:
                print(f"  ⚠ Delay: {delay} days")


# =======================
# TEST AND DEMONSTRATION
# =======================

def parse_arguments():
    """
    Parse command-line arguments for the Hybrid Solver.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Hybrid Solver for Patient Allocation Problem (Metaheuristics + MILP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (ILS metaheuristic)
  python3 hybrid_solver_cli.py -f flexible_large.dat
  
  # Run with VNS metaheuristic
  python3 hybrid_solver_cli.py -f flexible_large.dat -m VNS
  
  # Run with custom time limits
  python3 hybrid_solver_cli.py -f data.dat --mh-time 5 --milp-time 300
  
  # Run with custom weights
  python3 hybrid_solver_cli.py -f data.dat --lambda1 0.6 --lambda2 0.4
  
  # Run in quiet mode
  python3 hybrid_solver_cli.py -f data.dat -q
  
  # Disable warm start
  python3 hybrid_solver_cli.py -f data.dat --no-warm-start
  
  # Run comparison between ILS and VNS
  python3 hybrid_solver_cli.py -f data.dat --compare
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Path to the .dat file containing the problem instance"
    )
    
    # Metaheuristic selection
    parser.add_argument(
        "-m", "--metaheuristic",
        type=str,
        choices=["ILS", "VNS"],
        default="ILS",
        help="Metaheuristic to use: 'ILS' (Iterated Local Search) or 'VNS' (Variable Neighborhood Search). Default: ILS"
    )
    
    # Objective weights
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
    
    # Metaheuristic parameters
    parser.add_argument(
        "--mh-iter",
        type=int,
        default=50,
        help="Maximum iterations for metaheuristic. Default: 50"
    )
    
    parser.add_argument(
        "--mh-time",
        type=int,
        default=3,
        help="Maximum time (minutes) for metaheuristic phase. Default: 3"
    )
    
    # MILP parameters
    parser.add_argument(
        "--milp-time",
        type=int,
        default=180,
        help="Time limit (seconds) for MILP solver (CBC). Default: 180"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for CBC solver. Default: 4"
    )
    
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable warm start (don't use metaheuristic solution as initial MILP solution)"
    )
    
    # Output options
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Run in quiet mode (minimal output)"
    )
    
    parser.add_argument(
        "--max-patients",
        type=int,
        default=5,
        help="Maximum number of patients to show in detailed output. Default: 5"
    )
    
    # Comparison mode
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison between ILS and VNS (takes longer)"
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the Hybrid Solver CLI.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Print header
    if not args.quiet:
        print("\n" + "=" * 78)
        print("HYBRID SOLVER FOR PATIENT ALLOCATION")
        print("Metaheuristics + Mixed Integer Linear Programming")
        print("=" * 78)
        print(f"File: {args.file}")
        print(f"Metaheuristic: {args.metaheuristic}")
        print(f"Weights: λ1={args.lambda1}, λ2={args.lambda2}")
        print(f"MH Time: {args.mh_time} min, MILP Time: {args.milp_time} sec")
        print(f"Warm Start: {'Enabled' if not args.no_warm_start else 'Disabled'}")
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
    
    # Run solver or comparison
    try:
    

        # Run single solver
        if not args.quiet:
            print("\n" + "=" * 78)
            print(f"RUNNING HYBRID SOLVER WITH {args.metaheuristic}")
            print("=" * 78)
            
            
        hybrid = HybridSolverCBC(data, lambda1=args.lambda1, lambda2=args.lambda2)
        results = hybrid.solve(
                metaheuristic=args.metaheuristic,
                mh_max_iter=args.mh_iter,
                mh_max_time_min=args.mh_time,
                milp_time_limit=args.milp_time,
                threads=args.threads,
                use_warm_start=not args.no_warm_start,
                verbose=not args.quiet
            )
             
        # Show solution details
        if not args.quiet:
                hybrid.print_solution_details(max_patients=args.max_patients)
            

        print(f"Would run: metaheuristic={args.metaheuristic}, mh_time={args.mh_time}min, milp_time={args.milp_time}s")
        
        if not args.quiet:
            print("\n" + "=" * 78)
            print("✓ EXECUTION COMPLETED SUCCESSFULLY")
            print("=" * 78)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
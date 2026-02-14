"""
MILP Model (Mixed Integer Linear Programming) for the patient allocation problem.
Uses CBC (open-source) as solver through the PuLP library.

CBC VERSION - Conversion from Gurobi model to free solver
"""

import argparse
import sys
import pulp
import time
from data_parser import PatientAllocationData


class PatientAllocationMILP_CBC:
    """MILP model for patient allocation in hospitals using CBC."""
    
    def __init__(self, data: PatientAllocationData, lambda1=0.5, lambda2=0.5):
        """
        Initialize the MILP model.
        
        Args:
            data: Object with problem data
            lambda1: Weight for objective 1 (operational cost)
            lambda2: Weight for objective 2 (workload balance)
        """
        self.data = data
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Create PuLP model
        self.model = pulp.LpProblem("PatientAllocation", pulp.LpMinimize)
        
        # Decision variables
        self.y = {}  # y[p,w,d] = 1 if patient p is admitted to ward w on day d
        self.x = {}  # x[w,d] = normalized workload in ward w on day d
        self.z = None  # z = maximum workload (to minimize)
        self.v_overtime = {}   # v[s,d] = overtime for specialization s on day d
        self.u_undertime = {}  # u[s,d] = undertime for specialization s on day d
        
        # Auxiliary variables
        self.solution = None
        self.objective_value = None
        self.solve_time = None
        self.status = None
        
    def build_model(self):
        """Build the complete model with variables, constraints, and objective function."""
        print("Building MILP model with CBC...")
        
        # 1. CREATE DECISION VARIABLES
        self._create_variables()
        
        # 2. ADD CONSTRAINTS
        self._add_constraints()
        
        # 3. DEFINE OBJECTIVE FUNCTION
        self._set_objective()
        
        print("✓ Model built successfully!")
        print(f"  - Variables: {len(self.model.variables())}")
        print(f"  - Constraints: {len(self.model.constraints)}")
    
    def _create_variables(self):
        """Create all decision variables of the model."""
        
        # Y[p,w,d] - Binary allocation variable
        print("  Creating Y variables (allocation)...")
        for patient_id, patient in self.data.patients.items():
            spec = patient['specialization']
            earliest = patient['earliest']
            latest = patient['latest']
            
            for ward_name, ward in self.data.wards.items():
                # Only create variable if the ward accepts the specialization
                if (spec == ward['major_specialization'] or 
                    spec in ward['minor_specializations']):
                    
                    for d in range(earliest, min(latest + 1, self.data.num_days)):
                        self.y[patient_id, ward_name, d] = pulp.LpVariable(
                            f"y_{patient_id}_{ward_name}_{d}",
                            cat='Binary'
                        )
        
        # X[w,d] - Normalized workload
        print("  Creating X variables (workload)...")
        for ward_name in self.data.wards.keys():
            for d in range(self.data.num_days):
                self.x[ward_name, d] = pulp.LpVariable(
                    f"x_{ward_name}_{d}",
                    lowBound=0,
                    cat='Continuous'
                )
        
        # Z - Maximum workload
        print("  Creating Z variable (maximum)...")
        self.z = pulp.LpVariable(
            "z_max_workload",
            lowBound=0,
            cat='Continuous'
        )
        
        # V[s,d] - Overtime
        print("  Creating V variables (overtime)...")
        for spec in self.data.specialisms.keys():
            for d in range(self.data.num_days):
                self.v_overtime[spec, d] = pulp.LpVariable(
                    f"v_overtime_{spec}_{d}",
                    lowBound=0,
                    cat='Continuous'
                )
        
        # U[s,d] - Undertime
        print("  Creating U variables (undertime)...")
        for spec in self.data.specialisms.keys():
            for d in range(self.data.num_days):
                self.u_undertime[spec, d] = pulp.LpVariable(
                    f"u_undertime_{spec}_{d}",
                    lowBound=0,
                    cat='Continuous'
                )
    
    def _add_constraints(self):
        """Add all model constraints."""
        
        # CONSTRAINT 1: Each patient must be admitted exactly once
        print("  Adding constraint: each patient admitted once...")
        for patient_id in self.data.patients.keys():
            constraint_vars = [self.y[key] for key in self.y.keys() if key[0] == patient_id]
            self.model += (
                pulp.lpSum(constraint_vars) == 1,
                f"admit_once_{patient_id}"
            )
        
        # CONSTRAINT 2: Ward bed capacity
        print("  Adding constraint: bed capacity...")
        for ward_name, ward in self.data.wards.items():
            for d in range(self.data.num_days):
                # Sum ONLY the decision variables
                patients_in_ward = []
                
                for patient_id, patient in self.data.patients.items():
                    for admit_day in range(max(0, d - patient['los'] + 1), min(d + 1, self.data.num_days)):
                        if (patient_id, ward_name, admit_day) in self.y:
                            if admit_day <= d < admit_day + patient['los']:
                                patients_in_ward.append(self.y[patient_id, ward_name, admit_day])
                
                # Constraint: carryover (constant) + new admissions (variables) <= capacity
                self.model += (
                    ward['carryover_patients'][d] + pulp.lpSum(patients_in_ward) <= ward['bed_capacity'],
                    f"bed_capacity_{ward_name}_{d}"
                )
        
        # CONSTRAINT 3: Operating theater (OT) time
        print("  Adding constraint: operating theater time...")
        for spec in self.data.specialisms.keys():
            for d in range(self.data.num_days):
                # Total time used = sum of surgery durations of patients admitted on day d
                ot_used_vars = []
                
                for (patient_id, ward_name, admit_day) in self.y.keys():
                    if admit_day == d and self.data.patients[patient_id]['specialization'] == spec:
                        surgery_duration = self.data.patients[patient_id]['surgery_duration']
                        ot_used_vars.append(surgery_duration * self.y[patient_id, ward_name, admit_day])
                
                ot_available = self.data.specialisms[spec]['ot_time'][d]
                
                # ot_used + u - v = ot_available
                self.model += (
                    pulp.lpSum(ot_used_vars) + self.u_undertime[spec, d] - self.v_overtime[spec, d] == ot_available,
                    f"ot_time_{spec}_{d}"
                )
        
        # CONSTRAINT 4: Calculation of normalized workload X[w,d]
        print("  Adding constraint: workload calculation...")
        for ward_name, ward in self.data.wards.items():
            workload_capacity = ward['workload_capacity']
            
            for d in range(self.data.num_days):
                # Workload from NEW admissions (only variables)
                workload_terms = []
                
                for patient_id, patient in self.data.patients.items():
                    spec = patient['specialization']
                    los = patient['los']
                    workload_per_day = patient['workload_per_day']
                    
                    # Scaling factor if it's a minor specialization
                    if spec != ward['major_specialization'] and spec in ward['minor_specializations']:
                        scaling_factor = self.data.specialisms[spec]['workload_factor']
                    else:
                        scaling_factor = 1.0
                    
                    # For each possible admission day
                    for admit_day in range(self.data.num_days):
                        if (patient_id, ward_name, admit_day) in self.y:
                            # Check if patient is hospitalized on day d
                            if admit_day <= d < admit_day + los:
                                day_of_stay = d - admit_day
                                if day_of_stay < len(workload_per_day):
                                    workload_contribution = workload_per_day[day_of_stay] * scaling_factor
                                    workload_terms.append(
                                        workload_contribution * self.y[patient_id, ward_name, admit_day]
                                    )
                
                # X[w,d] * capacity = carryover (constant) + new_patients (variables)
                self.model += (
                    self.x[ward_name, d] * workload_capacity == ward['carryover_workload'][d] + pulp.lpSum(workload_terms),
                    f"workload_{ward_name}_{d}"
                )
        
        # CONSTRAINT 5: Z >= X[w,d] for all w,d (Z is the maximum)
        print("  Adding constraint: Z is the maximum workload...")
        for ward_name in self.data.wards.keys():
            for d in range(self.data.num_days):
                self.model += (
                    self.z >= self.x[ward_name, d],
                    f"z_max_{ward_name}_{d}"
                )
    
    def _set_objective(self):
        """Define the objective function (combination of two objectives)."""
        print("  Defining objective function...")
        
        # OBJECTIVE 1: Operational cost (overtime + undertime + delays)
        f1_overtime_terms = []
        for spec in self.data.specialisms.keys():
            for d in range(self.data.num_days):
                f1_overtime_terms.append(self.data.weight_overtime * self.v_overtime[spec, d])
        
        f1_undertime_terms = []
        for spec in self.data.specialisms.keys():
            for d in range(self.data.num_days):
                f1_undertime_terms.append(self.data.weight_undertime * self.u_undertime[spec, d])
        
        f1_delay_terms = []
        for (patient_id, ward_name, d) in self.y.keys():
            delay_cost = self.data.weight_delay * (d - self.data.patients[patient_id]['earliest'])
            f1_delay_terms.append(delay_cost * self.y[patient_id, ward_name, d])
        
        f1 = pulp.lpSum(f1_overtime_terms) + pulp.lpSum(f1_undertime_terms) + pulp.lpSum(f1_delay_terms)
        
        # OBJECTIVE 2: Workload balance (minimize the maximum)
        f2 = self.z
        
        # COMBINED OBJECTIVE
        objective = self.lambda1 * f1 + self.lambda2 * f2
        
        self.model += objective
        
        print(f"  ✓ Objective defined: {self.lambda1}*f1 + {self.lambda2}*f2")
    
    def solve(self, time_limit=600, threads=4, verbose=True):
        """
        Solve the model using CBC.
        
        Args:
            time_limit: Time limit in seconds (default: 600s = 10min)
            threads: Number of threads to use
            verbose: If True, show solver output
        
        Returns:
            Dict with solution results
        """
        print("\n" + "="*60)
        print("SOLVING WITH CBC (Open-Source Solver)")
        print("="*60)
        
        # Configure CBC solver
        solver = pulp.PULP_CBC_CMD(
            timeLimit=time_limit,
            threads=threads,
            msg=verbose,
            options=[
                f'sec {time_limit}',  # Time limit
                f'threads {threads}',  # Number of threads
                'ratioGap 0.01'  # MIP gap tolerance (1%)
            ]
        )
        
        # Solve
        start_time = time.time()
        self.status = self.model.solve(solver)
        self.solve_time = time.time() - start_time
        
        # Process results
        if self.status == pulp.LpStatusOptimal:
            print(f"\n✓ OPTIMAL SOLUTION FOUND!")
            self.objective_value = pulp.value(self.model.objective)
            self._extract_solution()
            return self._get_results()
        
        elif self.status == pulp.LpStatusNotSolved:
            print(f"\n⚠ TIME LIMIT REACHED")
            if self.model.objective.value() is not None:
                print(f"  Best solution found (not necessarily optimal)")
                self.objective_value = pulp.value(self.model.objective)
                self._extract_solution()
                return self._get_results()
            else:
                print(f"  No feasible solution found")
                return None
        
        else:
            print(f"\n✗ ERROR: Status = {pulp.LpStatus[self.status]}")
            return None
    
    def _extract_solution(self):
        """Extract the solution from the variables."""
        self.solution = {}
        
        for (patient_id, ward_name, d), var in self.y.items():
            if pulp.value(var) > 0.5:  # Binary variable = 1
                self.solution[patient_id] = {
                    'ward': ward_name,
                    'day': d,
                    'patient_data': self.data.patients[patient_id]
                }
    
    def _get_results(self):
        """Return a dictionary with the results."""
        return {
            'objective_value': self.objective_value,
            'solve_time': self.solve_time,
            'solution': self.solution,
            'num_patients': len(self.solution),
            'status': pulp.LpStatus[self.status]
        }
    
    def print_solution(self, max_patients=10):
        """Print the solution in a readable format."""
        if not self.solution:
            print("No solution available.")
            return
        
        print("\n" + "="*60)
        print("SOLUTION")
        print("="*60)
        print(f"Objective value: {self.objective_value:.2f}")
        print(f"Solution time: {self.solve_time:.2f}s")
        print(f"Patients allocated: {len(self.solution)}")
        print(f"Status: {pulp.LpStatus[self.status]}")
        
        print(f"\nFirst {max_patients} patients:")
        print("-" * 60)
        
        for i, (patient_id, alloc) in enumerate(list(self.solution.items())[:max_patients]):
            patient = alloc['patient_data']
            print(f"\n{patient_id}:")
            print(f"  Specialization: {patient['specialization']}")
            print(f"  Ward: {alloc['ward']}")
            print(f"  Admission day: {alloc['day']}")
            print(f"  Allowed window: [{patient['earliest']}, {patient['latest']}]")
            print(f"  Length of stay: {patient['los']} days")


def parse_arguments():
    """
    Parse command-line arguments for the MILP Solver.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="MILP Solver (CBC) for Patient Allocation Problem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python3 milp_solver_cli.py -f flexible_large.dat
  
  # Run with custom time limit
  python3 milp_solver_cli.py -f data.dat -t 600
  
  # Run with custom weights
  python3 milp_solver_cli.py -f data.dat --lambda1 0.6 --lambda2 0.4
  
  # Run with more threads
  python3 milp_solver_cli.py -f data.dat --threads 8
  
  # Run in quiet mode
  python3 milp_solver_cli.py -f data.dat -q
  
  # Set MIP gap tolerance
  python3 milp_solver_cli.py -f data.dat --gap 0.01
  
  # Show detailed solution
  python3 milp_solver_cli.py -f data.dat --show-details
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Path to the .dat file containing the problem instance"
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
    
    # Solver parameters
    parser.add_argument(
        "-t", "--time-limit",
        type=int,
        default=300,
        help="Time limit in seconds for CBC solver. Default: 300"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for CBC solver. Default: 4"
    )
    
    parser.add_argument(
        "--gap",
        type=float,
        default=None,
        help="MIP gap tolerance (e.g., 0.01 for 1%%). If not specified, uses solver default"
    )
    
    parser.add_argument(
        "--emphasis",
        type=str,
        choices=["feasibility", "optimality", "balanced"],
        default=None,
        help="Search emphasis: 'feasibility', 'optimality', or 'balanced'"
    )
    
    # Output options
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Run in quiet mode (minimal output)"
    )
    
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed solution breakdown"
    )
    
    parser.add_argument(
        "--max-patients",
        type=int,
        default=10,
        help="Maximum number of patients to show in detailed output. Default: 10"
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the MILP Solver CLI.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Print header
    if not args.quiet:
        print("\n" + "=" * 78)
        print("MILP SOLVER (CBC) FOR PATIENT ALLOCATION")
        print("=" * 78)
        print(f"File: {args.file}")
        print(f"Weights: λ1={args.lambda1}, λ2={args.lambda2}")
        print(f"Time Limit: {args.time_limit} seconds ({args.time_limit/60:.1f} minutes)")
        print(f"Threads: {args.threads}")
        if args.gap is not None:
            print(f"MIP Gap: {args.gap * 100:.2f}%")
        if args.emphasis:
            print(f"Emphasis: {args.emphasis}")
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
    
    # Create and solve model
    try:
        if not args.quiet:
            print("\n" + "=" * 78)
            print("BUILDING AND SOLVING MILP MODEL")
            print("=" * 78)
        
        if not args.quiet:
            print("\n🔨 Building MILP model...")
        
        model = PatientAllocationMILP_CBC(data, lambda1=args.lambda1, lambda2=args.lambda2)
        model.build_model()
        
        if not args.quiet:
            print("✓ Model built successfully")
            print(f"\n🔍 Solving with CBC (time limit: {args.time_limit}s, threads: {args.threads})...")
        
        # Prepare solver parameters
        solver_params = {
            'time_limit': args.time_limit,
            'threads': args.threads
        }
        
        if args.gap is not None:
            solver_params['mip_gap'] = args.gap
        
        if args.emphasis:
            solver_params['emphasis'] = args.emphasis
        
        # Solve
        results = model.solve(**solver_params)
        
        if results:
            if not args.quiet:
                print("\n✓ Solution found!")
                print("\n" + "=" * 78)
                print("SOLUTION SUMMARY")
                print("=" * 78)
            
            # Print solution
            if args.show_details:
                model.print_solution(max_patients=args.max_patients)
            else:
                model.print_solution()
            
            if not args.quiet:
                print("\n" + "=" * 78)
                print("✓ EXECUTION COMPLETED SUCCESSFULLY")
                print("=" * 78)
            
            return 0
        else:
            print("\n❌ No solution found within time limit or problem is infeasible")
            return 1
        
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
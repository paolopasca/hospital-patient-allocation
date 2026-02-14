"""
Pareto Front Generator for Patient Allocation Algorithms

This script runs multiple iterations of algorithms with different lambda weights
and generates a Pareto front plot showing the trade-off between fo1 (operational cost)
and fo2 (workload balance).

Usage:
    python pareto_front_generator.py -d dataset1.dat dataset2.dat --runs 10
    python pareto_front_generator.py -d data/*.dat --runs 5
"""

import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

# Import the algorithm modules
from data_parser import PatientAllocationData
from CBC_milp import PatientAllocationMILP_CBC
from metaheuristics import run_metaheuristics, run_metaheuristics_no_opt, compute_objective_components


class ParetoFrontGenerator:
    """
    Run multiple iterations of all algorithms and generate Pareto front plots.
    """
    
    def __init__(self, datasets, lambda2_values=None, output_dir="output_pareto"):
        """
        Initialize the Pareto front generator.
        
        Args:
            datasets: List of paths to .dat files
            lambda2_values: List of lambda2 values to test (lambda1 fixed at 1.0)
            output_dir: Directory to save output plots
        """
        self.datasets = datasets
        
        # Default lambda2 values for Pareto front exploration
        if lambda2_values is None:
            self.lambda2_values = [1, 10, 50, 100, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000, 10000]
        else:
            self.lambda2_values = lambda2_values
        
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Results storage: list of dictionaries for each run
        self.all_results = []
        
    def run_all(self, 
                milp_time_limit=300,
                mh_time=3,
                threads=4):
        """
        Run all algorithms multiple times on all datasets.
        
        Args:
            milp_time_limit: Time limit for pure MILP (seconds)
            mh_time: Time for metaheuristics (minutes)
            threads: Number of threads for solvers
        """
        print("\n" + "=" * 80)
        print("PARETO FRONT GENERATOR - FIXED LAMBDA1 STRATEGY")
        print("=" * 80)
        print(f"Datasets: {len(self.datasets)}")
        print(f"Lambda values per dataset: {len(self.lambda2_values)}")
        print(f"Algorithms: MILP-CBC, MH-ILS, MH-VNS, MH-NoOpt-ILS, MH-NoOpt-VNS")
        print(f"Weight strategy: λ1=1 (fixed), λ2={self.lambda2_values}")
        print("=" * 80)
        
        for dataset_idx, dataset_path in enumerate(self.datasets):
            dataset_name = Path(dataset_path).stem
            print(f"\n{'=' * 80}")
            print(f"DATASET {dataset_idx+1}/{len(self.datasets)}: {dataset_name}")
            print(f"{'=' * 80}")
            
            try:
                print(f"\n📂 Loading data from '{dataset_path}'...")
                data = PatientAllocationData(dataset_path)
                print(f"✓ Loaded: {len(data.patients)} patients, {len(data.wards)} wards, {data.num_days} days")
                
                # Run algorithms with different lambda2 values (lambda1 fixed at 1)
                for run_idx, lambda2 in enumerate(self.lambda2_values):
                    lambda1 = 1.0  # Fixed
                    
                    print(f"\n{'─' * 80}")
                    print(f"RUN {run_idx+1}/{len(self.lambda2_values)} for {dataset_name}")
                    print(f"λ1 = {lambda1} (fixed), λ2 = {lambda2}")
                    print(f"{'─' * 80}")
                    
                    # Run MILP for this specific lambda combination
                    print(f"\nRunning MILP-CBC with λ1={lambda1}, λ2={lambda2}...")
                    milp_run_result = self._run_milp_cbc(data, milp_time_limit, threads, lambda1, lambda2)
                    
                    # Run Metaheuristics (with optimization)
                    print("Running Metaheuristics (ILS + VNS with optimization)...")
                    mh_result = run_metaheuristics(data, lambda1, lambda2, verbose=False)
                    
                    # Run Metaheuristics NO-OPT
                    print("Running Metaheuristics NO-OPT (ILS + VNS without optimization)...")
                    mh_no_opt_result = run_metaheuristics_no_opt(data, lambda1, lambda2, verbose=False)
                    
                    # Store results for this run
                    self._store_run_results(
                        dataset_name, 
                        run_idx,
                        lambda1,
                        lambda2,
                        data,
                        milp_run_result,
                        mh_result, 
                        mh_no_opt_result
                    )
                    
                    print(f"✓ Run {run_idx+1} completed")
                
            except Exception as e:
                print(f"\n❌ ERROR processing dataset '{dataset_name}': {e}")
                import traceback
                traceback.print_exc()
        
        # Generate Pareto front plots
        print(f"\n{'=' * 80}")
        print("GENERATING PARETO FRONT PLOTS")
        print(f"{'=' * 80}")
        self._generate_pareto_plots()
        
        # Print summary statistics
        self._print_summary_statistics()
    
    def _run_milp_cbc(self, data, time_limit, threads, lambda1, lambda2):
        """Run pure MILP solver with CBC."""
        try:
            model = PatientAllocationMILP_CBC(data, lambda1, lambda2)
            model.build_model()
            
            start_time = time.time()
            results = model.solve(time_limit=time_limit, threads=threads, verbose=False)
            elapsed = time.time() - start_time
            
            if results and results['solution']:
                return {
                    'allocation': results['solution'],
                    'objective': results['objective_value'],
                    'time': elapsed,
                    'status': results['status']
                }
            else:
                return {
                    'allocation': None,
                    'objective': None,
                    'time': elapsed,
                    'status': 'NO_SOLUTION'
                }
        except Exception as e:
            print(f"❌ MILP failed: {e}")
            return {
                'allocation': None,
                'objective': None,
                'time': None,
                'status': 'ERROR'
            }
    
    def _compute_fo1_fo2(self, data, allocation):
        """
        Compute fo1 and fo2 from an allocation.
        
        fo1 = weight_overtime * overtime + weight_undertime * undertime + weight_delay * delays
        fo2 = z (maximum normalized workload)
        """
        if allocation is None or len(allocation) == 0:
            return None, None
        
        delays, overtime, undertime, z, _, _ = compute_objective_components(data, allocation)
        fo1 = data.weight_overtime * overtime + data.weight_undertime * undertime + data.weight_delay * delays
        fo2 = z
        
        return fo1, fo2
    
    def _store_run_results(self, dataset_name, run_idx, lambda1, lambda2, data,
                          milp_result, mh_result, mh_no_opt_result):
        """Store results from one run with fo1 and fo2 values."""
        
        # Helper function to extract allocation and convert MILP format
        def extract_allocation(result, is_milp=False):
            if result is None or result.get('allocation') is None:
                return None
            
            alloc = result['allocation']
            
            # MILP returns: {patient_id: {'ward': w, 'day': d, 'patient_data': ...}}
            # Metaheuristics return: {patient_id: {'ward': w, 'day': d}}
            # We need to standardize to metaheuristics format
            if is_milp:
                return {pid: {'ward': a['ward'], 'day': a['day']} for pid, a in alloc.items()}
            return alloc
        
        # Extract allocations
        milp_alloc = extract_allocation(milp_result, is_milp=True)
        ils_alloc = mh_result['ils'].get('allocation')
        vns_alloc = mh_result['vns'].get('allocation')
        ils_no_opt_alloc = mh_no_opt_result['ils'].get('allocation')
        vns_no_opt_alloc = mh_no_opt_result['vns'].get('allocation')
        
        # Compute fo1 and fo2 for each algorithm
        milp_fo1, milp_fo2 = self._compute_fo1_fo2(data, milp_alloc)
        ils_fo1, ils_fo2 = self._compute_fo1_fo2(data, ils_alloc)
        vns_fo1, vns_fo2 = self._compute_fo1_fo2(data, vns_alloc)
        ils_no_opt_fo1, ils_no_opt_fo2 = self._compute_fo1_fo2(data, ils_no_opt_alloc)
        vns_no_opt_fo1, vns_no_opt_fo2 = self._compute_fo1_fo2(data, vns_no_opt_alloc)
        
        # Store MILP results
        if milp_fo1 is not None:
            self.all_results.append({
                'dataset': dataset_name,
                'run': run_idx,
                'lambda1': lambda1,
                'lambda2': lambda2,
                'algorithm': 'MILP-CBC',
                'fo1': milp_fo1,
                'fo2': milp_fo2,
                'time': milp_result['time']
            })
        
        # Store ILS results
        if ils_fo1 is not None:
            self.all_results.append({
                'dataset': dataset_name,
                'run': run_idx,
                'lambda1': lambda1,
                'lambda2': lambda2,
                'algorithm': 'MH-ILS',
                'fo1': ils_fo1,
                'fo2': ils_fo2,
                'time': mh_result['ils']['solve_time']
            })
        
        # Store VNS results
        if vns_fo1 is not None:
            self.all_results.append({
                'dataset': dataset_name,
                'run': run_idx,
                'lambda1': lambda1,
                'lambda2': lambda2,
                'algorithm': 'MH-VNS',
                'fo1': vns_fo1,
                'fo2': vns_fo2,
                'time': mh_result['vns']['solve_time']
            })
        
        # Store ILS NO-OPT results
        if ils_no_opt_fo1 is not None:
            self.all_results.append({
                'dataset': dataset_name,
                'run': run_idx,
                'lambda1': lambda1,
                'lambda2': lambda2,
                'algorithm': 'MH-NoOpt-ILS',
                'fo1': ils_no_opt_fo1,
                'fo2': ils_no_opt_fo2,
                'time': mh_no_opt_result['ils']['solve_time']
            })
        
        # Store VNS NO-OPT results
        if vns_no_opt_fo1 is not None:
            self.all_results.append({
                'dataset': dataset_name,
                'run': run_idx,
                'lambda1': lambda1,
                'lambda2': lambda2,
                'algorithm': 'MH-NoOpt-VNS',
                'fo1': vns_no_opt_fo1,
                'fo2': vns_no_opt_fo2,
                'time': mh_no_opt_result['vns']['solve_time']
            })
    
    def _is_dominated(self, point, points):
        """
        Check if a point is dominated by any other point in the set.
        A point (fo1_a, fo2_a) dominates (fo1_b, fo2_b) if:
        - fo1_a <= fo1_b AND fo2_a <= fo2_b
        - AND at least one inequality is strict
        
        We're minimizing both objectives.
        """
        fo1, fo2 = point
        for other_fo1, other_fo2 in points:
            if (other_fo1, other_fo2) == (fo1, fo2):
                continue
            # Check if other point dominates this point
            if other_fo1 <= fo1 and other_fo2 <= fo2:
                if other_fo1 < fo1 or other_fo2 < fo2:
                    return True
        return False
    
    def _extract_pareto_front(self, df_algo):
        """Extract non-dominated points (Pareto front) from algorithm results."""
        points = list(zip(df_algo['fo1'], df_algo['fo2']))
        
        pareto_points = []
        for i, (fo1, fo2) in enumerate(points):
            if not self._is_dominated((fo1, fo2), points):
                pareto_points.append(i)
        
        return df_algo.iloc[pareto_points]
    
    def _generate_pareto_plots(self):
        """Generate Pareto front plots - 6 separate plots, one per algorithm."""
        if not self.all_results:
            print("⚠️ No results to plot")
            return
        
        df = pd.DataFrame(self.all_results)
        
        # Define colors and styles for each algorithm
        algo_styles = {
            'MILP-CBC': {'color': '#d62728', 'marker': 'o', 'linewidth': 3, 'markersize': 12, 'title': 'MILP-CBC (Optimal)'},
            'MH-ILS': {'color': '#1f77b4', 'marker': 's', 'linewidth': 2.5, 'markersize': 10, 'title': 'Metaheuristic: ILS'},
            'MH-VNS': {'color': '#ff7f0e', 'marker': '^', 'linewidth': 2.5, 'markersize': 10, 'title': 'Metaheuristic: VNS'},
            'MH-NoOpt-ILS': {'color': '#2ca02c', 'marker': 'D', 'linewidth': 2.5, 'markersize': 10, 'title': 'Metaheuristic: ILS (No-Opt)'},
            'MH-NoOpt-VNS': {'color': '#9467bd', 'marker': 'v', 'linewidth': 2.5, 'markersize': 10, 'title': 'Metaheuristic: VNS (No-Opt)'}
        }
        
        # Create separate plot for each algorithm
        for algo_name, style in algo_styles.items():
            df_algo = df[df['algorithm'] == algo_name].copy()
            
            if df_algo.empty:
                print(f"  ⚠️ No data for {algo_name}, skipping...")
                continue
            
            # Extract Pareto front
            df_pareto = self._extract_pareto_front(df_algo)
            
            if df_pareto.empty:
                print(f"  ⚠️ No Pareto points for {algo_name}, skipping...")
                continue
            
            # Sort by fo1 for proper line plotting
            df_pareto = df_pareto.sort_values('fo1')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot all explored points (faded)
            ax.scatter(df_algo['fo1'], df_algo['fo2'], 
                      color=style['color'], 
                      marker=style['marker'],
                      s=100,
                      alpha=0.2,
                      edgecolors='none',
                      label='All explored solutions')
            
            # Plot Pareto front line (continuous curve)
            ax.plot(df_pareto['fo1'], df_pareto['fo2'],
                   color=style['color'],
                   linewidth=style['linewidth'],
                   alpha=0.8,
                   linestyle='-',
                   zorder=2,
                   label='Pareto front')
            
            # Plot Pareto front points (markers on top)
            ax.scatter(df_pareto['fo1'], df_pareto['fo2'],
                      color=style['color'],
                      marker=style['marker'],
                      s=style['markersize']**2 * 5,
                      alpha=1.0,
                      edgecolors='white',
                      linewidths=2,
                      zorder=3,
                      label='Non-dominated solutions')
            
            # Formatting
            ax.set_xlabel('fo1 (Operational Cost: Overtime + Undertime + Delays)', 
                         fontsize=12, fontweight='bold')
            ax.set_ylabel('fo2 (Workload Balance: Max Normalized Workload)', 
                         fontsize=12, fontweight='bold')
            ax.set_title(f'{style["title"]}\nPareto Front (Lower is Better for Both Objectives)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
            
            # Add statistics box
            stats_text = f"Total solutions: {len(df_algo)}\n"
            stats_text += f"Pareto points: {len(df_pareto)}\n"
            stats_text += f"Pareto ratio: {len(df_pareto)/len(df_algo)*100:.1f}%\n"
            stats_text += f"────────────────────\n"
            stats_text += f"fo1 range: [{df_pareto['fo1'].min():.1f}, {df_pareto['fo1'].max():.1f}]\n"
            stats_text += f"fo2 range: [{df_pareto['fo2'].min():.3f}, {df_pareto['fo2'].max():.3f}]"
            
            ax.text(0.98, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   family='monospace')
            
            plt.tight_layout()
            
            # Save plot
            algo_filename = algo_name.replace('-', '_').lower()
            filename = f'{self.output_dir}/pareto_{algo_filename}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filename}")
            plt.close()
    
    def _print_summary_statistics(self):
        """Print summary statistics."""
        if not self.all_results:
            print("⚠️ No results to summarize")
            return
        
        df = pd.DataFrame(self.all_results)
        
        print(f"\n{'=' * 80}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 80}")
        
        print("\n📊 OBJECTIVE COMPONENT STATISTICS")
        print(f"{'─' * 80}")
        print("\nfo1 (Operational Cost):")
        fo1_stats = df.groupby('algorithm')['fo1'].agg(['mean', 'median', 'std', 'min', 'max'])
        print(fo1_stats.to_string())
        
        print("\n\nfo2 (Workload Balance):")
        fo2_stats = df.groupby('algorithm')['fo2'].agg(['mean', 'median', 'std', 'min', 'max'])
        print(fo2_stats.to_string())
        
        print("\n\n📊 EXECUTION TIME STATISTICS (seconds)")
        print(f"{'─' * 80}")
        time_stats = df.groupby('algorithm')['time'].agg(['mean', 'median', 'std', 'min', 'max'])
        print(time_stats.to_string())
        
        # Pareto front statistics
        print(f"\n\n📊 PARETO FRONT STATISTICS")
        print(f"{'─' * 80}")
        for algo in df['algorithm'].unique():
            df_algo = df[df['algorithm'] == algo]
            df_pareto = self._extract_pareto_front(df_algo)
            pareto_percentage = (len(df_pareto) / len(df_algo) * 100) if len(df_algo) > 0 else 0
            print(f"{algo:<20} {len(df_pareto)}/{len(df_algo)} points on Pareto front ({pareto_percentage:.1f}%)")
        
        print(f"\n{'=' * 80}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pareto front generator for patient allocation algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default lambda2 values
  python pareto_front_generator.py -d data1.dat data2.dat
  
  # Run on all .dat files in a directory
  python pareto_front_generator.py -d data/*.dat
  
  # Custom lambda2 values
  python pareto_front_generator.py -d data/*.dat --lambda2 1 10 100 1000 10000
        """
    )
    
    parser.add_argument(
        "-d", "--datasets",
        nargs="+",
        required=True,
        help="List of .dat files to process"
    )
    
    parser.add_argument(
        "--lambda2",
        nargs="+",
        type=float,
        default=None,
        help="Lambda2 values to test (lambda1 fixed at 1.0). Default: [1, 10, 50, 100, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000, 10000]"
    )
    
    parser.add_argument(
        "--milp-time",
        type=int,
        default=300,
        help="Time limit for MILP (seconds). Default: 300"
    )
    
    parser.add_argument(
        "--mh-time",
        type=int,
        default=3,
        help="Time for metaheuristics (minutes). Default: 3"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for solvers. Default: 4"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_pareto",
        help="Output directory for plots. Default: output_pareto"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate datasets exist
    valid_datasets = []
    for dataset in args.datasets:
        if os.path.exists(dataset):
            valid_datasets.append(dataset)
        else:
            print(f"⚠️ Warning: Dataset '{dataset}' not found, skipping...")
    
    if not valid_datasets:
        print("❌ ERROR: No valid datasets found!")
        sys.exit(1)
    
    print(f"\n✓ Found {len(valid_datasets)} valid dataset(s)")
    
    # Create Pareto front generator
    generator = ParetoFrontGenerator(
        valid_datasets,
        lambda2_values=args.lambda2,
        output_dir=args.output
    )
    
    # Run all benchmarks
    try:
        generator.run_all(
            milp_time_limit=args.milp_time,
            mh_time=args.mh_time,
            threads=args.threads
        )
        
        print(f"\n✓ Pareto front generation completed successfully!")
        print(f"  Results saved to: {args.output}/")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Execution interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

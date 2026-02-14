"""
Benchmark Script with Box Plots for Patient Allocation Algorithms

This script runs multiple iterations of 6 different algorithms on datasets and generates
box plots for execution times and solution quality (optimality gap).

Usage:
    python benchmark_boxplot.py -d dataset1.dat dataset2.dat --runs 10
    python benchmark_boxplot.py -d data/*.dat --runs 5 --lambda1 0.5 --lambda2 0.5
"""

import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns

# Import the algorithm modules
from data_parser import PatientAllocationData
from CBC_milp import PatientAllocationMILP_CBC
from metaheuristics import run_metaheuristics, run_metaheuristics_no_opt
from hybrid_solver import HybridSolverCBC


class BenchmarkBoxPlot:
    """
    Run multiple iterations of all algorithms and generate box plots.
    """
    
    def __init__(self, datasets, num_runs=10, lambda1=0.5, lambda2=0.5, output_dir="output_boxplot"):
        """
        Initialize the benchmark runner.
        
        Args:
            datasets: List of paths to .dat files
            num_runs: Number of runs per algorithm per dataset
            lambda1: Weight for objective 1
            lambda2: Weight for objective 2
            output_dir: Directory to save output plots
        """
        self.datasets = datasets
        self.num_runs = num_runs
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Results storage: list of dictionaries for each run
        self.all_results = []
        
    def run_all(self, 
                milp_time_limit=300,
                mh_time=3,
                hybrid_mh_time=3,
                hybrid_milp_time=180,
                threads=4):
        """
        Run all algorithms multiple times on all datasets.
        
        Args:
            milp_time_limit: Time limit for pure MILP (seconds)
            mh_time: Time for metaheuristics (minutes)
            hybrid_mh_time: Time for metaheuristic phase in hybrid (minutes)
            hybrid_milp_time: Time for MILP phase in hybrid (seconds)
            threads: Number of threads for solvers
        """
        print("\n" + "=" * 80)
        print("BENCHMARK WITH BOX PLOTS - MULTIPLE RUNS")
        print("=" * 80)
        print(f"Datasets: {len(self.datasets)}")
        print(f"Runs per dataset: {self.num_runs}")
        print(f"Algorithms: MILP-CBC, MH-ILS, MH-VNS, MH-NoOpt-ILS, MH-NoOpt-VNS, Hybrid")
        print(f"Weights: λ1={self.lambda1}, λ2={self.lambda2}")
        print("=" * 80)
        
        for dataset_idx, dataset_path in enumerate(self.datasets):
            dataset_name = Path(dataset_path).stem
            print(f"\n{'=' * 80}")
            print(f"DATASET {dataset_idx+1}/{len(self.datasets)}: {dataset_name}")
            print(f"{'=' * 80}")
            
            try:
                # Load data once
                print(f"\n📂 Loading data from '{dataset_path}'...")
                data = PatientAllocationData(dataset_path)
                print(f"✓ Loaded: {len(data.patients)} patients, {len(data.wards)} wards, {data.num_days} days")
                
                # Run MILP once to get the optimal value
                print(f"\n{'─' * 80}")
                print("Running MILP-CBC to get optimal value...")
                print(f"{'─' * 80}")
                milp_result = self._run_milp_cbc(data, milp_time_limit, threads)
                milp_optimal = milp_result['objective']
                
                if milp_optimal is None:
                    print(f"⚠️ Warning: MILP did not find a solution for {dataset_name}. Skipping this dataset.")
                    continue
                
                print(f"✓ MILP Optimal Value: {milp_optimal:.2f}")
                print(f"✓ MILP Time: {milp_result['time']:.2f}s")
                
                # Now run metaheuristics and hybrid multiple times
                for run_idx in range(self.num_runs):
                    print(f"\n{'─' * 80}")
                    print(f"RUN {run_idx+1}/{self.num_runs} for {dataset_name}")
                    print(f"{'─' * 80}")
                    
                    # Run Metaheuristics (with optimization)
                    print("\nRunning Metaheuristics (ILS + VNS with optimization)...")
                    mh_result = run_metaheuristics(data)
                    
                    # Run Metaheuristics NO-OPT
                    print("Running Metaheuristics NO-OPT (ILS + VNS without optimization)...")
                    mh_no_opt_result = run_metaheuristics_no_opt(data)
                    
                    # Run Hybrid
                    print("Running Hybrid solver...")
                    hybrid_result = self._run_hybrid(data, hybrid_mh_time, hybrid_milp_time, threads)
                    
                    # Store results for this run
                    self._store_run_results(
                        dataset_name, 
                        run_idx, 
                        milp_optimal,
                        milp_result['time'],
                        mh_result, 
                        mh_no_opt_result, 
                        hybrid_result
                    )
                    
                    print(f"✓ Run {run_idx+1} completed")
                
            except Exception as e:
                print(f"\n❌ ERROR processing dataset '{dataset_name}': {e}")
                import traceback
                traceback.print_exc()
        
        # Generate box plots
        print(f"\n{'=' * 80}")
        print("GENERATING BOX PLOTS")
        print(f"{'=' * 80}")
        self._generate_boxplots()
        
        # Print summary statistics
        self._print_summary_statistics()
    
    def _run_milp_cbc(self, data, time_limit, threads):
        """Run pure MILP solver with CBC."""
        try:
            model = PatientAllocationMILP_CBC(data, self.lambda1, self.lambda2)
            model.build_model()
            
            start_time = time.time()
            results = model.solve(time_limit=time_limit, threads=threads, verbose=False)
            elapsed = time.time() - start_time
            
            if results:
                return {
                    'objective': results['objective_value'],
                    'time': elapsed,
                    'status': results['status']
                }
            else:
                return {
                    'objective': None,
                    'time': elapsed,
                    'status': 'NO_SOLUTION'
                }
        except Exception as e:
            print(f"❌ MILP failed: {e}")
            return {
                'objective': None,
                'time': None,
                'status': 'ERROR'
            }
    
    def _run_hybrid(self, data, mh_time, milp_time, threads):
        """Run hybrid solver."""
        try:
            hybrid = HybridSolverCBC(data, self.lambda1, self.lambda2)
            results = hybrid.solve(
                metaheuristic='ILS',
                mh_max_iter=50,
                mh_max_time_min=mh_time,
                milp_time_limit=milp_time,
                threads=threads,
                use_warm_start=True,
                verbose=False
            )
            return results
        except Exception as e:
            print(f"❌ Hybrid failed: {e}")
            return {
                'best_objective': None,
                'total_time': None,
                'best_method': 'ERROR'
            }
    
    def _store_run_results(self, dataset_name, run_idx, milp_optimal, milp_time,
                          mh_result, mh_no_opt_result, hybrid_result):
        """Store results from one run."""
        
        # Helper function to calculate optimality gap
        def calc_gap(algorithm_obj, optimal):
            if algorithm_obj is None or optimal is None or optimal == 0:
                return None
            # Gap = 1 - (optimal / algorithm_value)
            # If algorithm is worse (higher), gap is positive
            # If algorithm equals optimal, gap is 0
            gap = 1.0 - (optimal / algorithm_obj)
            return gap * 100  # Convert to percentage
        
        # Extract values
        mh_ils_obj = mh_result['ils']['objective_value']
        mh_ils_time = mh_result['ils']['solve_time']
        
        mh_vns_obj = mh_result['vns']['objective_value']
        mh_vns_time = mh_result['vns']['solve_time']
        
        mh_no_opt_ils_obj = mh_no_opt_result['ils']['objective_value']
        mh_no_opt_ils_time = mh_no_opt_result['ils']['solve_time']
        
        mh_no_opt_vns_obj = mh_no_opt_result['vns']['objective_value']
        mh_no_opt_vns_time = mh_no_opt_result['vns']['solve_time']
        
        hybrid_obj = hybrid_result['best_objective']
        hybrid_time = hybrid_result['total_time']
        
        # Store each algorithm result separately
        algorithms_data = [
            {
                'dataset': dataset_name,
                'run': run_idx,
                'algorithm': 'MILP-CBC',
                'time': milp_time,
                'objective': milp_optimal,
                'gap': 0.0  # MILP is the reference, gap is 0
            },
            {
                'dataset': dataset_name,
                'run': run_idx,
                'algorithm': 'MH-ILS',
                'time': mh_ils_time,
                'objective': mh_ils_obj,
                'gap': calc_gap(mh_ils_obj, milp_optimal)
            },
            {
                'dataset': dataset_name,
                'run': run_idx,
                'algorithm': 'MH-VNS',
                'time': mh_vns_time,
                'objective': mh_vns_obj,
                'gap': calc_gap(mh_vns_obj, milp_optimal)
            },
            {
                'dataset': dataset_name,
                'run': run_idx,
                'algorithm': 'MH-NoOpt-ILS',
                'time': mh_no_opt_ils_time,
                'objective': mh_no_opt_ils_obj,
                'gap': calc_gap(mh_no_opt_ils_obj, milp_optimal)
            },
            {
                'dataset': dataset_name,
                'run': run_idx,
                'algorithm': 'MH-NoOpt-VNS',
                'time': mh_no_opt_vns_time,
                'objective': mh_no_opt_vns_obj,
                'gap': calc_gap(mh_no_opt_vns_obj, milp_optimal)
            },
            {
                'dataset': dataset_name,
                'run': run_idx,
                'algorithm': 'Hybrid',
                'time': hybrid_time,
                'objective': hybrid_obj,
                'gap': calc_gap(hybrid_obj, milp_optimal)
            }
        ]
        
        self.all_results.extend(algorithms_data)
    
    def _generate_boxplots(self):
        """Generate box plots for execution times and optimality gaps."""
        if not self.all_results:
            print("⚠️ No results to plot")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)
        
        # Remove rows with None values
        df_time = df[df['time'].notna()].copy()
        df_gap = df[df['gap'].notna()].copy()
        
        if df_time.empty:
            print("⚠️ No valid time data to plot")
            return
        
        # Set style
        sns.set_style("whitegrid")
        
        # Plot 1: Execution Time Box Plot
        self._plot_time_boxplot(df_time)
        
        # Plot 2: Optimality Gap Box Plot
        if not df_gap.empty:
            self._plot_gap_boxplot(df_gap)
        else:
            print("⚠️ No valid gap data to plot")
        
        # Plot 3: Combined plot for each dataset
        for dataset in df['dataset'].unique():
            df_dataset = df[df['dataset'] == dataset]
            self._plot_dataset_combined(df_dataset, dataset)
        
        print(f"\n✓ Box plots saved to '{self.output_dir}' directory")
    
    def _plot_time_boxplot(self, df):
        """Plot box plot for execution times."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define color palette
        colors = {
            'MILP-CBC': '#2E86AB',
            'MH-ILS': '#792BAE',
            'MH-VNS': '#A23B72',
            'MH-NoOpt-ILS': '#8FA23B',
            'MH-NoOpt-VNS': '#C9184A',
            'Hybrid': '#F18F01'
        }
        
        # Order algorithms
        order = ['MILP-CBC', 'MH-ILS', 'MH-VNS', 'MH-NoOpt-ILS', 'MH-NoOpt-VNS', 'Hybrid']
        
        # Create box plot
        bp = sns.boxplot(
            data=df,
            x='algorithm',
            y='time',
            order=order,
            palette=colors,
            ax=ax
        )
        
        # Add strip plot for individual points
        sns.stripplot(
            data=df,
            x='algorithm',
            y='time',
            order=order,
            color='black',
            alpha=0.3,
            size=3,
            ax=ax
        )
        
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Execution Time Comparison Across All Datasets\n(Lower is Better)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/boxplot_execution_time.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {self.output_dir}/boxplot_execution_time.png")
        plt.close()
    
    def _plot_gap_boxplot(self, df):
        """Plot box plot for optimality gaps."""
        # Exclude MILP from gap plot (it's always 0)
        df_no_milp = df[df['algorithm'] != 'MILP-CBC'].copy()
        
        if df_no_milp.empty:
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define color palette
        colors = {
            'MH-ILS': '#792BAE',
            'MH-VNS': '#A23B72',
            'MH-NoOpt-ILS': '#8FA23B',
            'MH-NoOpt-VNS': '#C9184A',
            'Hybrid': '#F18F01'
        }
        
        # Order algorithms
        order = ['MH-ILS', 'MH-VNS', 'MH-NoOpt-ILS', 'MH-NoOpt-VNS', 'Hybrid']
        
        # Create box plot
        bp = sns.boxplot(
            data=df_no_milp,
            x='algorithm',
            y='gap',
            order=order,
            palette=colors,
            ax=ax
        )
        
        # Add strip plot for individual points
        sns.stripplot(
            data=df_no_milp,
            x='algorithm',
            y='gap',
            order=order,
            color='black',
            alpha=0.3,
            size=3,
            ax=ax
        )
        
        # Add reference line at 0 (optimal)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='MILP Optimal')
        
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Optimality Gap (%)', fontsize=12, fontweight='bold')
        ax.set_title('Optimality Gap from MILP Solution\n(Lower is Better, 0% = Optimal)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/boxplot_optimality_gap.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {self.output_dir}/boxplot_optimality_gap.png")
        plt.close()
    
    def _plot_dataset_combined(self, df, dataset_name):
        """Plot combined time and gap for a specific dataset."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Define color palette
        colors = {
            'MILP-CBC': '#2E86AB',
            'MH-ILS': '#792BAE',
            'MH-VNS': '#A23B72',
            'MH-NoOpt-ILS': '#8FA23B',
            'MH-NoOpt-VNS': '#C9184A',
            'Hybrid': '#F18F01'
        }
        
        order = ['MILP-CBC', 'MH-ILS', 'MH-VNS', 'MH-NoOpt-ILS', 'MH-NoOpt-VNS', 'Hybrid']
        
        # Plot 1: Time
        df_time = df[df['time'].notna()]
        if not df_time.empty:
            sns.boxplot(data=df_time, x='algorithm', y='time', order=order, palette=colors, ax=ax1)
            sns.stripplot(data=df_time, x='algorithm', y='time', order=order, 
                         color='black', alpha=0.3, size=3, ax=ax1)
            ax1.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
            ax1.set_title('Execution Time', fontsize=12, fontweight='bold')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Gap (exclude MILP)
        df_gap = df[(df['gap'].notna()) & (df['algorithm'] != 'MILP-CBC')]
        if not df_gap.empty:
            order_no_milp = [o for o in order if o != 'MILP-CBC']
            sns.boxplot(data=df_gap, x='algorithm', y='gap', order=order_no_milp, palette=colors, ax=ax2)
            sns.stripplot(data=df_gap, x='algorithm', y='gap', order=order_no_milp,
                         color='black', alpha=0.3, size=3, ax=ax2)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='MILP Optimal')
            ax2.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Optimality Gap (%)', fontsize=11, fontweight='bold')
            ax2.set_title('Optimality Gap', fontsize=12, fontweight='bold')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'Dataset: {dataset_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/boxplot_{dataset_name}.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {self.output_dir}/boxplot_{dataset_name}.png")
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
        
        print("\n📊 EXECUTION TIME STATISTICS (seconds)")
        print(f"{'─' * 80}")
        time_stats = df.groupby('algorithm')['time'].agg(['mean', 'median', 'std', 'min', 'max'])
        print(time_stats.to_string())
        
        print(f"\n\n📊 OPTIMALITY GAP STATISTICS (%)")
        print(f"{'─' * 80}")
        df_no_milp = df[df['algorithm'] != 'MILP-CBC']
        if not df_no_milp.empty:
            gap_stats = df_no_milp.groupby('algorithm')['gap'].agg(['mean', 'median', 'std', 'min', 'max'])
            print(gap_stats.to_string())
        
        # Count how many times each algorithm achieved optimal (gap = 0)
        print(f"\n\n🎯 OPTIMAL SOLUTIONS ACHIEVED")
        print(f"{'─' * 80}")
        for algo in df_no_milp['algorithm'].unique():
            df_algo = df_no_milp[df_no_milp['algorithm'] == algo]
            optimal_count = (df_algo['gap'] == 0).sum()
            total_count = len(df_algo)
            percentage = (optimal_count / total_count * 100) if total_count > 0 else 0
            print(f"{algo:<20} {optimal_count}/{total_count} ({percentage:.1f}%)")
        
        print(f"\n{'=' * 80}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark with box plots for patient allocation algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 iterations on multiple datasets
  python benchmark_boxplot.py -d data1.dat data2.dat --runs 10
  
  # Run on all .dat files in a directory
  python benchmark_boxplot.py -d data/*.dat --runs 5
  
  # Custom parameters
  python benchmark_boxplot.py -d data/*.dat --runs 10 --lambda1 0.6 --lambda2 0.4
        """
    )
    
    parser.add_argument(
        "-d", "--datasets",
        nargs="+",
        required=True,
        help="List of .dat files to process"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per algorithm per dataset. Default: 10"
    )
    
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.5,
        help="Weight for objective 1. Default: 0.5"
    )
    
    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.5,
        help="Weight for objective 2. Default: 0.5"
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
        "--hybrid-mh-time",
        type=int,
        default=3,
        help="Time for metaheuristic phase in hybrid (minutes). Default: 3"
    )
    
    parser.add_argument(
        "--hybrid-milp-time",
        type=int,
        default=180,
        help="Time for MILP phase in hybrid (seconds). Default: 180"
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
        default="output_boxplot",
        help="Output directory for plots. Default: output_boxplot"
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
    
    # Create benchmark runner
    runner = BenchmarkBoxPlot(
        valid_datasets,
        num_runs=args.runs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        output_dir=args.output
    )
    
    # Run all benchmarks
    try:
        runner.run_all(
            milp_time_limit=args.milp_time,
            mh_time=args.mh_time,
            hybrid_mh_time=args.hybrid_mh_time,
            hybrid_milp_time=args.hybrid_milp_time,
            threads=args.threads
        )
        
        print(f"\n✓ Benchmark completed successfully!")
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
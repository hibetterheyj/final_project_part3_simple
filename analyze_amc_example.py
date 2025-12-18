# %% [markdown]
# AMC Dataset Analysis Example

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import math

# %% [markdown]
# Data Loading Functions

def load_eval_file(file_path: str) -> List[Dict]:
    """Load a JSONL evaluation file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding line in {file_path}")
    return data


def load_amc_data(base_dir: str) -> Dict[str, Dict[float, List[Dict]]]:
    """Load AMC dataset evaluation data for both baseline and instruct models."""
    amc_data = {}

    # Load baseline model data
    baseline_dir = os.path.join(base_dir, 'amc_baseline')
    baseline_data = defaultdict(list)

    for file_path in glob.glob(os.path.join(baseline_dir, '*_eval.jsonl')):
        file_name = os.path.basename(file_path)

        # Extract temperature using string indexing
        if 'temp' in file_name:
            # Get the index of 'temp' and extract the following two characters
            temp_index = file_name.find('temp')
            temp_value = file_name[temp_index + 4:temp_index + 6]  # Get two digits after 'temp'
            temp = float(temp_value) / 10.0
        else:
            continue

        baseline_data[temp] = load_eval_file(file_path)

    amc_data['baseline'] = baseline_data

    # Load instruct model data
    instruct_dir = os.path.join(base_dir, 'amc_instruct')
    instruct_data = defaultdict(list)

    for file_path in glob.glob(os.path.join(instruct_dir, '*_eval.jsonl')):
        file_name = os.path.basename(file_path)

        # Extract temperature using string indexing
        if 'temp' in file_name:
            # Get the index of 'temp' and extract the following two characters
            temp_index = file_name.find('temp')
            temp_value = file_name[temp_index + 4:temp_index + 6]  # Get two digits after 'temp'
            temp = float(temp_value) / 10.0
        else:
            continue

        instruct_data[temp] = load_eval_file(file_path)

    amc_data['instruct'] = instruct_data

    return amc_data

# %% [markdown]
# Metric Calculation

def calculate_metrics_for_problem(samples: List[Dict]) -> Tuple[int, int]:
    """Calculate total samples and correct samples for a single problem."""
    n = len(samples)
    c = sum(1 for s in samples if s['score'] > 0.0)
    return n, c


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k metric."""
    if n == 0 or c == 0:
        return 0.0

    if k > n:
        k = n

    try:
        pass_at_k = 1.0 - (math.comb(n - c, k) / math.comb(n, k))
    except (ValueError, ZeroDivisionError):
        pass_at_k = 0.0

    return pass_at_k


def calculate_majority_vote(samples: List[Dict]) -> bool:
    """Calculate majority vote result for a single problem."""
    # Extract predictions and scores
    pred_score_list = [(s['extracted_pred'], s['score']) for s in samples]

    # Extract just the predictions for counting
    preds = [item[0] for item in pred_score_list]

    if not preds:
        return False

    # Count votes
    counts = Counter(preds)
    max_count = max(counts.values())
    candidates = [p for p, c in counts.items() if c == max_count]

    # Tie-breaking logic (same as evaluate.py)
    if len(candidates) == 1:
        winner = candidates[0]
    else:
        # Sort to ensure deterministic behavior (convert to str to handle None/numbers)
        candidates.sort(key=lambda x: str(x))
        winner = candidates[0]

    # Check if winner is correct
    for pred, score in pred_score_list:
        if pred == winner:
            return score > 0.0

    return False

# %% [markdown]
# Data Analysis

def analyze_amc_results(amc_data: Dict) -> Dict[str, Dict]:
    """Analyze AMC results for both models and all temperatures."""
    results = defaultdict(lambda: defaultdict(dict))

    for model_type, temp_data in amc_data.items():
        for temp, samples in temp_data.items():
            # Group samples by problem ID
            problem_samples = defaultdict(list)
            for s in samples:
                problem_samples[s['id']].append(s)

            # Calculate metrics for each problem
            pass_k_scores = defaultdict(list)
            maj_vote_correct = 0
            total_problems = len(problem_samples)

            for prob_id, prob_samples in problem_samples.items():
                n, c = calculate_metrics_for_problem(prob_samples)

                # Calculate pass@k for various k values
                for k in [1, 2, 4, 8, 16, 32, 64]:
                    pass_k = calculate_pass_at_k(n, c, k)
                    pass_k_scores[k].append(pass_k)

                # Calculate majority vote
                if calculate_majority_vote(prob_samples):
                    maj_vote_correct += 1

            # Calculate average metrics
            avg_pass_k = {f'pass@{k}': sum(v)/len(v) * 100 for k, v in pass_k_scores.items()}
            avg_maj_vote = (maj_vote_correct / total_problems) * 100

            # Store results
            results[model_type][temp] = {
                **avg_pass_k,
                'maj@1': avg_maj_vote
            }

    return results

# %% [markdown]
# Visualization Functions

def plot_temperature_effects(results: Dict) -> None:
    """Plot how temperature affects model performance."""
    temperatures = sorted(list(results['baseline'].keys()))
    metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'pass@32', 'pass@64', 'maj@1']

    # Calculate grid layout: 2 rows, auto columns based on number of metrics
    num_metrics = len(metrics)
    num_rows = 2
    num_cols = (num_metrics + num_rows - 1) // num_rows  # Ceiling division

    plt.figure(figsize=(24, 12))  # Adjust figure height for 2 rows

    for i, metric in enumerate(metrics, 1):
        plt.subplot(num_rows, num_cols, i)

        # Baseline model
        baseline_values = [results['baseline'][temp][metric] for temp in temperatures]
        plt.plot(temperatures, baseline_values, 'o-', label='Baseline', linewidth=2)

        # Instruct model
        instruct_values = [results['instruct'][temp][metric] for temp in temperatures]
        plt.plot(temperatures, instruct_values, 's--', label='Instruct', linewidth=2)

        plt.title(f'AMC23 - {metric}')
        plt.xlabel('Temperature')
        plt.ylabel('Accuracy (%)')
        plt.xticks(temperatures)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./visualizations/amc_temperature_effects.png', dpi=300, bbox_inches='tight')


def plot_model_comparison(results: Dict) -> None:
    """Compare baseline and instruct models at different temperatures."""
    temperatures = sorted(list(results['baseline'].keys()))
    metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'pass@32', 'pass@64', 'maj@1']

    # Calculate grid layout: 2 rows, auto columns based on number of metrics
    num_metrics = len(metrics)
    num_rows = 2
    num_cols = (num_metrics + num_rows - 1) // num_rows  # Ceiling division

    plt.figure(figsize=(24, 12))  # Adjust figure height for 2 rows

    for i, metric in enumerate(metrics, 1):
        plt.subplot(num_rows, num_cols, i)

        width = 0.35
        x = np.arange(len(temperatures))

        # Baseline model
        baseline_values = [results['baseline'][temp][metric] for temp in temperatures]
        plt.bar(x - width/2, baseline_values, width, label='Baseline', color='#1f77b4')

        # Instruct model
        instruct_values = [results['instruct'][temp][metric] for temp in temperatures]
        plt.bar(x + width/2, instruct_values, width, label='Instruct', color='#ff7f0e')

        plt.title(f'AMC23 - {metric} - Baseline vs Instruct')
        plt.xlabel('Temperature')
        plt.ylabel('Accuracy (%)')
        plt.xticks(x, temperatures)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('./visualizations/amc_model_comparison.png', dpi=300, bbox_inches='tight')


def plot_pass_at_k_curve(results: Dict) -> None:
    """Plot pass@k curves for different models and temperatures."""
    ks = [1, 2, 4, 8, 16, 32, 64]

    plt.figure(figsize=(12, 6))

    # Get all unique temperatures across both models and sort them
    all_temperatures = sorted(set(results['baseline'].keys()) | set(results['instruct'].keys()))

    # Create a color map with unique colors for each temperature
    color_map = plt.get_cmap('tab10', len(all_temperatures))
    temp_to_color = {temp: color_map(i) for i, temp in enumerate(all_temperatures)}

    # Model style mapping: different line styles and markers for different models
    model_styles = {
        'baseline': {'marker': 'o', 'linestyle': '-'},
        'instruct': {'marker': 's', 'linestyle': '--'}
    }

    # Baseline model
    for temp in sorted(results['baseline'].keys()):
        values = [results['baseline'][temp][f'pass@{k}'] for k in ks]
        color = temp_to_color[temp]
        plt.plot(ks, values, marker='o', linestyle='-', color=color, label=f'Baseline (T={temp})')

    # Instruct model
    for temp in sorted(results['instruct'].keys()):
        values = [results['instruct'][temp][f'pass@{k}'] for k in ks]
        color = temp_to_color[temp]
        plt.plot(ks, values, marker='s', linestyle='--', color=color, label=f'Instruct (T={temp})')

    plt.title('pass@k vs k for AMC23 Dataset')
    plt.xlabel('k')
    plt.ylabel('pass@k (%)')
    plt.xscale('log', base=2)
    plt.xticks(ks, ks)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./visualizations/amc_pass_at_k_curve.png', dpi=300, bbox_inches='tight')


def plot_majority_vote_improvement(results: Dict) -> None:
    """Plot the improvement from majority vote compared to pass@1."""
    temperatures = sorted(list(results['baseline'].keys()))

    plt.figure(figsize=(10, 6))

    x = np.arange(len(temperatures))
    width = 0.35

    # Baseline model improvement
    baseline_pass1 = [results['baseline'][temp]['pass@1'] for temp in temperatures]
    baseline_maj1 = [results['baseline'][temp]['maj@1'] for temp in temperatures]
    baseline_improvement = [m - p for p, m in zip(baseline_pass1, baseline_maj1)]

    # Instruct model improvement
    instruct_pass1 = [results['instruct'][temp]['pass@1'] for temp in temperatures]
    instruct_maj1 = [results['instruct'][temp]['maj@1'] for temp in temperatures]
    instruct_improvement = [m - p for p, m in zip(instruct_pass1, instruct_maj1)]

    # Bar plot
    baseline_bars = plt.bar(x - width/2, baseline_improvement, width, label='Baseline', color='#1f77b4')
    instruct_bars = plt.bar(x + width/2, instruct_improvement, width, label='Instruct', color='#ff7f0e')

    # Add improvement values above bars
    for bars, improvements in zip([baseline_bars, instruct_bars], [baseline_improvement, instruct_improvement]):
        for bar, improvement in zip(bars, improvements):
            # For positive improvements, display above the bar
            # For negative improvements, display above 0.0% line
            if improvement > 0:
                height = bar.get_height()
                y_pos = height + 0.1
            else:
                y_pos = 0.1  # Position above 0.0% line

            # Format improvement value
            label = f"{improvement:.2f}%"

            # Position text in the middle of the bar
            x_pos = bar.get_x() + bar.get_width() / 2

            # Add text with ha='center' to center it above the bar
            plt.text(x_pos, y_pos, label, ha='center', va='bottom', fontsize=9)

    # Add zero line
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.title('AMC23 - Majority Vote Improvement over pass@1')
    plt.xlabel('Temperature')
    plt.ylabel('Improvement (%)')
    plt.xticks(x, temperatures)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('./visualizations/amc_majority_vote_improvement.png', dpi=300, bbox_inches='tight')

# %% [markdown]
# Main Analysis

def main():
    # Create output directory for visualizations
    os.makedirs('./visualizations', exist_ok=True)

    # Load AMC data
    base_dir = './res/cleaned'
    print(f"Loading AMC data from {base_dir}...")
    amc_data = load_amc_data(base_dir)

    # Analyze results
    print("Analyzing AMC results...")
    results = analyze_amc_results(amc_data)

    # Display results in a table format
    print("\nAMC Dataset Results:")
    print("=" * 80)

    # Create a pandas DataFrame for better display
    df_results = []
    for model_type in ['baseline', 'instruct']:
        for temp in sorted(results[model_type].keys()):
            row = {
                'Model': model_type,
                'Temperature': temp,
                **results[model_type][temp]
            }
            df_results.append(row)

    df = pd.DataFrame(df_results)
    print(df.to_string(index=False, float_format="%.2f"))

    # Save results to CSV in visualizations folder with specific formatting
    metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'pass@32', 'pass@64', 'maj@1']

    # Create a copy of the DataFrame for CSV export to avoid affecting visualization
    df_csv = df.copy()

    # Format metric columns as strings with exactly 2 decimal places to preserve trailing zeros
    for metric in metrics:
        df_csv[metric] = df_csv[metric].apply(lambda x: '{0:.2f}'.format(x))

    # Export the formatted DataFrame
    df_csv.to_csv('./visualizations/amc_results_summary.csv', index=False)

    # Generate majority vote improvements JSON for cross-validation
    improvement_data = {'amc': []}
    for model_type in results.keys():
        for temp in sorted(results[model_type].keys()):
            pass1 = results[model_type][temp]['pass@1']
            maj1 = results[model_type][temp]['maj@1']
            improvement = maj1 - pass1
            improvement_data['amc'].append({
                'model_type': model_type,
                'temperature': temp,
                'pass@1': float(f"{pass1:.2f}"),
                'maj@1': float(f"{maj1:.2f}"),
                'improvement': float(f"{improvement:.2f}")
            })

    # Save improvement data to JSON file
    with open('./visualizations/amc_majority_vote_improvements.json', 'w', encoding='utf-8') as f:
        json.dump(improvement_data, f, indent=2, ensure_ascii=False)

    print("Saved AMC majority vote improvements to ./visualizations/amc_majority_vote_improvements.json")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Temperature effects
    plot_temperature_effects(results)

    # 2. Model comparison
    plot_model_comparison(results)

    # 3. Pass@k curve
    plot_pass_at_k_curve(results)

    # 4. Majority vote improvement
    plot_majority_vote_improvement(results)

    print("\nAnalysis complete! Results saved in ./visualizations/ directory.")

# %% [markdown]
# Run the Analysis

if __name__ == "__main__":
    main()
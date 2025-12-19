# %% [markdown]
# Data Analysis Script for Qwen Math Model Evaluation

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import math

# Dataset name mapping for formal display
DATASET_NAME_MAP = {
    'amc': 'AMC23',
    'aime': 'AIME25',
    'math500': 'MATH500'
}

# Global temperature range parameter
TEMPERATURE_RANGE = [0.6, 1.0, 1.2]

# Font size settings for plots
PLOT_FONT_SIZE = 14
TITLE_FONT_SIZE = 16

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


def load_all_eval_data(base_dir: str) -> Dict[str, Dict[str, Dict[str, List[Dict]]]]:
    """Load all evaluation data from the specified directory structure.

    Returns a nested dictionary: dataset -> model_type -> temperature -> data
    """
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Define datasets to include
    # included_datasets = ['amc', 'aime']  # Only include amc and aime datasets
    included_datasets = ['amc', 'aime', 'math500']  # Uncomment to include all datasets

    # Get all directories in the base directory
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        # Parse directory name to get dataset and model type
        if '_baseline' in dir_name:
            dataset = dir_name.replace('_baseline', '')
            model_type = 'baseline'
        elif '_instruct' in dir_name:
            dataset = dir_name.replace('_instruct', '')
            model_type = 'instruct'
        else:
            continue  # Skip directories that don't match the pattern

        # Skip datasets not in the included list
        if dataset not in included_datasets:
            continue

        # Get all evaluation files in the directory
        for file_path in glob.glob(os.path.join(dir_path, '*_eval.jsonl')):
            file_name = os.path.basename(file_path)

            # Extract temperature using dynamic string indexing (similar to analyze_amc_example.py)
            if 'temp' in file_name:
                # Get the index of 'temp' and extract the following two characters
                temp_index = file_name.find('temp')
                temp_value = file_name[temp_index + 4:temp_index + 6]  # Get two digits after 'temp'
                try:
                    temperature = float(temp_value) / 10.0
                except (ValueError, IndexError):
                    continue  # Skip files with invalid temperature format
            else:
                continue  # Skip files that don't have temperature in name

            # Only load data for temperatures in the specified range
            if temperature not in TEMPERATURE_RANGE:
                continue

            # Load the data
            data = load_eval_file(file_path)
            all_data[dataset][model_type][temperature] = data

    return all_data

# %% [markdown]
# Metric Calculation Functions

def calculate_pass_at_k(scores: List[float], k: int) -> float:
    """Calculate pass@k for a list of scores.

    Args:
        scores: List of 0.0/1.0 scores for a single problem
        k: Number of samples to consider for pass@k

    Returns:
        pass@k value
    """
    n = len(scores)
    if n == 0:
        return 0.0

    c = sum(scores)
    if c == 0:
        return 0.0

    if k > n:
        k = n

    try:
        pass_at_k = 1.0 - (math.comb(n - int(c), k) / math.comb(n, k))
    except (ValueError, ZeroDivisionError):
        pass_at_k = 0.0

    return pass_at_k


def calculate_majority_vote(data: List[Dict]) -> float:
    """Calculate majority vote accuracy for a dataset.

    Args:
        data: List of samples for all problems

    Returns:
        Majority vote accuracy
    """
    # Group data by problem ID
    problem_groups = defaultdict(list)
    for item in data:
        problem_groups[item['id']].append(item)

    total_problems = len(problem_groups)
    if total_problems == 0:
        return 0.0

    maj_correct = 0

    for problem_id, samples in problem_groups.items():
        # Calculate pass@1 for this problem
        pass1_score = samples[0]['score'] if samples else 0.0

        # Calculate majority vote
        if len(samples) < 2:
            # Not enough samples for majority vote
            if pass1_score > 0.0:
                maj_correct += 1
            continue

        # Extract predictions and scores
        pred_score_list = [(s['extracted_pred'], s['score']) for s in samples]

        # Extract just the predictions for counting
        preds = [item[0] for item in pred_score_list]

        # Count frequencies
        counts = Counter(preds)
        max_freq = max(counts.values())
        candidates = [p for p, c in counts.items() if c == max_freq]

        # Tie-breaking logic (same as evaluate.py and analyze_amc_example.py)
        if len(candidates) == 1:
            winner = candidates[0]
        else:
            # Sort to ensure deterministic behavior (convert to str to handle None/numbers)
            candidates.sort(key=lambda x: str(x))
            winner = candidates[0]

        # Check if winner is correct
        is_correct = False
        for s in samples:
            if s['extracted_pred'] == winner:
                if s['score'] > 0.0:
                    is_correct = True
                break

        if is_correct:
            maj_correct += 1

    maj_acc = maj_correct / total_problems

    return maj_acc


def calculate_all_metrics(data: List[Dict], ks: List[int] = None) -> Dict[str, float]:
    """Calculate all metrics for a dataset.

    Args:
        data: List of all samples
        ks: List of k values for pass@k calculation

    Returns:
        Dictionary of metrics
    """
    # Default k values if not provided
    if ks is None:
        ks = [1, 2, 4, 8, 16, 32, 64]

    # Group scores by problem ID
    problem_scores = defaultdict(list)
    for item in data:
        problem_scores[item['id']].append(item['score'])

    # Calculate pass@k for each problem and average
    metrics = {}
    for k in ks:
        pass_k_values = []
        for scores in problem_scores.values():
            pass_k_values.append(calculate_pass_at_k(scores, k))

        if pass_k_values:
            metrics[f'pass@{k}'] = sum(pass_k_values) / len(pass_k_values)
        else:
            metrics[f'pass@{k}'] = 0.0

    # Calculate majority vote
    maj_acc = calculate_majority_vote(data)
    metrics['maj@1'] = maj_acc

    # Ensure we have the correct pass@1 (already calculated above)
    if 'pass@1' not in metrics:
        metrics['pass@1'] = 0.0

    return metrics

# %% [markdown]
# Data Analysis Functions

def analyze_temperature_effects(all_data: Dict) -> pd.DataFrame:
    """Analyze how temperature affects performance."""
    results = []

    for dataset, models in all_data.items():
        # Ensure baseline comes first in the majority vote analysis output
        model_types = sorted(models.keys(), key=lambda x: 0 if x == 'baseline' else 1)
        for model_type in model_types:
            temps = models[model_type]
            for temp, data in temps.items():
                # Use appropriate k values based on dataset type
                if dataset == 'math500':
                    ks = [1, 2, 4, 8, 16]
                else:  # AMC/AIME
                    ks = [1, 2, 4, 8, 16, 32, 64]
                metrics = calculate_all_metrics(data, ks=ks)

                # Create a result entry with all metrics
                result_entry = {
                    'dataset': dataset,
                    'model_type': model_type,
                    'temperature': temp
                }

                # Add all pass@k metrics
                for k_str, value in metrics.items():
                    if k_str.startswith('pass@'):
                        result_entry[k_str] = value * 100

                # Add maj@1 last
                result_entry['maj@1'] = metrics['maj@1'] * 100

                results.append(result_entry)

    return pd.DataFrame(results)


def analyze_model_comparison(all_data: Dict) -> pd.DataFrame:
    """Compare baseline and instruct models."""
    results = []

    for dataset, models in all_data.items():
        # Get all model types that exist for this dataset, ensuring baseline comes first
        model_types = sorted(list(models.keys()), key=lambda x: 0 if x == 'baseline' else 1)

        # Get all temperatures that exist across any model type for this dataset
        all_temps = set()
        for model_type in model_types:
            all_temps.update(models[model_type].keys())
        all_temps = sorted(all_temps)

        # Process each temperature
        for temp in all_temps:
            # Use appropriate k values based on dataset type
            if dataset == 'math500':
                ks = [1, 2, 4, 8, 16]
            else:  # AMC/AIME
                ks = [1, 2, 4, 8, 16, 32, 64]

            # Process each model type that has data for this temperature
            for model_type in model_types:
                if temp not in models[model_type]:
                    continue

                # Get metrics for this model
                metrics = calculate_all_metrics(models[model_type][temp], ks=ks)

                # Create result entry
                entry = {
                    'dataset': dataset,
                    'temperature': temp,
                    'model_type': model_type
                }

                # Add all pass@k metrics
                for k_str, value in metrics.items():
                    if k_str.startswith('pass@'):
                        entry[k_str] = value * 100

                # Add maj@1 last
                entry['maj@1'] = metrics['maj@1'] * 100

                results.append(entry)

    return pd.DataFrame(results)


def analyze_pass_at_k_vs_k(all_data: Dict) -> Dict[str, pd.DataFrame]:
    """Analyze pass@k as a function of k."""
    results_dict = {}

    for dataset, models in all_data.items():
        results = []

        # Sort model types with baseline first
        model_types = sorted(models.keys(), key=lambda x: 0 if x == 'baseline' else 1)
        for model_type in model_types:
            temps = models[model_type]
            for temp, data in temps.items():
                # Use appropriate k values based on dataset type
                if dataset == 'math500':
                    ks = [1, 2, 4, 8, 16]
                else:  # AMC/AIME
                    ks = [1, 2, 4, 8, 16, 32, 64]
                metrics = calculate_all_metrics(data, ks=ks)

                for k_str, value in metrics.items():
                    if k_str.startswith('pass@'):
                        k = int(k_str.split('@')[1])
                        results.append({
                            'model_type': model_type,
                            'temperature': temp,
                            'k': k,
                            'pass@k': value * 100
                        })

        results_dict[dataset] = pd.DataFrame(results)

    return results_dict


def generate_dataset_summary(all_data: Dict) -> None:
    """Generate dataset-specific summary CSV files in the same format as analyze_amc_example.py."""
    for dataset, models in all_data.items():
        # Create a pandas DataFrame for the summary
        df_results = []

        # Sort model types with baseline first
        model_types = sorted(models.keys(), key=lambda x: 0 if x == 'baseline' else 1)
        for model_type in model_types:
            for temp in sorted(models[model_type].keys()):
                # Get the metrics for this dataset, model, and temperature
                data = models[model_type][temp]
                if dataset == 'math500':
                    ks = [1, 2, 4, 8, 16]
                else:  # AMC/AIME
                    ks = [1, 2, 4, 8, 16, 32, 64]
                metrics = calculate_all_metrics(data, ks=ks)

                # Create a row with all metrics
                row = {
                    'Model': model_type,
                    'Temperature': temp
                }

                # Add all pass@k metrics
                for k_str, value in metrics.items():
                    if k_str.startswith('pass@'):
                        row[k_str] = value * 100

                # Add maj@1 last
                row['maj@1'] = metrics['maj@1'] * 100

                df_results.append(row)

        # Create DataFrame
        df = pd.DataFrame(df_results)

        # Get all metric columns
        metrics = ['maj@1'] + [col for col in df.columns if col.startswith('pass@')]

        # Create a copy for CSV export with formatted metrics
        df_csv = df.copy()

        # Format metric columns as strings with exactly 2 decimal places to preserve trailing zeros
        for metric in metrics:
            df_csv[metric] = df_csv[metric].apply(lambda x: '{0:.2f}'.format(x))

        # Export the formatted DataFrame
        df_csv.to_csv(f'./visualizations/{dataset}_results_summary.csv', index=False)
        print(f"Saved {dataset} results summary to ./visualizations/{dataset}_results_summary.csv")


def analyze_majority_vote_effect(all_data: Dict) -> pd.DataFrame:
    """Analyze the effectiveness of majority vote compared to pass@1."""
    results = []
    improvement_data = {}

    for dataset, models in all_data.items():
        dataset_improvements = []

        # Sort model types with baseline first
        model_types = sorted(models.keys(), key=lambda x: 0 if x == 'baseline' else 1)
        for model_type in model_types:
            temps = models[model_type]
            for temp, data in temps.items():
                # Use appropriate k values based on dataset type
                if dataset == 'math500':
                    ks = [1, 2, 4, 8, 16]
                else:  # AMC/AIME
                    ks = [1, 2, 4, 8, 16, 32, 64]
                metrics = calculate_all_metrics(data, ks=ks)

                improvement = metrics['maj@1'] - metrics['pass@1']

                result_entry = {
                    'dataset': dataset,
                    'model_type': model_type,
                    'temperature': temp,
                    'pass@1': metrics['pass@1'] * 100,
                    'improvement': improvement * 100,
                    'maj@1': metrics['maj@1'] * 100
                }

                results.append(result_entry)

                # Add to improvement data for JSON export
                dataset_improvements.append({
                    'model_type': model_type,
                    'temperature': temp,
                    'pass@1': float(f"{metrics['pass@1'] * 100:.2f}"),
                    'maj@1': float(f"{metrics['maj@1'] * 100:.2f}"),
                    'improvement': float(f"{improvement * 100:.2f}")
                })

        # Store improvements by dataset
        improvement_data[dataset] = dataset_improvements

    # Save improvement data to JSON file
    with open('./visualizations/majority_vote_improvements.json', 'w', encoding='utf-8') as f:
        json.dump(improvement_data, f, indent=2, ensure_ascii=False)

    print("Saved majority vote improvements to ./visualizations/majority_vote_improvements.json")

    return pd.DataFrame(results)

# %% [markdown]
# Visualization Functions

def plot_temperature_effects(df: pd.DataFrame) -> None:
    """Plot the effects of temperature on model performance."""
    datasets = df['dataset'].unique()

    # Set global font size
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    for dataset in datasets:
        # Select metrics based on dataset type
        if dataset == 'math500':
            metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'maj@1']
        else:  # amc/aime
            metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'pass@32', 'pass@64', 'maj@1']

        # Calculate grid layout: 2 rows, auto columns based on number of metrics
        num_metrics = len(metrics)
        num_rows = 2
        num_cols = (num_metrics + num_rows - 1) // num_rows  # Ceiling division

        # Set appropriate figure size based on number of columns
        if num_cols <= 3:
            figsize = (18, 10)
        else:
            figsize = (24, 12)

        # Plot with title for PNG
        plt.figure(figsize=figsize)

        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_rows, num_cols, i)

            # Only show model types that have actual data, ensuring baseline comes first
            model_types = sorted(df[(df['dataset'] == dataset)]['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)
            colors = ['#1f77b4', '#ff7f0e']  # blue for baseline, orange for instruct
            for j, model_type in enumerate(model_types):
                model_data = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)]
                # Sort data by temperature to ensure proper ordering
                model_data = model_data.sort_values('temperature')
                plt.plot(model_data['temperature'], model_data[metric], marker='o', label=model_type.capitalize(), color=colors[j % len(colors)])

            plt.title(f'{DATASET_NAME_MAP.get(dataset, dataset)} - {metric}', fontsize=TITLE_FONT_SIZE)
            plt.xlabel('Temperature')
            plt.ylabel(f'{metric} (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_temperature_effects.png', dpi=300, bbox_inches='tight')
        # plt.savefig(f'./visualizations/{dataset}_temperature_effects.pdf', bbox_inches='tight')  # Commented out: PDF with title
        plt.close()

        # Plot without title for PDF (for report integration)
        plt.figure(figsize=figsize)

        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_rows, num_cols, i)

            # Only show model types that have actual data, ensuring baseline comes first
            model_types = sorted(df[(df['dataset'] == dataset)]['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)
            colors = ['#1f77b4', '#ff7f0e']  # blue for baseline, orange for instruct
            for j, model_type in enumerate(model_types):
                model_data = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)]
                # Sort data by temperature to ensure proper ordering
                model_data = model_data.sort_values('temperature')
                plt.plot(model_data['temperature'], model_data[metric], marker='o', label=model_type.capitalize(), color=colors[j % len(colors)])

            # No title for PDF version
            plt.xlabel('Temperature')
            plt.ylabel(f'{metric} (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_temperature_effects.pdf', bbox_inches='tight')  # Removed _no_title suffix
        plt.close()


def plot_model_comparison(df: pd.DataFrame) -> None:
    """Plot comparison between baseline and instruct models."""
    datasets = df['dataset'].unique()

    # Set global font size
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    for dataset in datasets:
        # Get temperatures that exist for this dataset
        temperatures = sorted(df[(df['dataset'] == dataset)]['temperature'].unique())

        # Select metrics based on dataset type
        if dataset == 'math500':
            metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'maj@1']
        else:  # amc/aime
            metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'pass@32', 'pass@64', 'maj@1']

        # Calculate grid layout: 2 rows, auto columns based on number of metrics
        num_metrics = len(metrics)
        num_rows = 2
        num_cols = (num_metrics + num_rows - 1) // num_rows  # Ceiling division

        # Set appropriate figure size based on number of columns (compressed width)
        if num_cols <= 3:
            figsize = (14, 8)  # Reduced width from 18 to 14
        else:
            figsize = (20, 10)  # Reduced width from 24 to 20

        # Plot with title for PNG
        plt.figure(figsize=figsize)

        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_rows, num_cols, i)

            x = np.arange(len(temperatures))

            # Only show model types that have actual data, ensuring baseline comes first
            dataset_models = sorted(df[(df['dataset'] == dataset)]['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)
            colors = ['#1f77b4', '#ff7f0e']  # blue for baseline, orange for instruct
            width = 0.35 / len(dataset_models)

            for j, model_type in enumerate(dataset_models):
                # Filter data for this dataset, model type, and metric, sorted by temperature
                model_data = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)].sort_values('temperature')[metric]
                plt.bar(x + (j - len(dataset_models)/2 + 0.5) * width, model_data, width, label=model_type.capitalize(), color=colors[j % len(colors)])

            plt.title(f'{DATASET_NAME_MAP.get(dataset, dataset)} - {metric}', fontsize=TITLE_FONT_SIZE)
            plt.xlabel('Temperature')
            plt.ylabel(f'{metric} (%)')
            plt.xticks(x, temperatures)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_model_comparison.png', dpi=300, bbox_inches='tight')
        # plt.savefig(f'./visualizations/{dataset}_model_comparison.pdf', bbox_inches='tight')  # Commented out: PDF with title
        plt.close()

        # Plot without title for PDF (for report integration)
        plt.figure(figsize=figsize)

        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_rows, num_cols, i)

            x = np.arange(len(temperatures))

            # Only show model types that have actual data, ensuring baseline comes first
            dataset_models = sorted(df[(df['dataset'] == dataset)]['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)
            colors = ['#1f77b4', '#ff7f0e']  # blue for baseline, orange for instruct
            width = 0.35 / len(dataset_models)

            for j, model_type in enumerate(dataset_models):
                # Filter data for this dataset, model type, and metric, sorted by temperature
                model_data = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)].sort_values('temperature')[metric]
                plt.bar(x + (j - len(dataset_models)/2 + 0.5) * width, model_data, width, label=model_type.capitalize(), color=colors[j % len(colors)])

            # No title for PDF version
            plt.xlabel('Temperature')
            plt.ylabel(f'{metric} (%)')
            plt.xticks(x, temperatures)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_model_comparison.pdf', bbox_inches='tight')  # Removed _no_title suffix
        plt.close()


def plot_pass_at_k_vs_k(results_dict: Dict[str, pd.DataFrame]) -> None:
    """Plot pass@k as a function of k for different datasets and models."""
    # Set global font size
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    for dataset, df in results_dict.items():
        # Plot with title for PNG
        plt.figure(figsize=(12, 6))

        # Get all unique temperatures across both models and sort them
        unique_temperatures = sorted(df['temperature'].unique())

        # Create a color map with unique colors for each temperature
        color_map = plt.get_cmap('tab10', len(unique_temperatures))
        temp_to_color = {temp: color_map(i) for i, temp in enumerate(unique_temperatures)}

        # Model style mapping: different line styles and markers for different models
        model_styles = {
            'baseline': {'marker': 'o', 'linestyle': '-'},
            'instruct': {'marker': 's', 'linestyle': '--'}
        }

        # Ensure baseline comes first in the legend and plot ordering
        model_types = sorted(df['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)

        for model_type in model_types:
            model_data = df[df['model_type'] == model_type]
            style = model_styles[model_type]

            for temp in sorted(model_data['temperature'].unique()):
                temp_data = model_data[model_data['temperature'] == temp]
                color = temp_to_color[temp]

                plt.plot(temp_data['k'], temp_data['pass@k'],
                         marker=style['marker'], linestyle=style['linestyle'],
                         color=color, label=f'{model_type.capitalize()} (T={temp})')

        plt.title(f'pass@k vs k for {DATASET_NAME_MAP.get(dataset, dataset)} Dataset', fontsize=TITLE_FONT_SIZE)
        plt.xlabel('k')
        plt.ylabel('pass@k (%)')
        plt.xscale('log', base=2)
        plt.xticks(sorted(df['k'].unique()), sorted(df['k'].unique()))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_pass_at_k_curve.png', dpi=300, bbox_inches='tight')
        # plt.savefig(f'./visualizations/{dataset}_pass_at_k_curve.pdf', bbox_inches='tight')  # Commented out: PDF with title
        plt.close()

        # Plot without title for PDF (for report integration)
        plt.figure(figsize=(12, 6))

        # Get all unique temperatures across both models and sort them
        unique_temperatures = sorted(df['temperature'].unique())

        # Create a color map with unique colors for each temperature
        color_map = plt.get_cmap('tab10', len(unique_temperatures))
        temp_to_color = {temp: color_map(i) for i, temp in enumerate(unique_temperatures)}

        # Model style mapping: different line styles and markers for different models
        model_styles = {
            'baseline': {'marker': 'o', 'linestyle': '-'},
            'instruct': {'marker': 's', 'linestyle': '--'}
        }

        # Ensure baseline comes first in the legend and plot ordering
        model_types = sorted(df['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)

        for model_type in model_types:
            model_data = df[df['model_type'] == model_type]
            style = model_styles[model_type]

            for temp in sorted(model_data['temperature'].unique()):
                temp_data = model_data[model_data['temperature'] == temp]
                color = temp_to_color[temp]

                plt.plot(temp_data['k'], temp_data['pass@k'],
                         marker=style['marker'], linestyle=style['linestyle'],
                         color=color, label=f'{model_type.capitalize()} (T={temp})')

        # No title for PDF version
        plt.xlabel('k')
        plt.ylabel('pass@k (%)')
        plt.xscale('log', base=2)
        plt.xticks(sorted(df['k'].unique()), sorted(df['k'].unique()))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_pass_at_k_curve.pdf', bbox_inches='tight')  # Removed _no_title suffix
        plt.close()


def plot_majority_vote_improvement(df: pd.DataFrame) -> None:
    """Plot the improvement from majority vote compared to pass@1."""
    datasets = df['dataset'].unique()

    # Set global font size
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    for dataset in datasets:
        # Plot with title for PNG
        plt.figure(figsize=(8, 6))  # Reduced width from 10 to 8

        # Get sorted list of temperatures
        temperatures = sorted(df[(df['dataset'] == dataset)]['temperature'].unique())

        x = np.arange(len(temperatures))
        width = 0.35

        # Get all model types, ensuring baseline comes first
        model_types = sorted(df[(df['dataset'] == dataset)]['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)
        colors = ['#1f77b4', '#ff7f0e']  # blue for baseline, orange for instruct

        for i, model_type in enumerate(model_types):
            model_data = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)]
            # Sort data by temperature
            model_data = model_data.sort_values('temperature')

            # Calculate improvement values
            improvements = model_data['improvement'].values

            # Create bars for improvements
            bars = plt.bar(x + (i - len(model_types)/2 + 0.5) * width,
                          improvements,
                          width,
                          label=model_type.capitalize(),
                          color=colors[i % len(colors)])

            # Add improvement values above bars
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
                plt.text(x_pos, y_pos, label, ha='center', va='bottom', fontsize=PLOT_FONT_SIZE-2)

        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.title(f'{DATASET_NAME_MAP.get(dataset, dataset)} - Majority Vote Improvement over pass@1', fontsize=TITLE_FONT_SIZE)
        plt.xlabel('Temperature')
        plt.ylabel('Improvement (%)')
        plt.xticks(x, temperatures)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_majority_vote_improvement.png', dpi=300, bbox_inches='tight')
        # plt.savefig(f'./visualizations/{dataset}_majority_vote_improvement.pdf', bbox_inches='tight')  # Commented out: PDF with title
        plt.close()

        # Plot without title for PDF (for report integration)
        plt.figure(figsize=(8, 6))  # Reduced width from 10 to 8

        # Get sorted list of temperatures
        temperatures = sorted(df[(df['dataset'] == dataset)]['temperature'].unique())

        x = np.arange(len(temperatures))
        width = 0.35

        # Get all model types, ensuring baseline comes first
        model_types = sorted(df[(df['dataset'] == dataset)]['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)
        colors = ['#1f77b4', '#ff7f0e']  # blue for baseline, orange for instruct

        for i, model_type in enumerate(model_types):
            model_data = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)]
            # Sort data by temperature
            model_data = model_data.sort_values('temperature')

            # Calculate improvement values
            improvements = model_data['improvement'].values

            # Create bars for improvements
            bars = plt.bar(x + (i - len(model_types)/2 + 0.5) * width,
                          improvements,
                          width,
                          label=model_type.capitalize(),
                          color=colors[i % len(colors)])

            # Add improvement values above bars
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
                plt.text(x_pos, y_pos, label, ha='center', va='bottom', fontsize=PLOT_FONT_SIZE-2)

        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # No title for PDF version
        plt.xlabel('Temperature')
        plt.ylabel('Improvement (%)')
        plt.xticks(x, temperatures)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_majority_vote_improvement.pdf', bbox_inches='tight')  # Removed _no_title suffix
        plt.close()

# %% [markdown]
# Main Analysis Function

def main():
    # Create output directory for visualizations
    os.makedirs('./visualizations', exist_ok=True)

    # Load all evaluation data
    base_dir = './res/cleaned'
    print(f"Loading data from {base_dir}...")
    all_data = load_all_eval_data(base_dir)

    # Print summary of loaded data
    print("\nLoaded data summary:")
    for dataset, models in all_data.items():
        print(f"Dataset: {dataset}")
        for model_type, temps in models.items():
            print(f"  Model: {model_type}, Temperatures: {list(temps.keys())}")
            for temp, data in temps.items():
                print(f"    Temp {temp}: {len(data)} samples across {len(set(d['id'] for d in data))} problems")

    # 1. Analyze temperature effects
    print("\n1. Analyzing temperature effects...")
    temp_df = analyze_temperature_effects(all_data)
    print(temp_df)

    # Create a copy for CSV export with formatted metrics
    temp_df_csv = temp_df.copy()
    # Format metric columns with 2 decimal places
    for col in temp_df_csv.columns:
        if col.startswith('pass@') or col == 'maj@1':
            temp_df_csv[col] = temp_df_csv[col].apply(lambda x: '{0:.2f}'.format(x))
    temp_df_csv.to_csv('./visualizations/temperature_effects.csv', index=False)

    # 2. Analyze model comparison
    print("\n2. Analyzing model comparison...")
    model_df = analyze_model_comparison(all_data)
    print(model_df)

    # Create a copy for CSV export with formatted metrics
    model_df_csv = model_df.copy()
    # Format metric columns with 2 decimal places
    for col in model_df_csv.columns:
        if col.startswith('pass@') or col == 'maj@1':
            model_df_csv[col] = model_df_csv[col].apply(lambda x: '{0:.2f}'.format(x))
    model_df_csv.to_csv('./visualizations/model_comparison.csv', index=False)

    # 3. Analyze pass@k vs k
    print("\n3. Analyzing pass@k vs k...")
    passk_df_dict = analyze_pass_at_k_vs_k(all_data)
    for dataset, df in passk_df_dict.items():
        print(f"\nDataset: {dataset}")
        print(df)

    # 4. Generate dataset summary CSV files
    print("\n4. Generating dataset summary files...")
    generate_dataset_summary(all_data)

    # 5. Analyze majority vote effect
    print("\n5. Analyzing majority vote effect...")
    maj_df = analyze_majority_vote_effect(all_data)
    print(maj_df)

    # Create a copy for CSV export with formatted metrics
    maj_df_csv = maj_df.copy()
    # Format metric columns with 2 decimal places
    for col in maj_df_csv.columns:
        if col.startswith('pass@') or col == 'maj@1' or col == 'improvement':
            maj_df_csv[col] = maj_df_csv[col].apply(lambda x: '{0:.2f}'.format(x))
    maj_df_csv.to_csv('./visualizations/majority_vote_effect.csv', index=False)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Temperature effects plots
    plot_temperature_effects(temp_df)

    # Model comparison plots
    plot_model_comparison(model_df)

    # Pass@k vs k plots
    plot_pass_at_k_vs_k(passk_df_dict)

    # Majority vote improvement plots
    plot_majority_vote_improvement(maj_df)

    print("\nAnalysis complete! Results and visualizations saved in ./visualizations/")

# %% [markdown]
# Run the Analysis

if __name__ == "__main__":
    main()
import os
import json
import glob
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, DefaultDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset name mapping for formal display
DATASET_NAME_MAP: Dict[str, str] = {
    'amc': 'AMC23',
    'aime': 'AIME25',
    'math500': 'MATH500'
}

# Global temperature range parameter
TEMPERATURE_RANGE: List[float] = [0.6, 0.8, 1.0, 1.2]

# Font size settings for plots
PLOT_FONT_SIZE: int = 14
TITLE_FONT_SIZE: int = 16

def load_eval_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL evaluation file and return a list of dictionaries."""
    data: List[Dict[str, Any]] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding line in {file_path}")
    return data


def load_all_eval_data(base_dir: str) -> Dict[str, Any]:
    """Load all evaluation data from the specified directory structure."""
    all_data: Dict[str, Any] = {}
    temp_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    included_datasets: List[str] = ['amc', 'aime', 'math500']

    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        if '_baseline' in dir_name:
            dataset = dir_name.replace('_baseline', '')
            model_type = 'baseline'
        elif '_instruct' in dir_name:
            dataset = dir_name.replace('_instruct', '')
            model_type = 'instruct'
        else:
            continue

        if dataset not in included_datasets:
            continue

        for file_path in glob.glob(os.path.join(dir_path, '*_eval.jsonl')):
            file_name = os.path.basename(file_path)

            if 'temp' in file_name:
                temp_index = file_name.find('temp')
                temp_value = file_name[temp_index + 4:temp_index + 6]
                try:
                    temperature = float(temp_value) / 10.0
                except (ValueError, IndexError):
                    continue
            else:
                continue

            if temperature not in TEMPERATURE_RANGE:
                continue

            data = load_eval_file(file_path)
            all_data[dataset][model_type][temperature] = data

    return all_data

def calculate_pass_at_k(scores: List[float], k: int) -> float:
    """Calculate pass@k for a list of scores."""
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


def calculate_majority_vote(data: List[Dict[str, Any]]) -> float:
    """Calculate majority vote accuracy for a dataset."""
    problem_groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in data:
        problem_groups[item['id']].append(item)

    total_problems = len(problem_groups)
    if total_problems == 0:
        return 0.0

    maj_correct = 0

    for problem_id, samples in problem_groups.items():
        pass1_score = samples[0]['score'] if samples else 0.0

        if len(samples) < 2:
            if pass1_score > 0.0:
                maj_correct += 1
            continue

        pred_score_list = [(s['extracted_pred'], s['score']) for s in samples]
        preds = [item[0] for item in pred_score_list]

        counts = Counter(preds)
        max_freq = max(counts.values())
        candidates = [p for p, c in counts.items() if c == max_freq]

        if len(candidates) == 1:
            winner = candidates[0]
        else:
            candidates.sort(key=lambda x: str(x))
            winner = candidates[0]

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


def calculate_all_metrics(data: List[Dict[str, Any]], ks: List[int] = None) -> Dict[str, float]:
    """Calculate all metrics for a dataset."""
    if ks is None:
        ks = [1, 2, 4, 8, 16, 32, 64]

    problem_scores: DefaultDict[str, List[float]] = defaultdict(list)
    for item in data:
        problem_scores[item['id']].append(item['score'])

    metrics: Dict[str, float] = {}
    for k in ks:
        pass_k_values: List[float] = []
        for scores in problem_scores.values():
            pass_k_values.append(calculate_pass_at_k(scores, k))

        if pass_k_values:
            metrics[f'pass@{k}'] = sum(pass_k_values) / len(pass_k_values)
        else:
            metrics[f'pass@{k}'] = 0.0

    maj_acc = calculate_majority_vote(data)
    metrics['maj@1'] = maj_acc

    if 'pass@1' not in metrics:
        metrics['pass@1'] = 0.0

    return metrics

def analyze_temperature_effects(all_data: Dict[str, Any]) -> pd.DataFrame:
    """Analyze how temperature affects performance."""
    results: List[Dict[str, Any]] = []

    for dataset, models in all_data.items():
        model_types = sorted(models.keys(), key=lambda x: 0 if x == 'baseline' else 1)
        for model_type in model_types:
            temps = models[model_type]
            for temp, data in temps.items():
                if dataset == 'math500':
                    ks = [1, 2, 4, 8, 16]
                else:
                    ks = [1, 2, 4, 8, 16, 32, 64]
                metrics = calculate_all_metrics(data, ks=ks)

                result_entry: Dict[str, Any] = {
                    'dataset': dataset,
                    'model_type': model_type,
                    'temperature': temp
                }

                for k_str, value in metrics.items():
                    if k_str.startswith('pass@'):
                        result_entry[k_str] = value * 100

                result_entry['maj@1'] = metrics['maj@1'] * 100
                results.append(result_entry)

    return pd.DataFrame(results)


def analyze_model_comparison(all_data: Dict[str, Any]) -> pd.DataFrame:
    """Compare baseline and instruct models."""
    results: List[Dict[str, Any]] = []

    for dataset, models in all_data.items():
        model_types = sorted(list(models.keys()), key=lambda x: 0 if x == 'baseline' else 1)

        all_temps = set()
        for model_type in model_types:
            all_temps.update(models[model_type].keys())
        all_temps = sorted(all_temps)

        for temp in all_temps:
            if dataset == 'math500':
                ks = [1, 2, 4, 8, 16]
            else:
                ks = [1, 2, 4, 8, 16, 32, 64]

            for model_type in model_types:
                if temp not in models[model_type]:
                    continue

                metrics = calculate_all_metrics(models[model_type][temp], ks=ks)

                entry: Dict[str, Any] = {
                    'dataset': dataset,
                    'temperature': temp,
                    'model_type': model_type
                }

                for k_str, value in metrics.items():
                    if k_str.startswith('pass@'):
                        entry[k_str] = value * 100

                entry['maj@1'] = metrics['maj@1'] * 100
                results.append(entry)

    return pd.DataFrame(results)


def analyze_pass_at_k_vs_k(all_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Analyze pass@k as a function of k."""
    results_dict: Dict[str, pd.DataFrame] = {}

    for dataset, models in all_data.items():
        results: List[Dict[str, Any]] = []

        model_types = sorted(models.keys(), key=lambda x: 0 if x == 'baseline' else 1)
        for model_type in model_types:
            temps = models[model_type]
            for temp, data in temps.items():
                if dataset == 'math500':
                    ks = [1, 2, 4, 8, 16]
                else:
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


def generate_dataset_summary(all_data: Dict[str, Any]) -> None:
    """Generate dataset-specific summary CSV files."""
    for dataset, models in all_data.items():
        df_results: List[Dict[str, Any]] = []

        model_types = sorted(models.keys(), key=lambda x: 0 if x == 'baseline' else 1)
        for model_type in model_types:
            for temp in sorted(models[model_type].keys()):
                data = models[model_type][temp]
                if dataset == 'math500':
                    ks = [1, 2, 4, 8, 16]
                else:
                    ks = [1, 2, 4, 8, 16, 32, 64]
                metrics = calculate_all_metrics(data, ks=ks)

                row: Dict[str, Any] = {
                    'Model': model_type,
                    'Temperature': temp
                }

                for k_str, value in metrics.items():
                    if k_str.startswith('pass@'):
                        row[k_str] = value * 100

                row['maj@1'] = metrics['maj@1'] * 100
                df_results.append(row)

        df = pd.DataFrame(df_results)
        metrics_cols = ['maj@1'] + [col for col in df.columns if col.startswith('pass@')]

        df_csv = df.copy()
        for metric in metrics_cols:
            df_csv[metric] = df_csv[metric].apply(lambda x: '{0:.2f}'.format(x))

        df_csv.to_csv(f'./visualizations/{dataset}_results_summary.csv', index=False)
        print(f"Saved {dataset} results summary to ./visualizations/{dataset}_results_summary.csv")


def analyze_majority_vote_effect(all_data: Dict[str, Any]) -> pd.DataFrame:
    """Analyze the effectiveness of majority vote compared to pass@1."""
    results: List[Dict[str, Any]] = []
    improvement_data: Dict[str, Any] = {}

    for dataset, models in all_data.items():
        dataset_improvements: List[Dict[str, Any]] = []

        model_types = sorted(models.keys(), key=lambda x: 0 if x == 'baseline' else 1)
        for model_type in model_types:
            temps = models[model_type]
            for temp, data in temps.items():
                if dataset == 'math500':
                    ks = [1, 2, 4, 8, 16]
                else:
                    ks = [1, 2, 4, 8, 16, 32, 64]
                metrics = calculate_all_metrics(data, ks=ks)

                improvement = metrics['maj@1'] - metrics['pass@1']

                result_entry: Dict[str, Any] = {
                    'dataset': dataset,
                    'model_type': model_type,
                    'temperature': temp,
                    'pass@1': metrics['pass@1'] * 100,
                    'improvement': improvement * 100,
                    'maj@1': metrics['maj@1'] * 100
                }

                results.append(result_entry)

                dataset_improvements.append({
                    'model_type': model_type,
                    'temperature': temp,
                    'pass@1': float(f"{metrics['pass@1'] * 100:.2f}"),
                    'maj@1': float(f"{metrics['maj@1'] * 100:.2f}"),
                    'improvement': float(f"{improvement * 100:.2f}")
                })

        improvement_data[dataset] = dataset_improvements

    with open('./visualizations/majority_vote_improvements.json', 'w', encoding='utf-8') as f:
        json.dump(improvement_data, f, indent=2, ensure_ascii=False)

    print("Saved majority vote improvements to ./visualizations/majority_vote_improvements.json")
    return pd.DataFrame(results)

def plot_temperature_effects(df: pd.DataFrame) -> None:
    """Plot the effects of temperature on model performance."""
    datasets = df['dataset'].unique()

    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    for dataset in datasets:
        if dataset == 'math500':
            metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'maj@1']
        else:
            metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'pass@32', 'pass@64', 'maj@1']

        num_metrics = len(metrics)
        num_rows = 2
        num_cols = (num_metrics + num_rows - 1) // num_rows

        if num_cols <= 3:
            figsize = (18, 10)
        else:
            figsize = (24, 12)

        plt.figure(figsize=figsize)

        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_rows, num_cols, i)

            model_types = sorted(df[(df['dataset'] == dataset)]['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)
            colors = ['#1f77b4', '#ff7f0e']
            for j, model_type in enumerate(model_types):
                model_data = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)]
                model_data = model_data.sort_values('temperature')
                plt.plot(model_data['temperature'], model_data[metric], marker='o', label=model_type.capitalize(), color=colors[j % len(colors)])

            plt.title(f'{DATASET_NAME_MAP.get(dataset, dataset)} - {metric}', fontsize=TITLE_FONT_SIZE)
            plt.xlabel('Temperature')
            plt.ylabel(f'{metric} (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_temperature_effects.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=figsize)

        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_rows, num_cols, i)

            model_types = sorted(df[(df['dataset'] == dataset)]['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)
            colors = ['#1f77b4', '#ff7f0e']
            for j, model_type in enumerate(model_types):
                model_data = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)]
                model_data = model_data.sort_values('temperature')
                plt.plot(model_data['temperature'], model_data[metric], marker='o', label=model_type.capitalize(), color=colors[j % len(colors)])

            plt.xlabel('Temperature')
            plt.ylabel(f'{metric} (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_temperature_effects.pdf', bbox_inches='tight')
        plt.close()


def plot_model_comparison(df: pd.DataFrame) -> None:
    """Plot comparison between baseline and instruct models."""
    datasets = df['dataset'].unique()

    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    for dataset in datasets:
        temperatures = sorted(df[(df['dataset'] == dataset)]['temperature'].unique())

        if dataset == 'math500':
            metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'maj@1']
        else:
            metrics = ['pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'pass@32', 'pass@64', 'maj@1']

        num_metrics = len(metrics)
        num_rows = 2
        num_cols = (num_metrics + num_rows - 1) // num_rows

        if num_cols <= 3:
            figsize = (14, 8)
        else:
            figsize = (20, 10)

        plt.figure(figsize=figsize)

        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_rows, num_cols, i)

            x = np.arange(len(temperatures))

            dataset_models = sorted(df[(df['dataset'] == dataset)]['model_type'].unique(), key=lambda x: 0 if x == 'baseline' else 1)
            colors = ['#1f77b4', '#ff7f0e']
            width = 0.35 / len(dataset_models)

            for j, model_type in enumerate(dataset_models):
                model_data = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)].sort_values('temperature')[metric]
                plt.bar(x + (j - len(dataset_models)/2 + 0.5) * width, model_data, width, label=model_type.capitalize(), color=colors[j % len(colors)])

            if dataset == 'aime':
                plt.ylim(0, 50)
            elif dataset == 'amc':
                plt.ylim(0, 100)  # AMC: 8 subplots to ymax=100%
            elif dataset == 'math500':
                plt.ylim(0, 100)  # MATH500: 6 subplots to ymax=100%

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

            # Set y-axis limits based on dataset type
            if dataset == 'aime':
                plt.ylim(0, 50)  # AIME: 8 subplots to ymax=50%
            elif dataset == 'amc':
                plt.ylim(0, 100)  # AMC: 8 subplots to ymax=100%
            elif dataset == 'math500':
                plt.ylim(0, 100)  # MATH500: 6 subplots to ymax=100%

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

        # Set y-axis limits based on dataset type
        if dataset == 'aime':
            plt.ylim(0, 50)  # AIME: ymax=50%
        elif dataset == 'amc':
            plt.ylim(0, 100)  # AMC: ymax=100%
        elif dataset == 'math500':
            plt.ylim(0, 100)  # MATH500: ymax=100%

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

        # Set y-axis limits based on dataset type
        if dataset == 'aime':
            plt.ylim(0, 50)  # AIME: ymax=50%
        elif dataset == 'amc':
            plt.ylim(0, 100)  # AMC: ymax=100%
        elif dataset == 'math500':
            plt.ylim(0, 100)  # MATH500: ymax=100%

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

        # Calculate y-axis limits that extend to next even number
        all_improvements = df[(df['dataset'] == dataset)]['improvement'].values
        y_min = min(all_improvements) if len(all_improvements) > 0 else 0
        y_max = max(all_improvements) if len(all_improvements) > 0 else 0

        # Extend y-axis limits to next even number
        y_min_extended = np.floor(y_min / 2) * 2 if y_min < 0 else 0
        y_max_extended = np.ceil(y_max / 2) * 2 if y_max > 0 else 0

        # Ensure we have a reasonable range if values are very small
        if y_max_extended - y_min_extended < 2:
            y_max_extended = y_min_extended + 2

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

        # Set y-axis limits and ensure tick labels are visible
        plt.ylim(y_min_extended, y_max_extended)

        # Generate y-axis ticks with step of 2 and ensure they are visible
        y_ticks = np.arange(y_min_extended, y_max_extended + 2, 2)
        plt.yticks(y_ticks, [f"{tick:.0f}%" for tick in y_ticks])

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

        # Calculate y-axis limits that extend to next even number
        all_improvements = df[(df['dataset'] == dataset)]['improvement'].values
        y_min = min(all_improvements) if len(all_improvements) > 0 else 0
        y_max = max(all_improvements) if len(all_improvements) > 0 else 0

        # Extend y-axis limits to next even number
        y_min_extended = np.floor(y_min / 2) * 2 if y_min < 0 else 0
        y_max_extended = np.ceil(y_max / 2) * 2 if y_max > 0 else 0

        # Ensure we have a reasonable range if values are very small
        if y_max_extended - y_min_extended < 2:
            y_max_extended = y_min_extended + 2

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

        # Set y-axis limits and ensure tick labels are visible
        plt.ylim(y_min_extended, y_max_extended)

        # Generate y-axis ticks with step of 2 and ensure they are visible
        y_ticks = np.arange(y_min_extended, y_max_extended + 2, 2)
        plt.yticks(y_ticks, [f"{tick:.0f}%" for tick in y_ticks])

        # No title for PDF version
        plt.xlabel('Temperature')
        plt.ylabel('Improvement (%)')
        plt.xticks(x, temperatures)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'./visualizations/{dataset}_majority_vote_improvement.pdf', bbox_inches='tight')  # Removed _no_title suffix
        plt.close()

def main() -> None:
    """Main analysis function."""
    os.makedirs('./visualizations', exist_ok=True)

    base_dir = './res/cleaned'
    print(f"Loading data from {base_dir}...")
    all_data = load_all_eval_data(base_dir)

    print("\nLoaded data summary:")
    for dataset, models in all_data.items():
        print(f"Dataset: {dataset}")
        for model_type, temps in models.items():
            print(f"  Model: {model_type}, Temperatures: {list(temps.keys())}")
            for temp, data in temps.items():
                print(f"    Temp {temp}: {len(data)} samples across {len(set(d['id'] for d in data))} problems")

    print("\n1. Analyzing temperature effects...")
    temp_df = analyze_temperature_effects(all_data)
    print(temp_df)

    temp_df_csv = temp_df.copy()
    for col in temp_df_csv.columns:
        if col.startswith('pass@') or col == 'maj@1':
            temp_df_csv[col] = temp_df_csv[col].apply(lambda x: '{0:.2f}'.format(x))
    temp_df_csv.to_csv('./visualizations/temperature_effects.csv', index=False)

    print("\n2. Analyzing model comparison...")
    model_df = analyze_model_comparison(all_data)
    print(model_df)

    model_df_csv = model_df.copy()
    for col in model_df_csv.columns:
        if col.startswith('pass@') or col == 'maj@1':
            model_df_csv[col] = model_df_csv[col].apply(lambda x: '{0:.2f}'.format(x))
    model_df_csv.to_csv('./visualizations/model_comparison.csv', index=False)

    print("\n3. Analyzing pass@k vs k...")
    passk_df_dict = analyze_pass_at_k_vs_k(all_data)
    for dataset, df in passk_df_dict.items():
        print(f"\nDataset: {dataset}")
        print(df)

    print("\n4. Generating dataset summary files...")
    generate_dataset_summary(all_data)

    print("\n5. Analyzing majority vote effect...")
    maj_df = analyze_majority_vote_effect(all_data)
    print(maj_df)

    maj_df_csv = maj_df.copy()
    for col in maj_df_csv.columns:
        if col.startswith('pass@') or col == 'maj@1' or col == 'improvement':
            maj_df_csv[col] = maj_df_csv[col].apply(lambda x: '{0:.2f}'.format(x))
    maj_df_csv.to_csv('./visualizations/majority_vote_effect.csv', index=False)

    print("\nGenerating visualizations...")

    plot_temperature_effects(temp_df)
    plot_model_comparison(model_df)
    plot_pass_at_k_vs_k(passk_df_dict)
    plot_majority_vote_improvement(maj_df)

    print("\nAnalysis complete! Results and visualizations saved in ./visualizations/")


if __name__ == "__main__":
    main()
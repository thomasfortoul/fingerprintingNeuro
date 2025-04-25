import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Function to calculate differentiability for a subset of subjects
def calculate_differentiability(corr_matrix_df, subject_subset):
    """
    Calculate differentiability metrics for a subset of subjects
    
    Parameters:
    -----------
    corr_matrix_df : pandas DataFrame
        The full correlation matrix
    subject_subset : list
        List of subject IDs to include in the analysis
        
    Returns:
    --------
    diff_df : pandas DataFrame
        Differentiability metrics for the selected subjects
    """
    # Subset the correlation matrix
    subset_matrix = corr_matrix_df.loc[subject_subset, subject_subset]
    
    # Extract diagonal (self-similarity)
    self_sim = np.diag(subset_matrix.values)
    n_subjects = len(subject_subset)
    
    # Calculate differentiability
    diff_col = np.zeros(n_subjects)
    diff_row = np.zeros(n_subjects)
    
    for i in range(n_subjects):
        # Column differentiability
        column_vals = subset_matrix.iloc[:, i].values
        column_vals_without_self = np.delete(column_vals, i)
        mean_others_col = np.mean(column_vals_without_self)
        std_others_col = np.std(column_vals_without_self)
        if std_others_col > 0:
            diff_col[i] = (self_sim[i] - mean_others_col) / std_others_col
        else:
            diff_col[i] = 0
            
        # Row differentiability
        row_vals = subset_matrix.iloc[i, :].values
        row_vals_without_self = np.delete(row_vals, i)
        mean_others_row = np.mean(row_vals_without_self)
        std_others_row = np.std(row_vals_without_self)
        if std_others_row > 0:
            diff_row[i] = (self_sim[i] - mean_others_row) / std_others_row
        else:
            diff_row[i] = 0
    
    # Average differentiability
    differentiability = (diff_col + diff_row) / 2
    patient_ids = ['4', '5', '12', '18', '29', '30', '37', '47', '49', '51', '52', '54', '57', '60', '70', '71', '74', '76', '77', '80', '81', '82', '84', '86', '90', '91', '119', '127', '129']


    # Create differentiability DataFrame
    diff_df = pd.DataFrame({
        'subject': subject_subset,
        'group': ['patient' if sid in patient_ids else 'healthy' for sid in subject_subset],
        'self_similarity': self_sim,
        'z_diff_col': diff_col,
        'z_diff_row': diff_row,
        'differentiability': differentiability
    })
    diff_df.set_index('subject', inplace=True)

    return diff_df

# Function to calculate differentiation accuracy for a subset of subjects
def calculate_differentiation_accuracy(corr_matrix_df, subject_subset):
    """
    Calculate differentiation accuracy for a subset of subjects
    
    Parameters:
    -----------
    corr_matrix_df : pandas DataFrame
        The full correlation matrix
    subject_subset : list
        List of subject IDs to include in the analysis
        
    Returns:
    --------
    acc_df : pandas DataFrame
        Differentiation accuracy metrics for the selected subjects
    """
    # Subset the correlation matrix
    subset_matrix = corr_matrix_df.loc[subject_subset, subject_subset]
    n_subjects = len(subject_subset)
    
    # Initialize results dictionaries
    row_accuracy = {'correct': [], 'subject': [], 'group': []}
    col_accuracy = {'correct': [], 'subject': [], 'group': []}
    overall_accuracy = {'correct': [], 'subject': [], 'group': []}
    patient_ids = ['4', '5', '12', '18', '29', '30', '37', '47', '49', '51', '52', '54', '57', '60', '70', '71', '74', '76', '77', '80', '81', '82', '84', '86', '90', '91', '119', '127', '129']

    # For each subject, check if they're correctly differentiated
    for i in range(n_subjects):
        subj_id = subset_matrix.index[i]
        group = 'patient' if subj_id in patient_ids else 'healthy'
        
        # Row accuracy (run 1 to run 2 comparisons)
        row_vals = subset_matrix.iloc[i, :].values
        self_val = row_vals[i]
        is_correct_row = all(self_val > row_vals[j] for j in range(n_subjects) if j != i)
        
        # Column accuracy (run 2 to run 1 comparisons)
        col_vals = subset_matrix.iloc[:, i].values
        self_val = col_vals[i]
        is_correct_col = all(self_val > col_vals[j] for j in range(n_subjects) if j != i)
        
        # Overall accuracy (both row and column)
        is_correct_overall = (is_correct_row and is_correct_col)
        
        # Store results
        row_accuracy['correct'].append(int(is_correct_row))
        row_accuracy['subject'].append(subj_id)
        row_accuracy['group'].append(group)
        
        col_accuracy['correct'].append(int(is_correct_col))
        col_accuracy['subject'].append(subj_id)
        col_accuracy['group'].append(group)
        
        overall_accuracy['correct'].append(int(is_correct_overall))
        overall_accuracy['subject'].append(subj_id)
        overall_accuracy['group'].append(group)
    
    # Combine all accuracy results
    acc_df = pd.DataFrame({
        'subject': row_accuracy['subject'],
        'group': row_accuracy['group'],
        'row_correct': row_accuracy['correct'],
        'col_correct': col_accuracy['correct'],
        'overall_correct': overall_accuracy['correct']
    })
    acc_df.set_index('subject', inplace=True)
    
    return acc_df

# Function to run the sample size impact analysis
def analyze_sample_size_impact(corr_matrix_df, healthy_ids, patient_ids, 
                              min_size=5, max_size=None, step=5, n_permutations=100):
    """
    Analyze the impact of sample size on differentiability and differentiation accuracy
    
    Parameters:
    -----------
    corr_matrix_df : pandas DataFrame
        The full correlation matrix
    healthy_ids : list
        List of healthy subject IDs
    patient_ids : list
        List of patient subject IDs
    min_size : int
        Minimum sample size to test
    max_size : int or None
        Maximum sample size to test. If None, use the maximum available.
    step : int
        Step size for increasing the sample size
    n_permutations : int
        Number of random permutations to run for each sample size
        
    Returns:
    --------
    results_df : pandas DataFrame
        Results of the analysis for each sample size and group
    """
    # Determine maximum sizes based on available data
    max_healthy = len(healthy_ids) if max_size is None else min(len(healthy_ids), max_size)
    max_patient = len(patient_ids) if max_size is None else min(len(patient_ids), max_size)
    max_all = min(max_healthy + max_patient, len(corr_matrix_df)) if max_size is None else min(max_size, len(corr_matrix_df))
    
    # Create size ranges
    healthy_sizes = list(range(min_size, max_healthy+1, step))
    if healthy_sizes[-1] != max_healthy:
        healthy_sizes.append(max_healthy)
        
    patient_sizes = list(range(min_size, max_patient+1, step))
    if patient_sizes[-1] != max_patient:
        patient_sizes.append(max_patient)
        
    all_sizes = list(range(min_size, max_all+1, step))
    if all_sizes[-1] != max_all:
        all_sizes.append(max_all)
    
    # Initialize results storage
    results = []
    
    # Progress bar for the analysis
    total_iterations = (len(healthy_sizes) + len(patient_sizes) + len(all_sizes)) * n_permutations
    pbar = tqdm(total=total_iterations, desc="Running permutation analysis")
    
    # 1. Analyze healthy subjects
    for size in healthy_sizes:
        for perm in range(n_permutations):
            # Randomly sample subjects
            sampled_subjects = np.random.choice(healthy_ids, size=size, replace=False)
            
            # Calculate differentiability
            diff_df = calculate_differentiability(corr_matrix_df, sampled_subjects)
            
            # Calculate differentiation accuracy
            acc_df = calculate_differentiation_accuracy(corr_matrix_df, sampled_subjects)
            
            # Store results
            for subject in sampled_subjects:
                results.append({
                    'subject': subject,
                    'group': 'healthy',
                    'sample_size': size,
                    'sample_type': 'healthy_only',
                    'permutation': perm,
                    'differentiability': diff_df.loc[subject, 'differentiability'],
                    'self_similarity': diff_df.loc[subject, 'self_similarity'],
                    'row_correct': acc_df.loc[subject, 'row_correct'],
                    'col_correct': acc_df.loc[subject, 'col_correct'],
                    'overall_correct': acc_df.loc[subject, 'overall_correct']
                })
            
            pbar.update(1)
    
    # 2. Analyze patient subjects
    for size in patient_sizes:
        for perm in range(n_permutations):
            # Randomly sample subjects
            sampled_subjects = np.random.choice(patient_ids, size=size, replace=False)
            
            # Calculate differentiability
            diff_df = calculate_differentiability(corr_matrix_df, sampled_subjects)
            
            # Calculate differentiation accuracy
            acc_df = calculate_differentiation_accuracy(corr_matrix_df, sampled_subjects)
            
            # Store results
            for subject in sampled_subjects:
                results.append({
                    'subject': subject,
                    'group': 'patient',
                    'sample_size': size,
                    'sample_type': 'patient_only',
                    'permutation': perm,
                    'differentiability': diff_df.loc[subject, 'differentiability'],
                    'self_similarity': diff_df.loc[subject, 'self_similarity'],
                    'row_correct': acc_df.loc[subject, 'row_correct'],
                    'col_correct': acc_df.loc[subject, 'col_correct'],
                    'overall_correct': acc_df.loc[subject, 'overall_correct']
                })
            
            pbar.update(1)
    
    # 3. Analyze mixed subjects (both healthy and patient)
    for size in all_sizes:
        for perm in range(n_permutations):
            # Determine how many from each group, keeping proportions similar to full dataset
            total_subjects = len(healthy_ids) + len(patient_ids)
            n_healthy = int(np.round(size * len(healthy_ids) / total_subjects))
            n_patient = size - n_healthy
            
            # Adjust if needed
            if n_healthy > len(healthy_ids):
                n_healthy = len(healthy_ids)
                n_patient = size - n_healthy
            if n_patient > len(patient_ids):
                n_patient = len(patient_ids)
                n_healthy = size - n_patient
            
            # Randomly sample subjects
            sampled_healthy = np.random.choice(healthy_ids, size=n_healthy, replace=False)
            sampled_patients = np.random.choice(patient_ids, size=n_patient, replace=False)
            sampled_subjects = np.concatenate([sampled_healthy, sampled_patients])
            
            # Calculate differentiability
            diff_df = calculate_differentiability(corr_matrix_df, sampled_subjects)
            
            # Calculate differentiation accuracy
            acc_df = calculate_differentiation_accuracy(corr_matrix_df, sampled_subjects)
            
            # Store results
            for subject in sampled_subjects:
                group = 'patient' if subject in patient_ids else 'healthy'
                results.append({
                    'subject': subject,
                    'group': group,
                    'sample_size': size,
                    'sample_type': 'mixed',
                    'permutation': perm,
                    'differentiability': diff_df.loc[subject, 'differentiability'],
                    'self_similarity': diff_df.loc[subject, 'self_similarity'],
                    'row_correct': acc_df.loc[subject, 'row_correct'],
                    'col_correct': acc_df.loc[subject, 'col_correct'],
                    'overall_correct': acc_df.loc[subject, 'overall_correct']
                })
            
            pbar.update(1)
    
    pbar.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

# Function to create figures and run statistical tests
def analyze_and_visualize_results(results_df, output_dir="./permutation_results"):
    patient_ids = ['4', '5', '12', '18', '29', '30', '37', '47', '49', '51', '52', '54', '57', '60', '70', '71', '74', '76', '77', '80', '81', '82', '84', '86', '90', '91', '119', '127', '129']
    healthy_ids = ['1', '3', '7', '8', '9', '10', '11', '13', '14', '15', '16', '19', '20', '21', '24', '26', '27', '28', '31', '32', '33', '34', '35', '36', '38', '40', '43', '44', '45', '46', '50', '53', '55', '56', '58', '59', '61', '62', '63', '64', '65', '66', '67', '68', '72', '73', '75', '79', '85', '87', '92', '93', '94', '96', '97', '98', '99', '100', '117', '118', '120', '121', '122', '123', '125', '128', '130']
    
    """
    Analyze and visualize the results of the sample size impact analysis
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Results from the sample size impact analysis
    output_dir : str
        Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full results
    results_df.to_csv(os.path.join(output_dir, "sample_size_impact_full_results.csv"))
    
    # Calculate mean metrics by sample size, group, and sample type
    grouped_results = results_df.groupby(['sample_size', 'group', 'sample_type']).agg({
        'differentiability': ['mean', 'std', 'count'],
        'self_similarity': ['mean', 'std'],
        'row_correct': ['mean', 'std'],
        'col_correct': ['mean', 'std'],
        'overall_correct': ['mean', 'std']
    })
    
    # Reset index for easier manipulation
    grouped_results = grouped_results.reset_index()
    
    # Flatten multi-level columns
    grouped_results.columns = ['_'.join(col).strip('_') for col in grouped_results.columns.values]
    
    # Save aggregated results
    grouped_results.to_csv(os.path.join(output_dir, "sample_size_impact_aggregated.csv"))
    
    # Create separate DataFrames for each sample type for easier plotting
    healthy_only = grouped_results[grouped_results['sample_type'] == 'healthy_only']
    patient_only = grouped_results[grouped_results['sample_type'] == 'patient_only']
    mixed = grouped_results[grouped_results['sample_type'] == 'mixed']
    
    # Set up color palette
    palette = {'healthy': 'forestgreen', 'patient': 'darkred'}
    
    # ----------------- Plot Differentiability by Sample Size -----------------
    
    # 1. Healthy only
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        healthy_only['sample_size'], 
        healthy_only['differentiability_mean'], 
        yerr=healthy_only['differentiability_std']/np.sqrt(healthy_only['differentiability_count']),
        marker='o', linestyle='-', color=palette['healthy'], capsize=5,
        label=f'CU (n={len(healthy_ids)})'
    )
    plt.xlabel('Sample Size')
    plt.ylabel('Average Differentiability (Z-score)')
    plt.title('Impact of Sample Size on Differentiability - CU Group Only')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "differentiability_healthy_only.png"), dpi=300)
    plt.close()
    
    # 2. Patient only
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        patient_only['sample_size'], 
        patient_only['differentiability_mean'], 
        yerr=patient_only['differentiability_std']/np.sqrt(patient_only['differentiability_count']),
        marker='o', linestyle='-', color=palette['patient'], capsize=5,
        label=f'CU (n={len(patient_ids)})'
    )
    plt.xlabel('Sample Size')
    plt.ylabel('Differentiability (Z-score)')
    plt.title('Impact of Sample Size on Differentiability - CU Group Only')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "differentiability_patient_only.png"), dpi=300)
    plt.close()
    
    # 3. Mixed groups
    plt.figure(figsize=(10, 6))
    
    # Filter mixed data for each group
    mixed_healthy = mixed[mixed['group'] == 'healthy']
    mixed_patient = mixed[mixed['group'] == 'patient']
    
    plt.errorbar(
        mixed_healthy['sample_size'], 
        mixed_healthy['differentiability_mean'], 
        yerr=mixed_healthy['differentiability_std']/np.sqrt(mixed_healthy['differentiability_count']),
        marker='o', linestyle='-', color=palette['healthy'], capsize=5,
        label=f'CU (n={len(healthy_ids)})'
    )
    plt.errorbar(
        mixed_patient['sample_size'], 
        mixed_patient['differentiability_mean'], 
        yerr=mixed_patient['differentiability_std']/np.sqrt(mixed_patient['differentiability_count']),
        marker='o', linestyle='-', color=palette['patient'], capsize=5,
        label=f'AD (n={len(patient_ids)})'
    )
    
    plt.xlabel('Sample Size')
    plt.ylabel('Differentiability (Z-score)')
    plt.title('Impact of Sample Size on Differentiability - Mixed Groups')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "differentiability_mixed.png"), dpi=300)
    plt.close()
    
    # ----------------- Plot Differentiation Accuracy by Sample Size -----------------
    
    # 1. Healthy only - Overall accuracy
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        healthy_only['sample_size'], 
        healthy_only['overall_correct_mean'] * 100, 
        yerr=healthy_only['overall_correct_std']/np.sqrt(healthy_only['differentiability_count']) * 100,
        marker='o', linestyle='-', color=palette['healthy'], capsize=5,
        label=f'Healthy (n={len(healthy_ids)})'
    )
    plt.xlabel('Sample Size')
    plt.ylabel('Differentiation Accuracy (%)')
    plt.title('Impact of Sample Size on Differentiation Accuracy - CU Group Only')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_healthy_only.png"), dpi=300)
    plt.close()
    
    # 2. Patient only - Overall accuracy
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        patient_only['sample_size'], 
        patient_only['overall_correct_mean'] * 100, 
        yerr=patient_only['overall_correct_std']/np.sqrt(patient_only['differentiability_count']) * 100,
        marker='o', linestyle='-', color=palette['patient'], capsize=5,
        label=f'Patient (n={len(patient_ids)})'
    )
    plt.xlabel('Sample Size')
    plt.ylabel('Differentiation Accuracy (%)')
    plt.title('Impact of Sample Size on Differentiation Accuracy - Patient Group Only')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_patient_only.png"), dpi=300)
    plt.close()
    
    # 3. Mixed groups - Overall accuracy
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(
        mixed_healthy['sample_size'], 
        mixed_healthy['overall_correct_mean'] * 100, 
        yerr=mixed_healthy['overall_correct_std']/np.sqrt(mixed_healthy['differentiability_count']) * 100,
        marker='o', linestyle='-', color=palette['healthy'], capsize=5,
        label=f'CU (n={len(healthy_ids)})'
    )
    plt.errorbar(
        mixed_patient['sample_size'], 
        mixed_patient['overall_correct_mean'] * 100, 
        yerr=mixed_patient['overall_correct_std']/np.sqrt(mixed_patient['differentiability_count']) * 100,
        marker='o', linestyle='-', color=palette['patient'], capsize=5,
        label=f'AD (n={len(patient_ids)})'
    )
    
    plt.xlabel('Sample Size')
    plt.ylabel('Differentiation Accuracy (%)')
    plt.title('Impact of Sample Size on Differentiation Accuracy - Mixed Groups')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_mixed.png"), dpi=300)
    plt.close()
    
    # ----------------- Statistical Analysis -----------------
    
    # Create a combined figure for comparison across sample types
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle('Impact of Sample Size across Different Group Compositions', fontsize=16)
    
    # Plot titles
    titles = [
        'CU Only', 'AD Only', 'Mixed Groups'
    ]
    
    # Plot differentiability
    for i, (data, title) in enumerate(zip([healthy_only, patient_only, mixed], titles)):
        ax = axes[0, i]
        
        if i < 2:  # Single group plots
            group = 'healthy' if i == 0 else 'patient'
            ax.errorbar(
                data['sample_size'], 
                data['differentiability_mean'], 
                yerr=data['differentiability_std']/np.sqrt(data['differentiability_count']),
                marker='o', linestyle='-', color=palette[group], capsize=5
            )
        else:  # Mixed groups plot
            for group in ['healthy', 'patient']:
                group_data = data[data['group'] == group]
                ax.errorbar(
                    group_data['sample_size'], 
                    group_data['differentiability_mean'], 
                    yerr=group_data['differentiability_std']/np.sqrt(group_data['differentiability_count']),
                    marker='o', linestyle='-', color=palette[group], capsize=5,
                    label=group.capitalize()
                )
            ax.legend()
        
        ax.set_title(f'Average Differentiability - {title}')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Average Differentiability (Z-score)')
        ax.grid(True, alpha=0.3)
    
    # Plot accuracy
    for i, (data, title) in enumerate(zip([healthy_only, patient_only, mixed], titles)):
        ax = axes[1, i]
        
        if i < 2:  # Single group plots
            group = 'healthy' if i == 0 else 'patient'
            ax.errorbar(
                data['sample_size'], 
                data['overall_correct_mean'] * 100, 
                yerr=data['overall_correct_std']/np.sqrt(data['differentiability_count']) * 100,
                marker='o', linestyle='-', color=palette[group], capsize=5
            )
        else:  # Mixed groups plot
            for group in ['healthy', 'patient']:
                group_data = data[data['group'] == group]
                ax.errorbar(
                    group_data['sample_size'], 
                    group_data['overall_correct_mean'] * 100, 
                    yerr=group_data['overall_correct_std']/np.sqrt(group_data['differentiability_count']) * 100,
                    marker='o', linestyle='-', color=palette[group], capsize=5,
                    label=group.capitalize()
                )
            ax.legend()
        
        ax.set_title(f'Differentiation Accuracy - {title}')
        ax.set_xlabel('Permutation Test Sample Size (number of patients)')
        ax.set_ylabel('Average Accuracy (%)')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "combined_sample_size_impact.png"), dpi=300)
    plt.close()
    
    # ----------------- Statistical Testing -----------------
    
    # 1. Spearman correlation between sample size and metrics
    spearman_results = []
    
    # Process each sample type and group combination
    for sample_type in ['healthy_only', 'patient_only', 'mixed']:
        if sample_type == 'mixed':
            for group in ['healthy', 'patient']:
                subset = results_df[(results_df['sample_type'] == sample_type) & 
                                    (results_df['group'] == group)]
                
                # Correlation for differentiability
                rho_diff, p_diff = stats.spearmanr(subset['sample_size'], subset['differentiability'])
                
                # Correlation for accuracy
                rho_acc, p_acc = stats.spearmanr(subset['sample_size'], subset['overall_correct'])
                
                spearman_results.append({
                    'sample_type': sample_type,
                    'group': group,
                    'metric': 'differentiability',
                    'spearman_rho': rho_diff,
                    'p_value': p_diff,
                    'significant': p_diff < 0.05
                })
                
                spearman_results.append({
                    'sample_type': sample_type,
                    'group': group,
                    'metric': 'accuracy',
                    'spearman_rho': rho_acc,
                    'p_value': p_acc,
                    'significant': p_acc < 0.05
                })
        else:
            group = 'healthy' if sample_type == 'healthy_only' else 'patient'
            subset = results_df[results_df['sample_type'] == sample_type]
            
            # Correlation for differentiability
            rho_diff, p_diff = stats.spearmanr(subset['sample_size'], subset['differentiability'])
            
            # Correlation for accuracy
            rho_acc, p_acc = stats.spearmanr(subset['sample_size'], subset['overall_correct'])
            
            spearman_results.append({
                'sample_type': sample_type,
                'group': group,
                'metric': 'differentiability',
                'spearman_rho': rho_diff,
                'p_value': p_diff,
                'significant': p_diff < 0.05
            })
            
            spearman_results.append({
                'sample_type': sample_type,
                'group': group,
                'metric': 'accuracy',
                'spearman_rho': rho_acc,
                'p_value': p_acc,
                'significant': p_acc < 0.05
            })
    
    # Create DataFrame and save results
    spearman_df = pd.DataFrame(spearman_results)
    spearman_df.to_csv(os.path.join(output_dir, "spearman_correlation_results.csv"))
    
    # Print significant results
    significant_results = spearman_df[spearman_df['significant']]
    print("\nSignificant Spearman correlations between sample size and metrics:")
    print(significant_results)
    
    # 2. Compare metrics across different sample sizes using Mann-Whitney U test
    # For each group, compare the smallest vs largest sample size
    mw_results = []
    
    for sample_type in ['healthy_only', 'patient_only', 'mixed']:
        subset = results_df[results_df['sample_type'] == sample_type]
        
        if sample_type == 'mixed':
            for group in ['healthy', 'patient']:
                group_subset = subset[subset['group'] == group]
                min_size = group_subset['sample_size'].min()
                max_size = group_subset['sample_size'].max()
                
                # Get data for min and max sizes
                min_size_data = group_subset[group_subset['sample_size'] == min_size]
                max_size_data = group_subset[group_subset['sample_size'] == max_size]
                
                # Mann-Whitney U test for differentiability
                u_diff, p_diff = stats.mannwhitneyu(
                    min_size_data['differentiability'], 
                    max_size_data['differentiability'],
                    alternative='two-sided'
                )
                
                # Mann-Whitney U test for accuracy
                u_acc, p_acc = stats.mannwhitneyu(
                    min_size_data['overall_correct'], 
                    max_size_data['overall_correct'],
                    alternative='two-sided'
                )
                
                # Store results
                mw_results.append({
                    'sample_type': sample_type,
                    'group': group,
                    'metric': 'differentiability',
                    'min_size': min_size,
                    'max_size': max_size,
                    'min_size_mean': min_size_data['differentiability'].mean(),
                    'max_size_mean': max_size_data['differentiability'].mean(),
                    'u_statistic': u_diff,
                    'p_value': p_diff,
                    'significant': p_diff < 0.05
                })
                
                mw_results.append({
                    'sample_type': sample_type,
                    'group': group,
                    'metric': 'accuracy',
                    'min_size': min_size,
                    'max_size': max_size,
                    'min_size_mean': min_size_data['overall_correct'].mean(),
                    'max_size_mean': max_size_data['overall_correct'].mean(),
                    'u_statistic': u_acc,
                    'p_value': p_acc,
                    'significant': p_acc < 0.05
                })
        else:
            # For non-mixed groups
            min_size = subset['sample_size'].min()
            max_size = subset['sample_size'].max()
            
            # Get data for min and max sizes
            min_size_data = subset[subset['sample_size'] == min_size]
            max_size_data = subset[subset['sample_size'] == max_size]
            
            # Mann-Whitney U test for differentiability
            u_diff, p_diff = stats.mannwhitneyu(
                min_size_data['differentiability'], 
                max_size_data['differentiability'],
                alternative='two-sided'
            )
            
            # Mann-Whitney U test for accuracy
            u_acc, p_acc = stats.mannwhitneyu(
                min_size_data['overall_correct'], 
                max_size_data['overall_correct'],
                alternative='two-sided'
            )
            
            group = 'healthy' if sample_type == 'healthy_only' else 'patient'
            
            # Store results
            mw_results.append({
                'sample_type': sample_type,
                'group': group,
                'metric': 'differentiability',
                'min_size': min_size,
                'max_size': max_size,
                'min_size_mean': min_size_data['differentiability'].mean(),
                'max_size_mean': max_size_data['differentiability'].mean(),
                'u_statistic': u_diff,
                'p_value': p_diff,
                'significant': p_diff < 0.05
            })
            
            mw_results.append({
                'sample_type': sample_type,
                'group': group,
                'metric': 'accuracy',
                'min_size': min_size,
                'max_size': max_size,
                'min_size_mean': min_size_data['overall_correct'].mean(),
                'max_size_mean': max_size_data['overall_correct'].mean(),
                'u_statistic': u_acc,
                'p_value': p_acc,
                'significant': p_acc < 0.05
            })
    
    # Create DataFrame and save results
    mw_df = pd.DataFrame(mw_results)
    mw_df.to_csv(os.path.join(output_dir, "mann_whitney_test_results.csv"))
    
    # Print significant results
    significant_mw = mw_df[mw_df['significant']]
    print("\nSignificant differences between smallest and largest sample sizes:")
    print(significant_mw)
    
    # ----------------- Additional Analysis: Permutation Test -----------------
    
    # Permutation test comparing differentiability between healthy and patient groups
    # in the mixed condition at different sample sizes
    
    perm_test_results = []
    mixed_sample_sizes = mixed['sample_size'].unique()
    
    for size in mixed_sample_sizes:
        # Get data for this sample size
        size_data = results_df[(results_df['sample_type'] == 'mixed') & 
                              (results_df['sample_size'] == size)]
        
        healthy_diff = size_data[size_data['group'] == 'healthy']['differentiability'].values
        patient_diff = size_data[size_data['group'] == 'patient']['differentiability'].values
        
        # Calculate observed difference in means
        observed_diff = np.mean(healthy_diff) - np.mean(patient_diff)
        
        # Run permutation test
        n_permutations = 1000
        permutation_diffs = np.zeros(n_permutations)
        
        combined = np.concatenate([healthy_diff, patient_diff])
        n_healthy = len(healthy_diff)
        
        for i in range(n_permutations):
            np.random.shuffle(combined)
            perm_healthy = combined[:n_healthy]
            perm_patient = combined[n_healthy:]
            permutation_diffs[i] = np.mean(perm_healthy) - np.mean(perm_patient)
        
        # Calculate p-value
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        
        # Store results
        perm_test_results.append({
            'sample_size': size,
            'observed_diff': observed_diff,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    # Create DataFrame and save results
    perm_df = pd.DataFrame(perm_test_results)
    perm_df.to_csv(os.path.join(output_dir, "permutation_test_results.csv"))
    
    # Plot permutation test results
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        perm_df['sample_size'].astype(str), 
        perm_df['observed_diff'],
        color=[palette['healthy'] if diff > 0 else palette['patient'] for diff in perm_df['observed_diff']]
    )
    
    # Add significance markers
    for i, is_sig in enumerate(perm_df['significant']):
        if is_sig:
            plt.text(i, perm_df['observed_diff'].iloc[i] + 0.05, '*', 
                     ha='center', va='bottom', fontsize=14)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Sample Size')
    plt.ylabel('Difference in Differentiability (Healthy - Patient)')
    plt.title('Permutation Test: Difference in Differentiability Between Groups')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "permutation_test_results.png"), dpi=300)
    plt.close()
    
    # ----------------- Regression Analysis -----------------
    
    # Fit linear regression models to quantify the relationship between 
    # sample size and our metrics of interest
    
    regression_results = []
    
    # For each combination of sample type and group
    for sample_type in ['healthy_only', 'patient_only', 'mixed']:
        if sample_type == 'mixed':
            for group in ['healthy', 'patient']:
                subset = results_df[(results_df['sample_type'] == sample_type) & 
                                   (results_df['group'] == group)]
                
                # Fit regression for differentiability
                X = subset['sample_size'].values.reshape(-1, 1)
                y_diff = subset['differentiability'].values
                
                model_diff = stats.linregress(X.flatten(), y_diff)
                
                # Fit regression for accuracy
                y_acc = subset['overall_correct'].values
                model_acc = stats.linregress(X.flatten(), y_acc)
                
                # Store results
                regression_results.append({
                    'sample_type': sample_type,
                    'group': group,
                    'metric': 'differentiability',
                    'slope': model_diff.slope,
                    'intercept': model_diff.intercept,
                    'r_squared': model_diff.rvalue**2,
                    'p_value': model_diff.pvalue,
                    'significant': model_diff.pvalue < 0.05
                })
                
                regression_results.append({
                    'sample_type': sample_type,
                    'group': group,
                    'metric': 'accuracy',
                    'slope': model_acc.slope,
                    'intercept': model_acc.intercept,
                    'r_squared': model_acc.rvalue**2,
                    'p_value': model_acc.pvalue,
                    'significant': model_acc.pvalue < 0.05
                })
        else:
            subset = results_df[results_df['sample_type'] == sample_type]
            
            # Fit regression for differentiability
            X = subset['sample_size'].values.reshape(-1, 1)
            y_diff = subset['differentiability'].values
            
            model_diff = stats.linregress(X.flatten(), y_diff)
            
            # Fit regression for accuracy
            y_acc = subset['overall_correct'].values
            model_acc = stats.linregress(X.flatten(), y_acc)
            
            group = 'healthy' if sample_type == 'healthy_only' else 'patient'
            
            # Store results
            regression_results.append({
                'sample_type': sample_type,
                'group': group,
                'metric': 'differentiability',
                'slope': model_diff.slope,
                'intercept': model_diff.intercept,
                'r_squared': model_diff.rvalue**2,
                'p_value': model_diff.pvalue,
                'significant': model_diff.pvalue < 0.05
            })
            
            regression_results.append({
                'sample_type': sample_type,
                'group': group,
                'metric': 'accuracy',
                'slope': model_acc.slope,
                'intercept': model_acc.intercept,
                'r_squared': model_acc.rvalue**2,
                'p_value': model_acc.pvalue,
                'significant': model_acc.pvalue < 0.05
            })
    
    # Create DataFrame and save results
    reg_df = pd.DataFrame(regression_results)
    reg_df.to_csv(os.path.join(output_dir, "regression_analysis_results.csv"))
    
    # Print significant results
    significant_reg = reg_df[reg_df['significant']]
    print("\nSignificant linear relationships between sample size and metrics:")
    print(significant_reg)
    
    # ----------------- Create Summary Figure -----------------
    
    # Create a summary figure showing the impact of sample size on both metrics
    # across all conditions
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Line styles and markers for different sample types
    line_styles = {
        'healthy_only': '-',
        'patient_only': '--',
        'mixed': ':'
    }
    
    markers = {
        'healthy_only': 'o',
        'patient_only': 's',
        'mixed': '^'
    }
    
    # Plot differentiability trends from regression models
    legend_elements = []
    
    for _, row in reg_df[reg_df['metric'] == 'differentiability'].iterrows():
        sample_type = row['sample_type']
        group = row['group']
        
        # Determine color based on group
        color = palette[group]
        
        # Create x values for the line
        x_min = results_df[results_df['sample_type'] == sample_type]['sample_size'].min()
        x_max = results_df[results_df['sample_type'] == sample_type]['sample_size'].max()
        x_vals = np.linspace(x_min, x_max, 100)
        
        # Calculate y values using regression parameters
        y_vals = row['slope'] * x_vals + row['intercept']
        
        # Plot the regression line
        line = ax.plot(
            x_vals, y_vals, 
            linestyle=line_styles[sample_type],
            color=color,
            alpha=0.7
        )[0]
        
        # Create legend element
        if sample_type == 'mixed':
            label = f"{group.capitalize()} (Mixed)"
        else:
            label = f"{group.capitalize()} Only"
        
        legend_elements.append(
            Line2D([0], [0], color=color, linestyle=line_styles[sample_type],
                  label=label)
        )
    
    # Add vertical line at optimal sample size
    # We'll define this as the size where accuracy reaches 95% for healthy subjects
    acc_data = reg_df[(reg_df['metric'] == 'accuracy') & (reg_df['group'] == 'healthy')]
    
    for _, row in acc_data.iterrows():
        # Calculate the sample size where accuracy is predicted to reach 95%
        if row['slope'] > 0:  # Only if accuracy increases with sample size
            target_acc = 0.95  # 95% accuracy
            required_size = (target_acc - row['intercept']) / row['slope']
            
            # Only plot if the required size is within our range
            x_min = results_df[results_df['sample_type'] == row['sample_type']]['sample_size'].min()
            x_max = results_df[results_df['sample_type'] == row['sample_type']]['sample_size'].max()
            
            if x_min <= required_size <= x_max:
                ax.axvline(x=required_size, color='gray', linestyle='--', alpha=0.5)
                ax.text(required_size, 0.2, f"95% accuracy\nat n≈{int(required_size)}", 
                       rotation=90, ha='right', va='bottom', alpha=0.7)
    
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Differentiability (Z-score)')
    ax.set_title('Impact of Sample Size on Differentiability Across Conditions')
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "differentiability_summary.png"), dpi=300)
    plt.close()
    
    # ----------------- Final Summary Statistics -----------------
    
    # Create a summary table of key metrics
    summary_stats = []
    
    # Calculate minimum sample size needed for 95% accuracy
    for sample_type in ['healthy_only', 'patient_only', 'mixed']:
        for group in ['healthy', 'patient']:
            # Skip if this combination doesn't exist in the data
            subset = results_df[(results_df['sample_type'] == sample_type) & 
                               (results_df['group'] == group)]
            if len(subset) == 0:
                continue
            
            # Get regression results for accuracy
            reg_row = reg_df[(reg_df['sample_type'] == sample_type) & 
                           (reg_df['group'] == group) &
                           (reg_df['metric'] == 'accuracy')]
            
            if len(reg_row) > 0:
                slope = reg_row['slope'].values[0]
                intercept = reg_row['intercept'].values[0]
                
                # Calculate minimum sample size for 95% accuracy
                target_acc = 0.95
                if slope > 0:
                    min_sample_size = (target_acc - intercept) / slope
                else:
                    min_sample_size = float('inf')
                
                # Get accuracy at maximum tested sample size
                max_size = subset['sample_size'].max()
                max_size_acc = subset[subset['sample_size'] == max_size]['overall_correct'].mean()
                
                # Get differentiability at maximum tested sample size
                max_size_diff = subset[subset['sample_size'] == max_size]['differentiability'].mean()
                
                # Store results
                summary_stats.append({
                    'sample_type': sample_type,
                    'group': group,
                    'slope_diff': reg_df[(reg_df['sample_type'] == sample_type) & 
                                      (reg_df['group'] == group) &
                                      (reg_df['metric'] == 'differentiability')]['slope'].values[0],
                    'min_size_for_95pct_acc': min_sample_size if min_sample_size > 0 else float('nan'),
                    'max_tested_size': max_size,
                    'acc_at_max_size': max_size_acc,
                    'diff_at_max_size': max_size_diff
                })
    
    # Create DataFrame and save results
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(output_dir, "summary_statistics.csv"))
    
    print("\nSummary Statistics:")
    print(summary_df)
    
    # Return the results for further analysis if needed
    return results_df, grouped_results, spearman_df, mw_df, perm_df, reg_df, summary_df

def main():
    """
    Main function to run the analysis using the real correlation matrix.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define output directory
    output_dir = "./permutation_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load the actual correlation matrix from file
    print("Loading real correlation matrix from file...")
    corr_matrix_df = pd.read_csv("subject_similarity_matrix_reordered.csv", index_col=0)

    # Extract subject IDs and group them based on naming convention
    all_subjects = corr_matrix_df.index.tolist()
    pat_ids = [4, 5, 12, 18, 29, 30, 37, 47, 49, 51, 52, 54, 57, 60,
               70, 71, 74, 76, 77, 80, 81, 82, 84, 86, 90, 91, 119, 127, 129]
    patient_ids = ['4', '5', '12', '18', '29', '30', '37', '47', '49', '51', '52', '54', '57', '60', '70', '71', '74', '76', '77', '80', '81', '82', '84', '86', '90', '91', '119', '127', '129']
    healthy_ids = ['1', '3', '7', '8', '9', '10', '11', '13', '14', '15', '16', '19', '20', '21', '24', '26', '27', '28', '31', '32', '33', '34', '35', '36', '38', '40', '43', '44', '45', '46', '50', '53', '55', '56', '58', '59', '61', '62', '63', '64', '65', '66', '67', '68', '72', '73', '75', '79', '85', '87', '92', '93', '94', '96', '97', '98', '99', '100', '117', '118', '120', '121', '122', '123', '125', '128', '130']
    corr_matrix_df.index = corr_matrix_df.index.astype(str)
    corr_matrix_df.columns = corr_matrix_df.columns.astype(str)

    # Run the analysis
    print("Running sample size impact analysis...")
    results_df = analyze_sample_size_impact(
        corr_matrix_df, 
        healthy_ids, 
        patient_ids,
        min_size=5,
        max_size=None,  # Use all available subjects
        step=5,
        n_permutations=1000
    )

    # Analyze and visualize the results
    print("Analyzing and visualizing results...")
    analyze_and_visualize_results(results_df, output_dir)

    print(f"Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()

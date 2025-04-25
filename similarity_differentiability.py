import numpy as np
import scipy.io
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import random
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
import matplotlib.gridspec as gridspec
from statsmodels.stats.multitest import multipletests

# Set the folder where the copied .mat files are stored
data_folder = '/Users/thfo2021/Desktop/McGill/BailletLab/Processed_Files_copy'

# Get a list of all .mat files in the folder
mat_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.mat')])

# Initialize an empty list to store TF data
tf_data = []
subject_ids = []
# Load each .mat file and extract the TF variable
for mat_file in mat_files:
    file_path = os.path.join(data_folder, mat_file)
    data = scipy.io.loadmat(file_path)

    if 'TF' in data:
        tf_data.append(data['TF'])
        # parse the subject ID from the filename
        sid = int(mat_file.split('.')[0][8:-2])
        subject_ids.append(sid)
    else:
        print(f"Warning: TF variable not found in file: {mat_file}")

# dedupe & sort
subject_ids = sorted(set(subject_ids))

# Convert list of TF matrices into a numpy array (68x1x81xN)
tf_matrix = np.stack(tf_data, axis=-1)

# Save the combined TF matrix for further analysis
scipy.io.savemat(os.path.join(data_folder, 'combined_TF_data2.mat'),
                 {'tf_matrix': tf_matrix})

# Reshape into subjects × features
num_subjects = tf_matrix.shape[3] // 2  # 2 runs per subject
reshaped_data = tf_matrix.reshape(68 * 81, -1).T  # (subjects*2) × features

# Split into run 1 and run 2, indexed by subject_ids
df_run_1 = pd.DataFrame(reshaped_data[::2],
                        columns=[f"Feature_{i}" for i in range(reshaped_data.shape[1])],
                        index=subject_ids)
df_run_2 = pd.DataFrame(reshaped_data[1::2],
                        columns=[f"Feature_{i}" for i in range(reshaped_data.shape[1])],
                        index=subject_ids)
df_combined = pd.concat([df_run_1, df_run_2], axis=0).reset_index(drop=True)

# Compute correlation matrix across all subjects
n = df_run_1.shape[0]
SubjectCorrMatrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        SubjectCorrMatrix[i, j] = np.corrcoef(df_run_1.iloc[i], df_run_2.iloc[j])[0, 1]

# Build DataFrame
SubjectCorrMatrix_df = pd.DataFrame(SubjectCorrMatrix,
                                    index=subject_ids,
                                    columns=subject_ids)

# ── Define your patient IDs and reorder ─────────────────────────────────
patient_ids = [4, 5, 12, 18, 29, 30, 37, 47, 49, 51, 52, 54, 57, 60,
               70, 71, 74, 76, 77, 80, 81, 82, 84, 86, 90, 91, 119, 127, 129]
# healthy IDs in original order
healthy_ids = [sid for sid in subject_ids if sid not in patient_ids]
# new ordering: all healthy first, then patient
new_order = healthy_ids + patient_ids

# reindex rows and columns
SubjectCorrMatrix_df = SubjectCorrMatrix_df.loc[new_order, new_order]
# ───────────────────────────────────────────────────────────────────────────────

# Save as CSV
SubjectCorrMatrix_df.to_csv("subject_similarity_matrix_reordered.csv")

# Visualize as heatmap
plt.figure(figsize=(12, 10))
mask = np.zeros_like(SubjectCorrMatrix_df, dtype=bool)
sns.heatmap(SubjectCorrMatrix_df, annot=False, cmap="coolwarm", center=0,
            linewidths=0.5, mask=mask)
plt.axvline(x=len(healthy_ids), color='black', linestyle='-', linewidth=2)
plt.axhline(y=len(healthy_ids), color='black', linestyle='-', linewidth=2)
plt.title("Subject Similarity Matrix")
plt.xlabel("Subject ID (Left: CU, Right: AD)")
plt.ylabel("Subject ID (Left: CU, Right: AD)")
plt.savefig("SubjectSimilarity_Reordered.png")
plt.close()

# ── Define subsets ────────────────────────────────────────────────────────────────
hh_corr = SubjectCorrMatrix_df.loc[healthy_ids, healthy_ids]
pp_corr = SubjectCorrMatrix_df.loc[patient_ids, patient_ids]
hp_corr = SubjectCorrMatrix_df.loc[healthy_ids, patient_ids]
# ────────────────────────────────────────────────────────────────────────────────

# ── Save each subset to CSV ───────────────────────────────────────────────────────
hh_corr.to_csv("corr_healthy_healthy.csv")
pp_corr.to_csv("corr_patient_patient.csv")
hp_corr.to_csv("corr_healthy_patient.csv")
# ────────────────────────────────────────────────────────────────────────────────

# ── Plot heatmaps with mask for diagonal ─────────────────────────────────────────
for mat, title, fname in [
    (hh_corr, "CU x CU Correlation", "heatmap_healthy_healthy.png"),
    (pp_corr, "AD x AD Correlation", "heatmap_patient_patient.png"),
    (hp_corr, "CU x AD Correlation", "heatmap_healthy_patient.png"),
]:
    plt.figure(figsize=(8, 6))
    mask = np.zeros_like(mat, dtype=bool) if "x Patient" not in title else None
    sns.heatmap(mat, cmap="coolwarm", center=0, linewidths=0.5, mask=mask)
    plt.title(title)
    plt.xlabel("Subject ID")
    plt.ylabel("Subject ID")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
# ────────────────────────────────────────────────────────────────────────────────

# Build the summary table
rows = []
for subj in SubjectCorrMatrix_df.index:
    # exclude self when averaging
    healthy_others = [h for h in healthy_ids if h != subj]
    patient_others = [p for p in patient_ids if p != subj]
    all_others = [s for s in SubjectCorrMatrix_df.index if s != subj]

    avg_h = SubjectCorrMatrix_df.loc[subj, healthy_others].mean()
    avg_p = SubjectCorrMatrix_df.loc[subj, patient_others].mean()
    avg_a = SubjectCorrMatrix_df.loc[subj, all_others].mean()
    grp = 'patient' if subj in patient_ids else 'healthy'

    rows.append({
        'subject': subj,
        'group': grp,
        'avg_corr_all_others': avg_a,
        'avg_corr_healthy': avg_h,
        'avg_corr_patients': avg_p,
        'self_similarity': SubjectCorrMatrix_df.loc[subj, subj]
    })

df_summary = pd.DataFrame(rows).set_index('subject')

# Show the table
print(df_summary)
df_summary.to_csv('Subject-level correlation comparison.csv')

# Scatter‐plot: each point is one subject
plt.figure(figsize=(10, 8))
scatter_h = plt.scatter(df_summary[df_summary['group'] == 'healthy']['avg_corr_healthy'], 
           df_summary[df_summary['group'] == 'healthy']['avg_corr_patients'],
           label='CU', alpha=0.7, s=80)
scatter_p = plt.scatter(df_summary[df_summary['group'] == 'patient']['avg_corr_healthy'],
           df_summary[df_summary['group'] == 'patient']['avg_corr_patients'],
           label='AD', alpha=0.7, s=80, marker='s')

# Add regression lines for both groups
for group, color in [('healthy', 'blue'), ('patient', 'orange')]:
    subset = df_summary[df_summary['group'] == group]
    x = subset['avg_corr_healthy']
    y = subset['avg_corr_patients']
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color=color, linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Unity line
plt.xlabel('Average Correlation with CU Subjects', fontsize=12)
plt.ylabel('Average Correlation with AD Subjects', fontsize=12)
plt.title('Subject‐level Correlation Comparison', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("subject_correlation_comparison.png")
plt.close()

###########################################################################
# SELF-SIMILARITY ANALYSIS
###########################################################################

# Extract diagonal (self-similarity)
self_sim = np.diag(SubjectCorrMatrix_df.values)
self_sim_df = pd.DataFrame({
    'subject': SubjectCorrMatrix_df.index,
    'group': ['patient' if sid in patient_ids else 'healthy' for sid in SubjectCorrMatrix_df.index],
    'self_similarity': self_sim
})
self_sim_df.set_index('subject', inplace=True)

# Statistical test for self-similarity between groups
healthy_self = self_sim_df[self_sim_df['group'] == 'healthy']['self_similarity']
patient_self = self_sim_df[self_sim_df['group'] == 'patient']['self_similarity']

# Test for normality
shapiro_healthy = stats.shapiro(healthy_self)
shapiro_patient = stats.shapiro(patient_self)
print("\nNormality Test for Self-Similarity:")
print(f"Healthy: Shapiro-Wilk p={shapiro_healthy.pvalue:.4f}")
print(f"Patient: Shapiro-Wilk p={shapiro_patient.pvalue:.4f}")

# Use appropriate test based on normality
if shapiro_healthy.pvalue > 0.05 and shapiro_patient.pvalue > 0.05:
    # Both are normally distributed, use t-test
    ttest_result = stats.ttest_ind(healthy_self, patient_self, equal_var=False)
    print(f"\nIndependent t-test for Self-Similarity: t={ttest_result.statistic:.4f}, p={ttest_result.pvalue:.4f}")
    test_used = "Independent t-test"
    test_stat = ttest_result.statistic
    p_val = ttest_result.pvalue
else:
    # Non-parametric test
    mannw_result = stats.mannwhitneyu(healthy_self, patient_self)
    print(f"\nMann-Whitney U test for Self-Similarity: U={mannw_result.statistic:.4f}, p={mannw_result.pvalue:.4f}")
    test_used = "Mann-Whitney U test"
    test_stat = mannw_result.statistic
    p_val = mannw_result.pvalue

# Save statistical results
stats_results = pd.DataFrame({
    'Metric': ['Self-Similarity'],
    'Test Used': [test_used],
    'Statistic': [test_stat],
    'p-value': [p_val],
    'Healthy Mean': [healthy_self.mean()],
    'Healthy Std': [healthy_self.std()],
    'Patient Mean': [patient_self.mean()],
    'Patient Std': [patient_self.std()]
})
stats_results.to_csv('self_similarity_stats.csv', index=False)

# Visualization of self-similarity distribution
plt.figure(figsize=(12, 6))
# Create a subplot grid: 1 row, 2 columns
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 

# Plot KDE in the first (larger) subplot
ax1 = plt.subplot(gs[0])
sns.kdeplot(healthy_self, label="CU", fill=True, alpha=0.4, ax=ax1)
sns.kdeplot(patient_self, label="AD", fill=True, alpha=0.4, ax=ax1)
ax1.axvline(healthy_self.mean(), color='blue', linestyle='--', 
           label=f"CU Mean: {healthy_self.mean():.3f}")
ax1.axvline(patient_self.mean(), color='orange', linestyle='--', 
           label=f"AD Mean: {patient_self.mean():.3f}")
ax1.set_xlabel("Self-Similarity Score")
ax1.set_ylabel("Density")
ax1.set_title("Self-Similarity Distributions")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot boxplot in the second subplot
# Map group names to CU/AD
self_sim_df['Group'] = self_sim_df['group'].map({'healthy': 'CU', 'patient': 'AD'})

# Create boxplot with custom group names
ax2 = plt.subplot(gs[1])
sns.boxplot(x='Group', y='self_similarity', data=self_sim_df, ax=ax2)
sns.stripplot(x='Group', y='self_similarity', data=self_sim_df, 
              jitter=True, dodge=True, alpha=0.7, ax=ax2)

# Title and formatting
ax2.set_title('Self-Similarity by Group')
ax2.set_xlabel('Group')  # Explicitly set x-axis label
ax2.set_ylabel('Self-Similarity')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("self_similarity_analysis.png")
plt.close()

########################### OUTLIERS##############
# ── Identify Outliers in Self-Similarity ─────────────────────────────
Q1 = self_sim_df['self_similarity'].quantile(0.25)
Q3 = self_sim_df['self_similarity'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Get subjects with outlier self-similarity values
outliers = self_sim_df[(self_sim_df['self_similarity'] < lower_bound) | 
                       (self_sim_df['self_similarity'] > upper_bound)]

# Filter to only patient outliers
outlier_patients = outliers[outliers['group'] == 'patient']
print("Outlier Patient IDs based on Self-Similarity:")
print(outlier_patients.index.tolist())
print(self_sim_df.loc[outlier_patients.index])


outlier_patients = outliers[outliers['group'] != 'patient']
print("Outlier Healthy IDs based on Self-Similarity:")
print(outlier_patients.index.tolist())
print(self_sim_df.loc[outlier_patients.index])


#GET OVERALL STATS

box_stats = self_sim_df.groupby('group')['self_similarity'].describe()
mean_values = self_sim_df.groupby('group')['self_similarity'].mean()
median_values = self_sim_df.groupby('group')['self_similarity'].median()

print("Box Plot Summary Statistics:")
print(box_stats)
print("\nMean Self-Similarity by Group:")
print(mean_values)
print("\nMedian Self-Similarity by Group:")
print(median_values)

########################### OUTLIERS##############

# Save self-similarity data
self_sim_df.to_csv("self_similarity_by_subject.csv")

# Subject-level analysis: self-similarity vs. avg correlation with same/different group
plt.figure(figsize=(15, 5))

# Subplot 1: Self-similarity vs avg correlation with same group
plt.subplot(1, 3, 1)
for group, color, marker in [('healthy', 'blue', 'o'), ('patient', 'orange', 's')]:
    subset = df_summary[df_summary['group'] == group]
    same_group_col = 'avg_corr_healthy' if group == 'healthy' else 'avg_corr_patients'
    plt.scatter(subset['self_similarity'], subset[same_group_col], 
                color=color, alpha=0.7, label=group, marker=marker)
    
    # Add linear regression
    x = subset['self_similarity']
    y = subset[same_group_col]
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color=color, linestyle='--')
    
    # Calculate correlation
    corr = stats.pearsonr(subset['self_similarity'], subset[same_group_col])
    print(f"{group} - Correlation between self-similarity and avg correlation with same group: r={corr[0]:.3f}, p={corr[1]:.4f}")

plt.xlabel('Self-Similarity')
plt.ylabel('Avg Correlation with Same Group')
plt.title('Self-Similarity vs. Same Group Correlation')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Self-similarity vs avg correlation with different group
plt.subplot(1, 3, 2)
for group, color, marker in [('healthy', 'blue', 'o'), ('patient', 'orange', 's')]:
    subset = df_summary[df_summary['group'] == group]
    diff_group_col = 'avg_corr_patients' if group == 'healthy' else 'avg_corr_healthy'
    plt.scatter(subset['self_similarity'], subset[diff_group_col], 
                color=color, alpha=0.7, label=group, marker=marker)
    
    # Add linear regression
    x = subset['self_similarity']
    y = subset[diff_group_col]
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color=color, linestyle='--')
    
    # Calculate correlation
    corr = stats.pearsonr(subset['self_similarity'], subset[diff_group_col])
    print(f"{group} - Correlation between self-similarity and avg correlation with different group: r={corr[0]:.3f}, p={corr[1]:.4f}")

plt.xlabel('Self-Similarity')
plt.ylabel('Avg Correlation with Different Group')
plt.title('Self-Similarity vs. Different Group Correlation')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Self-similarity vs group separation (difference between same-group and diff-group correlation)
plt.subplot(1, 3, 3)
# Calculate group separation
df_summary['group_separation'] = np.where(
    df_summary['group'] == 'healthy',
    df_summary['avg_corr_healthy'] - df_summary['avg_corr_patients'],
    df_summary['avg_corr_patients'] - df_summary['avg_corr_healthy']
)

for group, color, marker in [('healthy', 'blue', 'o'), ('patient', 'orange', 's')]:
    subset = df_summary[df_summary['group'] == group]
    plt.scatter(subset['self_similarity'], subset['group_separation'], 
                color=color, alpha=0.7, label=group, marker=marker)
    
    # Add linear regression
    x = subset['self_similarity']
    y = subset['group_separation']
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color=color, linestyle='--')
    
    # Calculate correlation
    corr = stats.pearsonr(subset['self_similarity'], subset['group_separation'])
    print(f"{group} - Correlation between self-similarity and group separation: r={corr[0]:.3f}, p={corr[1]:.4f}")

plt.xlabel('Self-Similarity')
plt.ylabel('Group Separation\n(Same Group - Different Group Correlation)')
plt.title('Self-Similarity vs. Group Separation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("self_similarity_relationships.png")
plt.close()

###########################################################################
# DIFFERENTIABILITY ANALYSIS
###########################################################################

# Extract diagonal (self-similarity)
self_sim = np.diag(SubjectCorrMatrix_df.values)

# Calculate differentiability properly
# Along columns: how different is self-similarity from similarity to others in column
diff_col = np.zeros(len(subject_ids))
for i in range(len(subject_ids)):
    column_vals = SubjectCorrMatrix_df.iloc[:, i].values
    column_vals_without_self = np.delete(column_vals, i)  # Remove self correlation
    mean_others = np.mean(column_vals_without_self)  # Mean of other correlations
    std_others = np.std(column_vals_without_self)  # Std of other correlations
    if std_others > 0:  # Avoid division by zero
        diff_col[i] = (self_sim[i] - mean_others) / std_others
    else:
        diff_col[i] = 0

# Along rows: how different is self-similarity from similarity to others in row
diff_row = np.zeros(len(subject_ids))
for i in range(len(subject_ids)):
    row_vals = SubjectCorrMatrix_df.iloc[i, :].values
    row_vals_without_self = np.delete(row_vals, i)  # Remove self correlation
    mean_others = np.mean(row_vals_without_self)  # Mean of other correlations
    std_others = np.std(row_vals_without_self)  # Std of other correlations
    if std_others > 0:  # Avoid division by zero
        diff_row[i] = (self_sim[i] - mean_others) / std_others
    else:
        diff_row[i] = 0

# Average differentiability
differentiability = (diff_col + diff_row) / 2

# Create differentiability DataFrame
diff_df = pd.DataFrame({
    'subject': SubjectCorrMatrix_df.index,
    'group': ['patient' if sid in patient_ids else 'healthy' for sid in SubjectCorrMatrix_df.index],
    'self_similarity': self_sim,
    'z_diff_col': diff_col,
    'z_diff_row': diff_row,
    'differentiability': differentiability,
    'diff_raw_col': self_sim - np.array([np.mean(np.delete(SubjectCorrMatrix_df.iloc[:, i].values, i)) for i in range(len(subject_ids))]),
    'diff_raw_row': self_sim - np.array([np.mean(np.delete(SubjectCorrMatrix_df.iloc[i, :].values, i)) for i in range(len(subject_ids))])
})
diff_df['diff_raw_avg'] = (diff_df['diff_raw_col'] + diff_df['diff_raw_row']) / 2
diff_df.set_index('subject', inplace=True)

# Subset by group
diff_healthy = diff_df[diff_df['group'] == 'healthy']
diff_patient = diff_df[diff_df['group'] == 'patient']

# Statistical tests for differentiability
# Test for normality
shapiro_healthy = stats.shapiro(diff_healthy['differentiability'])
shapiro_patient = stats.shapiro(diff_patient['differentiability'])
print("\nNormality Test for Differentiability:")
print(f"Healthy: Shapiro-Wilk p={shapiro_healthy.pvalue:.4f}")
print(f"Patient: Shapiro-Wilk p={shapiro_patient.pvalue:.4f}")

# Use appropriate test based on normality
if shapiro_healthy.pvalue > 0.05 and shapiro_patient.pvalue > 0.05:
    # Both are normally distributed, use t-test
    ttest_result = stats.ttest_ind(diff_healthy['differentiability'], 
                                   diff_patient['differentiability'], 
                                   equal_var=False)
    print(f"\nIndependent t-test for Differentiability: t={ttest_result.statistic:.4f}, p={ttest_result.pvalue:.4f}")
    diff_test_used = "Independent t-test"
    diff_test_stat = ttest_result.statistic
    diff_p_val = ttest_result.pvalue
else:
    # Non-parametric test
    mannw_result = stats.mannwhitneyu(diff_healthy['differentiability'], 
                                     diff_patient['differentiability'])
    print(f"\nMann-Whitney U test for Differentiability: U={mannw_result.statistic:.4f}, p={mannw_result.pvalue:.4f}")
    diff_test_used = "Mann-Whitney U test"
    diff_test_stat = mannw_result.statistic
    diff_p_val = mannw_result.pvalue

# Also test raw difference
ttest_raw = stats.ttest_ind(diff_healthy['diff_raw_avg'], 
                          diff_patient['diff_raw_avg'], 
                          equal_var=False)
print(f"\nIndependent t-test for Raw Difference: t={ttest_raw.statistic:.4f}, p={ttest_raw.pvalue:.4f}")

# Save the results
stats_results = pd.DataFrame({
    'Metric': ['Differentiability', 'Raw Difference'],
    'Test Used': [diff_test_used, "Independent t-test"],
    'Statistic': [diff_test_stat, ttest_raw.statistic],
    'p-value': [diff_p_val, ttest_raw.pvalue],
    'Healthy Mean': [diff_healthy['differentiability'].mean(), diff_healthy['diff_raw_avg'].mean()],
    'Healthy Std': [diff_healthy['differentiability'].std(), diff_healthy['diff_raw_avg'].std()],
    'Patient Mean': [diff_patient['differentiability'].mean(), diff_patient['diff_raw_avg'].mean()],
    'Patient Std': [diff_patient['differentiability'].std(), diff_patient['diff_raw_avg'].std()]
})
stats_results.to_csv('differentiability_stats.csv', index=False)

# Save to CSV
diff_df.to_csv("all_differentiability_scores.csv")
diff_healthy.to_csv("healthy_differentiability_scores.csv")
diff_patient.to_csv("patient_differentiability_scores.csv")

# Plot overall differentiability distributions
plt.figure(figsize=(15, 5))

# Plot 1: KDE of differentiability
plt.subplot(1, 3, 1)
sns.kdeplot(diff_healthy['differentiability'], label="CU", fill=True, alpha=0.4)
sns.kdeplot(diff_patient['differentiability'], label="AD", fill=True, alpha=0.4)
plt.axvline(diff_healthy['differentiability'].mean(), color='blue', linestyle='--', 
           label=f"CU Mean: {diff_healthy['differentiability'].mean():.2f}")
plt.axvline(diff_patient['differentiability'].mean(), color='orange', linestyle='--', 
           label=f"AD Mean: {diff_patient['differentiability'].mean():.2f}")
plt.xlabel("Differentiability Score (z-score)")
plt.ylabel("Density")
plt.title("Differentiability Distributions")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Box plot for differentiability
# Map group names for differentiability plot
diff_df['Group'] = diff_df['group'].map({'healthy': 'CU', 'patient': 'AD'})

plt.subplot(1, 3, 2)
sns.boxplot(x='Group', y='differentiability', data=diff_df)
sns.stripplot(x='Group', y='differentiability', data=diff_df, 
              jitter=True, dodge=True, alpha=0.7)

plt.title('Differentiability by Group')
plt.xlabel('Group')  # Set x-axis label explicitly
plt.ylabel('Differentiability')
plt.grid(True, alpha=0.3)

# Plot 3: Raw difference distributions
plt.subplot(1, 3, 3)
sns.kdeplot(diff_healthy['diff_raw_avg'], label="CU", fill=True, alpha=0.4)
sns.kdeplot(diff_patient['diff_raw_avg'], label="AD", fill=True, alpha=0.4)
plt.axvline(diff_healthy['diff_raw_avg'].mean(), color='blue', linestyle='--', 
           label=f"CU Mean: {diff_healthy['diff_raw_avg'].mean():.3f}")
plt.axvline(diff_patient['diff_raw_avg'].mean(), color='orange', linestyle='--', 
           label=f"AD Mean: {diff_patient['diff_raw_avg'].mean():.3f}")
plt.xlabel("Raw Difference (Self - Others)")
plt.ylabel("Density")
plt.title("Raw Difference Distributions")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("differentiability_distributions.png")
plt.close()

# Calculate group statistics
group_stats = diff_df.groupby('group')['differentiability'].agg(['mean', 'std', 'min', 'max', 'count'])
print("\nDifferentiability Statistics by Group:")
print(group_stats)
group_stats.to_csv("differentiability_group_statistics.csv")

# Relationship between self-similarity and differentiability
plt.figure(figsize=(10, 6))
scatter_h = plt.scatter(diff_healthy['self_similarity'], diff_healthy['differentiability'], 
                       label='CU', alpha=0.7, s=80)
scatter_p = plt.scatter(diff_patient['self_similarity'], diff_patient['differentiability'], 
                       label='AD', alpha=0.7, s=80, marker='s')

# Add regression lines for both groups
for subset, color in [(diff_healthy, 'blue'), (diff_patient, 'orange')]:
    x = subset['self_similarity']
    y = subset['differentiability']
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color=color, linestyle='--')
    
    # Calculate correlation
    corr = stats.pearsonr(x, y)
    label = 'CU' if color == 'blue' else 'AD'
    print(f"{label} - Correlation between self-similarity and differentiability: r={corr[0]:.3f}, p={corr[1]:.4f}")

plt.xlabel('Self-Similarity')
plt.ylabel('Differentiability Score')
plt.title('Relationship Between Self-Similarity and Differentiability')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('self_similarity_vs_differentiability.png')
plt.close()

# ROC Curve Analysis for Differentiability as a classifier of group
fpr, tpr, thresholds = roc_curve(diff_df['group'] == 'patient', diff_df['differentiability'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_differentiability.png')
plt.close()


# ── Identify Outliers in Differentiability ─────────────────────────────
Q1 = diff_df['differentiability'].quantile(0.25)
Q3 = diff_df['differentiability'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Get outliers
outliers_diff = diff_df[(diff_df['differentiability'] < lower_bound) | 
                        (diff_df['differentiability'] > upper_bound)]

# Split by group
outlier_healthy_ids = outliers_diff[outliers_diff['group'] == 'healthy'].index.tolist()
outlier_patient_ids = outliers_diff[outliers_diff['group'] == 'patient'].index.tolist()

print("Outlier Healthy IDs (Differentiability):", outlier_healthy_ids)
print("Outlier Patient IDs (Differentiability):", outlier_patient_ids)

# ── Print Stats for Outliers ─────────────────────────────
print("\nOutlier Healthy Stats:")
print(diff_df.loc[outlier_healthy_ids])

print("\nOutlier Patient Stats:")
print(diff_df.loc[outlier_patient_ids])

# ── Extract Box Plot Summary Stats ───────────────────────
box_stats_diff = diff_df.groupby('group')['differentiability'].describe()
mean_diff = diff_df.groupby('group')['differentiability'].mean()
median_diff = diff_df.groupby('group')['differentiability'].median()

print("\nDifferentiability Box Plot Summary Stats:")
print(box_stats_diff)
print("\nMean Differentiability by Group:")
print(mean_diff)
print("\nMedian Differentiability by Group:")
print(median_diff)


# ── Extract Regression and Correlation Values ─────────────────────────────
regression_stats = {}
correlation_stats = {}

for subset, label in [(diff_healthy, 'Healthy'), (diff_patient, 'Patient')]:
    x = subset['self_similarity']
    y = subset['differentiability']
    m, b = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)
    
    regression_stats[label] = {'slope': m, 'intercept': b}
    correlation_stats[label] = {'r': r, 'p-value': p}

print("\nRegression Coefficients:")
print(regression_stats)

print("\nPearson Correlation Coefficients:")
print(correlation_stats)

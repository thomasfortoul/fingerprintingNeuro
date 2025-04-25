###########################################################################
# LONGITUDINAL ANALYSIS OF SELF-SIMILARITY, DIFFERENTIABILITY, AND ACCURACY
###########################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Define the second scan dates for patients
second_scan_data = """
4	31/3/23
5	1/10/19
12	29/7/22
18	15/11/18
29	22/3/23
30	2/5/23
37	17/3/21
47	12/6/23
49	8/11/19
51	13/12/23
52	12/12/18
54	12/7/23
57	15/9/21
60	1/9/23
70	25/10/23
71	20/7/18
74	17/12/19
76	2/12/21
77	22/7/21
80	12/4/23
81	14/10/20
82	19/8/22
84	12/7/22
86	17/1/19
90	5/4/24
91	29/8/23
110	11/9/23
111	8/9/23
115	12/6/19
119	17/2/20
127	25/4/23
129	3/3/23
"""

# Process second scan data
second_scan_lines = [line.strip() for line in second_scan_data.strip().split('\n')]
second_scan_dict = {}

for line in second_scan_lines:
    parts = line.split('\t')
    if len(parts) == 2:
        subject_id = int(parts[0])
        date_str = parts[1]
        # Convert date string to datetime object (assuming day/month/year format)
        date_obj = datetime.strptime(date_str, '%d/%m/%y')
        second_scan_dict[subject_id] = date_obj

# Assuming the first scan was in 2017-01-01 for all subjects (adjust if you have actual dates)
first_scan_date = datetime(2017, 1, 1)

# Calculate time between scans in months
time_between_scans = {}
for subject_id, second_date in second_scan_dict.items():
    delta = second_date - first_scan_date
    months = delta.days / 30.44  # Average days in a month
    time_between_scans[subject_id] = months

# Load the combined dataset
try:
    combined_df = pd.read_csv('combined_differentiability_accuracy.csv', index_col='subject')
except FileNotFoundError:
    print("Error: combined_differentiability_accuracy.csv not found. Make sure to run the original analysis first.")
    # If file not found, we'll use the variables from the original code
    combined_df = pd.merge(diff_df, acc_df, left_index=True, right_index=True, suffixes=('', '_y'))
    combined_df = combined_df.drop('group_y', axis=1)

# Load the original correlation matrix
try:  
    SubjectCorrMatrix_df = pd.read_csv("subject_similarity_matrix_reordered.csv", index_col=0)
    SubjectCorrMatrix_df.index = SubjectCorrMatrix_df.index.astype(int)
    SubjectCorrMatrix_df.columns = SubjectCorrMatrix_df.columns.astype(int)
except FileNotFoundError:
    print("Warning: subject_similarity_matrix_reordered.csv not found. Using matrix from original code.")
    # The SubjectCorrMatrix_df should be available from the original code

# Add time between scans to the combined dataframe
combined_df['months_between_scans'] = pd.Series(time_between_scans)

# Create a dataframe with just the patients who had second scans
patients_with_second_scan = combined_df[combined_df.index.isin(second_scan_dict.keys())]

# Plot the relationship between time and metrics
plt.figure(figsize=(18, 12))

# 1. Self-similarity vs Time
plt.subplot(2, 3, 1)
sns.scatterplot(data=patients_with_second_scan, x='months_between_scans', y='self_similarity', hue='overall_correct')
plt.title('Self-similarity vs Months Between Scans')
plt.xlabel('Months Between Scans')
plt.ylabel('Self-similarity')
plt.grid(True, alpha=0.3)

# Add regression line
x = patients_with_second_scan['months_between_scans']
y = patients_with_second_scan['self_similarity']
if len(x) > 0 and len(y) > 0:  # Ensure there's data to fit
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--')
    # Calculate correlation coefficient and p-value
    corr, p_value = stats.pearsonr(x, y)
    plt.text(x.min() + 2, y.min() + 0.05, f'r = {corr:.3f}, p = {p_value:.3f}', fontsize=10)

# 2. Differentiability vs Time
plt.subplot(2, 3, 2)
sns.scatterplot(data=patients_with_second_scan, x='months_between_scans', y='differentiability', hue='overall_correct')
plt.title('Differentiability vs Months Between Scans')
plt.xlabel('Months Between Scans')
plt.ylabel('Differentiability')
plt.grid(True, alpha=0.3)

# Add regression line
x = patients_with_second_scan['months_between_scans']
y = patients_with_second_scan['differentiability']
if len(x) > 0 and len(y) > 0:
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--')
    corr, p_value = stats.pearsonr(x, y)
    plt.text(x.min() + 2, y.min() + 0.05, f'r = {corr:.3f}, p = {p_value:.3f}', fontsize=10)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression

plt.figure(figsize=(10, 6))

# Scatter plot
sns.scatterplot(data=patients_with_second_scan,
                x='months_between_scans', y='overall_correct',
                alpha=0.7, s=100)

plt.title('Correct Differentiation vs Months Between Scans', fontsize=14)
plt.xlabel('Months Between Scans', fontsize=12)
plt.ylabel('Correct Differentiation (0 or 1)', fontsize=12)
plt.yticks([0, 1])
plt.grid(True, alpha=0.3)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjust width here
ax = axes[0, 2]  # This is subplot 2,3,3

sns.scatterplot(data=patients_with_second_scan,
                x='months_between_scans', y='overall_correct',
                alpha=0.7, s=100, ax=ax)

ax.set_title('Correct Differentiation vs Months Between Scans')
ax.set_xlabel('Months Between Scans')
ax.set_ylabel('Correct Differentiation (0 or 1)')
ax.set_yticks([0, 1])
ax.grid(True, alpha=0.3)

if 0 in patients_with_second_scan['overall_correct'].values and 1 in patients_with_second_scan['overall_correct'].values:
    X = patients_with_second_scan['months_between_scans'].values.reshape(-1, 1)
    y = patients_with_second_scan['overall_correct'].values
    model = LogisticRegression(random_state=0)
    model.fit(X, y)

    x_pred = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred = model.predict_proba(x_pred)[:, 1]

    ax.plot(x_pred, y_pred, color='red', linestyle='--', linewidth=2)

    score = model.score(X, y)
    ax.text(X.min() + 2, 0.5, f'LogReg Accuracy = {score:.3f}', fontsize=11)

# 4. Time Bins Analysis - Group data by time intervals
plt.subplot(2, 3, 4)
# Create time bins (years)
patients_with_second_scan['years_bin'] = pd.cut(
    patients_with_second_scan['months_between_scans'] / 12,
    bins=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    labels=['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']
)

# Aggregate by time bins
time_bin_stats = patients_with_second_scan.groupby('years_bin').agg({
    'self_similarity': 'mean',
    'differentiability': 'mean',
    'overall_correct': 'mean'
}).reset_index()

# Create bar plot for metrics by time bin
sns.barplot(data=time_bin_stats, x='years_bin', y='self_similarity', color='blue', alpha=0.6)
plt.title('Average Self-similarity by Time Interval')
plt.xlabel('Years Between Scans')
plt.ylabel('Average Self-similarity')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 5. Time Bins - Differentiability 
plt.subplot(2, 3, 5)
sns.barplot(data=time_bin_stats, x='years_bin', y='differentiability', color='green', alpha=0.6)
plt.title('Average Differentiability by Time Interval')
plt.xlabel('Years Between Scans')
plt.ylabel('Average Differentiability')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 6. Time Bins - Overall Accuracy
plt.subplot(2, 3, 6)
sns.barplot(data=time_bin_stats, x='years_bin', y='overall_correct', color='orange', alpha=0.6)
plt.title('Average Correct Differentiation by Time Interval')
plt.xlabel('Years Between Scans')
plt.ylabel('Proportion Correctly Differentiated')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('longitudinal_analysis_overview.png', dpi=300)

# Statistical tests and detailed analysis
print("\n--- STATISTICAL ANALYSIS OF LONGITUDINAL METRICS ---\n")

# 1. Linear regression for continuous variables vs time
metrics = ['self_similarity', 'differentiability']
results = []

for metric in metrics:
    # Perform regression
    X = sm.add_constant(patients_with_second_scan['months_between_scans'])
    y = patients_with_second_scan[metric]
    model = sm.OLS(y, X).fit()
    
    # Extract results
    coef = model.params[1]
    p_value = model.pvalues[1]
    r_squared = model.rsquared
    
    results.append({
        'Metric': metric,
        'Coefficient': coef,
        'P-value': p_value,
        'R-squared': r_squared
    })

regression_results = pd.DataFrame(results)
print("Linear Regression Results (Metrics vs Time):")
print(regression_results)
regression_results.to_csv('longitudinal_regression_results.csv', index=False)

# 2. Logistic regression for binary outcomes vs time
if 0 in patients_with_second_scan['overall_correct'].values and 1 in patients_with_second_scan['overall_correct'].values:
    X = sm.add_constant(patients_with_second_scan['months_between_scans'])
    y = patients_with_second_scan['overall_correct']
    logit_model = sm.Logit(y, X).fit(disp=0)
    
    print("\nLogistic Regression Results (Correct Differentiation vs Time):")
    print(f"Coefficient: {logit_model.params[1]:.4f}")
    print(f"P-value: {logit_model.pvalues[1]:.4f}")
    print(f"Pseudo R-squared: {logit_model.prsquared:.4f}")
    
    # Odds ratio
    odds_ratio = np.exp(logit_model.params[1])
    print(f"Odds Ratio (per month): {odds_ratio:.4f}")
    print(f"Odds Ratio (per year): {np.exp(logit_model.params[1]*12):.4f}")

# 3. Time threshold analysis: Find critical time points
print("\n--- TIME THRESHOLD ANALYSIS ---")

# Create function to calculate metrics at different time thresholds
def time_threshold_analysis(data, time_col, metric_cols, thresholds):
    results = []
    
    for threshold in thresholds:
        group1 = data[data[time_col] <= threshold]
        group2 = data[data[time_col] > threshold]
        
        if len(group1) > 1 and len(group2) > 1:  # Ensure both groups have data
            for metric in metric_cols:
                # Calculate mean and std for both groups
                mean1 = group1[metric].mean()
                mean2 = group2[metric].mean()
                
                # T-test
                t_stat, p_value = stats.ttest_ind(group1[metric], group2[metric], equal_var=False)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((group1[metric].var() * (len(group1) - 1) + 
                                    group2[metric].var() * (len(group2) - 1)) / 
                                   (len(group1) + len(group2) - 2))
                
                effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                
                results.append({
                    'Threshold (months)': threshold,
                    'Threshold (years)': threshold / 12,
                    'Metric': metric,
                    'Mean (≤ threshold)': mean1,
                    'Mean (> threshold)': mean2,
                    'Difference': mean2 - mean1,
                    'T-statistic': t_stat,
                    'P-value': p_value,
                    "Cohen's d": effect_size,
                    'N (≤ threshold)': len(group1),
                    'N (> threshold)': len(group2)
                })
    
    return pd.DataFrame(results)

# Define time thresholds to test (in months)
thresholds = [12, 24, 36, 48, 60, 72]  # 1-6 years

# Run threshold analysis
threshold_results = time_threshold_analysis(
    patients_with_second_scan,
    'months_between_scans',
    ['self_similarity', 'differentiability', 'overall_correct'],
    thresholds
)

# Display and save results
print("\nTime Threshold Analysis Results:")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(threshold_results)
threshold_results.to_csv('time_threshold_analysis.csv', index=False)

# 4. Create an ROC curve to evaluate time's predictive power for correct differentiation
if 0 in patients_with_second_scan['overall_correct'].values and 1 in patients_with_second_scan['overall_correct'].values:
    plt.figure(figsize=(10, 8))
    
    # ROC curve for time predicting differentiation
    y_true = patients_with_second_scan['overall_correct']
    y_score = patients_with_second_scan['months_between_scans']
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(2, 1, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Time Between Scans Predicting Correct Differentiation')
    plt.legend(loc="lower right")
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    plt.subplot(2, 1, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve: Time Between Scans Predicting Correct Differentiation')
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig('time_prediction_curves.png', dpi=300)

# 5. Line plots with confidence intervals across time periods
plt.figure(figsize=(18, 10))

# Function to create line plots with confidence intervals
def plot_metric_over_time_periods(data, time_column, metric_column, ax, title, ylabel, color):
    # Create year bins for visualization
    data['year_bin'] = (data[time_column] // 12).astype(int)
    
    # Group by year bin and calculate statistics
    grouped = data.groupby('year_bin')[metric_column].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate confidence interval (95%)
    grouped['ci'] = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
    
    # Plot
    ax.errorbar(grouped['year_bin'], grouped['mean'], 
               yerr=grouped['ci'], fmt='-o', color=color, capsize=5,
               label=f'{metric_column} with 95% CI')
    
    # Add trend line
    if len(grouped) > 1:
        x = grouped['year_bin']
        y = grouped['mean']
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, '--', color='gray', label=f'Trend (slope={m:.3f})')
    
    ax.set_title(title)
    ax.set_xlabel('Years Since First Scan')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add sample size to the plot
    for i, row in grouped.iterrows():
        ax.annotate(f'n={int(row["count"])}', 
                   xy=(row['year_bin'], row['mean']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center')
    
    return grouped

# Create subplots
ax1 = plt.subplot(2, 2, 1)
self_sim_by_year = plot_metric_over_time_periods(
    patients_with_second_scan, 
    'months_between_scans', 
    'self_similarity', 
    ax1,
    'Self-similarity by Years Since First Scan',
    'Self-similarity',
    'blue'
)

ax2 = plt.subplot(2, 2, 2)
diff_by_year = plot_metric_over_time_periods(
    patients_with_second_scan, 
    'months_between_scans', 
    'differentiability', 
    ax2,
    'Differentiability by Years Since First Scan',
    'Differentiability',
    'green'
)

ax3 = plt.subplot(2, 2, 3)
acc_by_year = plot_metric_over_time_periods(
    patients_with_second_scan, 
    'months_between_scans', 
    'overall_correct', 
    ax3,
    'Proportion Correctly Differentiated by Years Since First Scan',
    'Proportion Correct',
    'orange'
)

# 6. Analysis of correctly differentiated vs incorrectly differentiated subjects
ax4 = plt.subplot(2, 2, 4)
sns.boxplot(
    data=patients_with_second_scan,
    x='overall_correct', 
    y='months_between_scans',
    ax=ax4
)
ax4.set_title('Time Between Scans by Differentiation Outcome')
ax4.set_xlabel('Correctly Differentiated (0=No, 1=Yes)')
ax4.set_ylabel('Months Between Scans')

# Add means and p-value to the plot
if 0 in patients_with_second_scan['overall_correct'].values and 1 in patients_with_second_scan['overall_correct'].values:
    correct = patients_with_second_scan[patients_with_second_scan['overall_correct'] == 1]['months_between_scans']
    incorrect = patients_with_second_scan[patients_with_second_scan['overall_correct'] == 0]['months_between_scans']
    
    # T-test
    t_stat, p_value = stats.ttest_ind(correct, incorrect, equal_var=False)
    
    ax4.text(0.5, 0.9, f'T-test p-value: {p_value:.3f}',
            transform=ax4.transAxes, ha='center')
    
    # Add sample sizes
    ax4.text(0, -0.1, f'n={len(incorrect)}', ha='center', transform=ax4.get_xticklabels()[0].get_transform())
    ax4.text(1, -0.1, f'n={len(correct)}', ha='center', transform=ax4.get_xticklabels()[1].get_transform())

plt.tight_layout()
plt.savefig('longitudinal_time_period_analysis.png', dpi=300)

# 7. ANOVA of metrics across time periods
print("\n--- ANOVA ANALYSIS ACROSS TIME PERIODS ---")

# Create a function for ANOVA analysis
def run_anova(data, time_column, metric_columns):
    # Create categorical time variable (years)
    data['year_category'] = pd.cut(
        data[time_column] / 12, 
        bins=[0, 2, 4, 6, 8],
        labels=['0-2 years', '2-4 years', '4-6 years', '6-8 years']
    )
    
    results = []
    
    for metric in metric_columns:
        try:
            # Run ANOVA
            model = ols(f'{metric} ~ C(year_category)', data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Store results
            results.append({
                'Metric': metric,
                'F-value': anova_table['F'][0],
                'P-value': anova_table['PR(>F)'][0],
                'R-squared': model.rsquared
            })
            
            # Print detailed results for this metric
            print(f"\nANOVA Results for {metric}:")
            print(anova_table)
            
            # Calculate and display means for each group
            means = data.groupby('year_category')[metric].mean()
            print("\nGroup Means:")
            print(means)
            
            # Posthoc tests (Tukey's HSD) if ANOVA is significant
            if anova_table['PR(>F)'][0] < 0.05:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                tukey = pairwise_tukeyhsd(data[metric], data['year_category'], alpha=0.05)
                print("\nTukey's HSD Results:")
                print(tukey)
        
        except Exception as e:
            print(f"Error running ANOVA for {metric}: {e}")
    
    return pd.DataFrame(results)

# Run ANOVA
anova_results = run_anova(
    patients_with_second_scan,
    'months_between_scans',
    ['self_similarity', 'differentiability', 'overall_correct']
)

print("\nANOVA Summary:")
print(anova_results)
anova_results.to_csv('longitudinal_anova_results.csv', index=False)

# 8. Persistence of correct/incorrect differentiation over time
persistence_analysis = patients_with_second_scan.copy()

# Create time quartiles
persistence_analysis['time_quartile'] = pd.qcut(
    persistence_analysis['months_between_scans'], 
    q=4, 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

# Create contingency table
contingency_table = pd.crosstab(
    persistence_analysis['time_quartile'],
    persistence_analysis['overall_correct'],
    normalize='index'  # Calculate row percentages
) * 100  # Convert to percentage

print("\nContingency Table (% correctly differentiated by time quartile):")
print(contingency_table)

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(
    pd.crosstab(persistence_analysis['time_quartile'], persistence_analysis['overall_correct'])
)

print(f"\nChi-square test: chi2={chi2:.3f}, p={p:.3f}, dof={dof}")

# 9. Save detailed patient-level data with time metrics
detailed_patient_data = patients_with_second_scan.copy()
detailed_patient_data['second_scan_date'] = [second_scan_dict.get(subj) for subj in detailed_patient_data.index]
detailed_patient_data['years_between_scans'] = detailed_patient_data['months_between_scans'] / 12

# Export
detailed_patient_data.to_csv('patient_longitudinal_metrics.csv')

# 10. Final summary table with correlations between time and all metrics
print("\n--- CORRELATION SUMMARY ---")

correlation_results = []
for col in ['self_similarity', 'differentiability', 'z_diff_col', 'z_diff_row', 'row_correct', 'col_correct', 'overall_correct']:
    if col in patients_with_second_scan.columns:
        corr, p = stats.pearsonr(patients_with_second_scan['months_between_scans'], patients_with_second_scan[col])
        correlation_results.append({
            'Metric': col,
            'Correlation with Time (r)': corr,
            'P-value': p
        })

corr_df = pd.DataFrame(correlation_results)
print(corr_df)
corr_df.to_csv('time_correlation_summary.csv', index=False)

print("\nLongitudinal analysis complete. Results saved to CSV files and plots saved as PNG images.")

# Calculate an overall differentiation accuracy metric per subject
# This combines both row and column accuracy into a comprehensive measure
print(combined_df)
combined_df['diff_accuracy'] = (combined_df['row_correct'] + combined_df['col_correct']) / 2 * 100
print(combined_df)
print(combined_df['diff_accuracy'])

# Add this to the first figure set - create a new plot for differentiation accuracy vs time
plt.figure(figsize=(18, 12))

# 1. Differentiation Accuracy vs Time
plt.subplot(2, 3, 1)
sns.scatterplot(data=patients_with_second_scan, x='months_between_scans', y='diff_accuracy', hue='overall_correct')
plt.title('Differentiation Accuracy vs Months Between Scans')
plt.xlabel('Months Between Scans')
plt.ylabel('Differentiation Accuracy (%)')
plt.grid(True, alpha=0.3)

# Add regression line
x = patients_with_second_scan['months_between_scans']
y = patients_with_second_scan['diff_accuracy']
if len(x) > 0 and len(y) > 0:
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--')
    corr, p_value = stats.pearsonr(x, y)
    plt.text(x.min() + 2, y.min() + 2, f'r = {corr:.3f}, p = {p_value:.3f}', fontsize=10)

# 2-6. Keep the other plots from your original code
# ...

plt.tight_layout()
plt.savefig('longitudinal_analysis_with_diff_accuracy.png', dpi=300)
plt.close()

# Create a dedicated figure for differentiation accuracy analysis
plt.figure(figsize=(18, 10))

# 1. Differentiation Accuracy Regression Analysis
plt.subplot(2, 2, 1)
sns.regplot(x='months_between_scans', y='diff_accuracy', data=patients_with_second_scan, 
           scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Differentiation Accuracy Regression Analysis')
plt.xlabel('Months Between Scans')
plt.ylabel('Differentiation Accuracy (%)')
plt.grid(True, alpha=0.3)

# Add stats to the plot
x = patients_with_second_scan['months_between_scans']
y = patients_with_second_scan['diff_accuracy']
m, b = np.polyfit(x, y, 1)
corr, p_val = stats.pearsonr(x, y)
plt.text(0.05, 0.95, f'Slope: {m:.3f}\nr: {corr:.3f}\np: {p_val:.3f}', 
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.5))

# 2. Differentiation Accuracy by Time Period
plt.subplot(2, 2, 2)
ax2 = plt.gca()
diff_acc_by_year = plot_metric_over_time_periods(
    patients_with_second_scan, 
    'months_between_scans', 
    'diff_accuracy', 
    ax2,
    'Differentiation Accuracy by Years Since First Scan',
    'Differentiation Accuracy (%)',
    'purple'
)

# 3. Box Plot of Differentiation Accuracy by Time Quartile
plt.subplot(2, 2, 3)
# Create time quartiles if not already done
patients_with_second_scan['time_quartile'] = pd.qcut(
    patients_with_second_scan['months_between_scans'], 
    q=4, 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)
sns.boxplot(x='time_quartile', y='diff_accuracy', data=patients_with_second_scan)
plt.title('Differentiation Accuracy by Time Quartile')
plt.xlabel('Time Quartile (Q1=shortest time, Q4=longest time)')
plt.ylabel('Differentiation Accuracy (%)')
plt.grid(True, alpha=0.3)

# Add ANOVA stats
model = ols('diff_accuracy ~ C(time_quartile)', data=patients_with_second_scan).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
f_val = anova_table['F'][0]
p_val = anova_table['PR(>F)'][0]
plt.text(0.5, 0.95, f'ANOVA: F={f_val:.3f}, p={p_val:.3f}', 
         transform=plt.gca().transAxes, fontsize=10, ha='center',
         bbox=dict(facecolor='white', alpha=0.5))

# 4. Differentiation Accuracy by Correct/Incorrect Overall Differentiation
plt.subplot(2, 2, 4)
sns.boxplot(x='overall_correct', y='diff_accuracy', data=patients_with_second_scan)
plt.title('Differentiation Accuracy by Overall Differentiation Result')
plt.xlabel('Correctly Differentiated Overall (0=No, 1=Yes)')
plt.ylabel('Differentiation Accuracy (%)')
plt.grid(True, alpha=0.3)

# Add t-test stats if we have both 0 and 1 values
if 0 in patients_with_second_scan['overall_correct'].values and 1 in patients_with_second_scan['overall_correct'].values:
    correct = patients_with_second_scan[patients_with_second_scan['overall_correct'] == 1]['diff_accuracy']
    incorrect = patients_with_second_scan[patients_with_second_scan['overall_correct'] == 0]['diff_accuracy']
    
    # T-test
    t_stat, p_value = stats.ttest_ind(correct, incorrect, equal_var=False)
    
    plt.text(0.5, 0.95, f'T-test: t={t_stat:.3f}, p={p_value:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, ha='center',
             bbox=dict(facecolor='white', alpha=0.5))
    
    # Add sample sizes
    plt.text(0, -0.1, f'n={len(incorrect)}', ha='center', 
             transform=plt.gca().get_xticklabels()[0].get_transform())
    plt.text(1, -0.1, f'n={len(correct)}', ha='center', 
             transform=plt.gca().get_xticklabels()[1].get_transform())

plt.tight_layout()
plt.savefig('differentiation_accuracy_analysis.png', dpi=300)
plt.close()

# Statistical Analysis for Differentiation Accuracy
print("\n--- STATISTICAL ANALYSIS FOR DIFFERENTIATION ACCURACY ---\n")

# 1. Add differentiation accuracy to regression analysis
X = sm.add_constant(patients_with_second_scan['months_between_scans'])
y = patients_with_second_scan['diff_accuracy']
model = sm.OLS(y, X).fit()

print("Linear Regression Results (Differentiation Accuracy vs Time):")
print(f"Regression coefficient: {model.params[1]:.4f}")
print(f"P-value: {model.pvalues[1]:.4f}")
print(f"R-squared: {model.rsquared:.4f}")

# 2. Add differentiation accuracy to time threshold analysis
diff_acc_threshold_results = time_threshold_analysis(
    patients_with_second_scan,
    'months_between_scans',
    ['diff_accuracy'],
    thresholds
)

print("\nTime Threshold Analysis for Differentiation Accuracy:")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(diff_acc_threshold_results)

# Add differentiation accuracy to ANOVA analysis
diff_acc_anova_results = run_anova(
    patients_with_second_scan,
    'months_between_scans',
    ['diff_accuracy']
)

print("\nANOVA Results for Differentiation Accuracy:")
print(diff_acc_anova_results)

# Create correlation matrix including differentiation accuracy
print("\n--- CORRELATION WITH TIME ---")
metrics_to_correlate = ['self_similarity', 'differentiability', 'diff_accuracy', 
                         'row_correct', 'col_correct', 'overall_correct']

correlation_results = []
for col in metrics_to_correlate:
    if col in patients_with_second_scan.columns:
        corr, p = stats.pearsonr(patients_with_second_scan['months_between_scans'], patients_with_second_scan[col])
        correlation_results.append({
            'Metric': col,
            'Correlation with Time (r)': corr,
            'P-value': p
        })

corr_df = pd.DataFrame(correlation_results)
print(corr_df)
corr_df.to_csv('time_correlation_with_all_metrics.csv', index=False)

# Inter-metric correlation analysis
print("\n--- INTER-METRIC CORRELATIONS ---")
# Create correlation matrix between all metrics
metrics_corr = patients_with_second_scan[metrics_to_correlate].corr()
print(metrics_corr)
metrics_corr.to_csv('inter_metric_correlations.csv')

# Create a heatmap of the correlations
plt.figure(figsize=(10, 8))
sns.heatmap(metrics_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix Between Metrics')
plt.tight_layout()
plt.savefig('inter_metric_correlation_heatmap.png', dpi=300)
plt.close()

# Time-based predictive modeling for differentiation accuracy
# Divide data into early and late timepoints based on median
median_time = patients_with_second_scan['months_between_scans'].median()
early_group = patients_with_second_scan[patients_with_second_scan['months_between_scans'] <= median_time]
late_group = patients_with_second_scan[patients_with_second_scan['months_between_scans'] > median_time]

print("\n--- EARLY VS LATE TIMEPOINT COMPARISON ---")
print(f"Early group (≤ {median_time:.1f} months, n={len(early_group)}) vs Late group (> {median_time:.1f} months, n={len(late_group)})")

metrics_to_compare = ['self_similarity', 'differentiability', 'diff_accuracy', 'overall_correct']
for metric in metrics_to_compare:
    if metric in patients_with_second_scan.columns:
        early_mean = early_group[metric].mean()
        late_mean = late_group[metric].mean()
        
        # T-test
        t_stat, p_value = stats.ttest_ind(early_group[metric], late_group[metric], equal_var=False)
        
        # Effect size
        pooled_std = np.sqrt((early_group[metric].var() * (len(early_group) - 1) + 
                            late_group[metric].var() * (len(late_group) - 1)) / 
                           (len(early_group) + len(late_group) - 2))
        
        effect_size = abs(early_mean - late_mean) / pooled_std if pooled_std > 0 else 0
        
        print(f"\n{metric}:")
        print(f"  Early group mean: {early_mean:.3f}")
        print(f"  Late group mean: {late_mean:.3f}")
        print(f"  Difference: {late_mean - early_mean:.3f}")
        print(f"  T-test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.3f}")

# Visualization of differentiation accuracy change over time with trend analysis
plt.figure(figsize=(12, 8))

# Create scatter plot with locally weighted scatterplot smoothing (LOWESS)
scatter = plt.scatter(patients_with_second_scan['months_between_scans'], 
                     patients_with_second_scan['diff_accuracy'],
                     c=patients_with_second_scan['overall_correct'],
                     cmap='viridis', alpha=0.7)

# Add LOWESS trend line
lowess = sm.nonparametric.lowess
z = lowess(patients_with_second_scan['diff_accuracy'], patients_with_second_scan['months_between_scans'], frac=0.6)
plt.plot(z[:, 0], z[:, 1], 'r-', linewidth=2)

# Add linear trend line
x = patients_with_second_scan['months_between_scans']
y = patients_with_second_scan['diff_accuracy']
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, '--', color='blue', linewidth=1.5, 
         label=f'Linear trend (slope={m:.3f})')

plt.colorbar(scatter, label='Overall Correct Differentiation')
plt.title('Differentiation Accuracy Over Time with Trend Analysis')
plt.xlabel('Months Between Scans')
plt.ylabel('Differentiation Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('diff_accuracy_trend_analysis.png', dpi=300)
plt.close()

# Save comprehensive results
comprehensive_results = patients_with_second_scan[['months_between_scans', 'years_between_scans', 
                                                   'self_similarity', 'differentiability', 'diff_accuracy',
                                                   'row_correct', 'col_correct', 'overall_correct', 
                                                   'time_quartile']]
comprehensive_results.to_csv('comprehensive_longitudinal_results.csv')

print("\nDifferentiation accuracy analysis complete. Results saved to files.")
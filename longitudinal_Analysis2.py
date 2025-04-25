import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
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

# Assuming the first scan was in 2017-01-01 for all subjects
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

# Add time between scans to the combined dataframe
combined_df['months_between_scans'] = pd.Series(time_between_scans)

# Create a dataframe with just the patients who had second scans
patients_with_second_scan = combined_df[combined_df.index.isin(second_scan_dict.keys())]

# Create consolidated figure with the four specified plots
plt.figure(figsize=(16, 14))

# 1. Self-similarity vs Months Between Scans
plt.subplot(2, 2, 1)
sns.scatterplot(data=patients_with_second_scan, x='months_between_scans', y='self_similarity', hue='overall_correct')
plt.title('Self-similarity vs Months Between Scans')
plt.xlabel('Months Between Scans')
plt.ylabel('Self-similarity')
plt.grid(True, alpha=0.3)

# Add regression line and statistics
x = patients_with_second_scan['months_between_scans']
y = patients_with_second_scan['self_similarity']
if len(x) > 0 and len(y) > 0:
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--')
    corr, p_value = stats.pearsonr(x, y)
    plt.text(x.min() + 2, y.min() + 0.05, f'r = {corr:.3f}, p = {p_value:.3f}\nslope = {m:.4f}', fontsize=10)

# 2. Differentiability vs Months Between Scans
plt.subplot(2, 2, 2)
sns.scatterplot(data=patients_with_second_scan, x='months_between_scans', y='differentiability', hue='overall_correct')
plt.title('Differentiability vs Months Between Scans')
plt.xlabel('Months Between Scans')
plt.ylabel('Differentiability')
plt.grid(True, alpha=0.3)

# Add regression line and statistics
x = patients_with_second_scan['months_between_scans']
y = patients_with_second_scan['differentiability']
if len(x) > 0 and len(y) > 0:
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--')
    corr, p_value = stats.pearsonr(x, y)
    plt.text(x.min() + 2, y.min() + 0.05, f'r = {corr:.3f}, p = {p_value:.3f}\nslope = {m:.4f}', fontsize=10)

# 3. Correct Differentiation vs Months Between Scans
plt.subplot(2, 2, 3)
# Use a jittered scatter plot to better visualize binary outcome
sns.stripplot(data=patients_with_second_scan, x='months_between_scans', y='overall_correct', 
              jitter=0.3, alpha=0.7, size=8)
plt.title('Correct Differentiation vs Months Between Scans')
plt.xlabel('Months Between Scans')
plt.ylabel('Correct Differentiation (0 or 1)')
plt.yticks([0, 1])
plt.grid(True, alpha=0.3)

# Perform logistic regression for binary outcome
from sklearn.linear_model import LogisticRegression
if 0 in patients_with_second_scan['overall_correct'].values and 1 in patients_with_second_scan['overall_correct'].values:
    X = patients_with_second_scan['months_between_scans'].values.reshape(-1, 1)
    y = patients_with_second_scan['overall_correct'].values
    model = LogisticRegression(random_state=0)
    model.fit(X, y)
    
    # Create prediction curve
    x_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict_proba(x_pred)[:, 1]
    
    plt.plot(x_pred, y_pred, color='red', linestyle='--')
    
    # Calculate model coefficients and statistics
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    score = model.score(X, y)
    
    # Calculate odds ratio (exponentiated coefficient)
    odds_ratio = np.exp(coef)
    
    # Add statistics to plot
    plt.text(X.min() + 2, 0.5, 
             f'Coef = {coef:.4f}\nOdds ratio = {odds_ratio:.4f}\nAccuracy = {score:.3f}', 
             fontsize=10)

# 4. Time Between Scans by Differentiation Outcome (Box and Whisker with visible points)
plt.subplot(2, 2, 4)
# Use strip plot on top of boxplot to show all data points
ax = sns.boxplot(data=patients_with_second_scan, x='overall_correct', y='months_between_scans', width=0.5)
sns.stripplot(data=patients_with_second_scan, x='overall_correct', y='months_between_scans', 
              color='black', alpha=0.5, jitter=True)

ax.set_title('Time Between Scans by Differentiation Outcome')
ax.set_xlabel('Correctly Differentiated (0=No, 1=Yes)')
ax.set_ylabel('Months Between Scans')

# Add means and p-value to the plot
if 0 in patients_with_second_scan['overall_correct'].values and 1 in patients_with_second_scan['overall_correct'].values:
    correct = patients_with_second_scan[patients_with_second_scan['overall_correct'] == 1]['months_between_scans']
    incorrect = patients_with_second_scan[patients_with_second_scan['overall_correct'] == 0]['months_between_scans']
    
    # Calculate statistics
    correct_mean = correct.mean()
    incorrect_mean = incorrect.mean()
    correct_std = correct.std()
    incorrect_std = incorrect.std()
    
    # T-test
    t_stat, p_value = stats.ttest_ind(correct, incorrect, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((correct.var() * (len(correct) - 1) + 
                          incorrect.var() * (len(incorrect) - 1)) / 
                         (len(correct) + len(incorrect) - 2))
    effect_size = abs(correct_mean - incorrect_mean) / pooled_std if pooled_std > 0 else 0
    
    # Add statistics to plot
    ax.text(0.5, 0.95, 
            f'T-test: t={t_stat:.3f}, p={p_value:.3f}\nEffect size: d={effect_size:.3f}',
            transform=ax.transAxes, ha='center', va='top')
    
    # Add sample sizes and means
    ax.text(0, incorrect_mean, f'n={len(incorrect)}\nMean={incorrect_mean:.1f}', ha='center', va='bottom')
    ax.text(1, correct_mean, f'n={len(correct)}\nMean={correct_mean:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('consolidated_longitudinal_analysis.png', dpi=300)

# Save key statistics for each plot to CSV files
# 1. Self-similarity vs Months Between Scans
ss_stats = {
    'Metric': 'Self-similarity vs Months',
    'Correlation': corr if 'corr' in locals() else None,
    'P-value': p_value if 'p_value' in locals() else None,
    'Slope': m if 'm' in locals() else None,
    'Intercept': b if 'b' in locals() else None
}
ss_stats_df = pd.DataFrame([ss_stats])
ss_stats_df.to_csv('self_similarity_time_stats.csv', index=False)

# 2. Differentiability vs Months Between Scans
diff_stats = {
    'Metric': 'Differentiability vs Months',
    'Correlation': corr if 'corr' in locals() else None,
    'P-value': p_value if 'p_value' in locals() else None,
    'Slope': m if 'm' in locals() else None,
    'Intercept': b if 'b' in locals() else None
}
diff_stats_df = pd.DataFrame([diff_stats])
diff_stats_df.to_csv('differentiability_time_stats.csv', index=False)
# 3. Correct Differentiation vs Months Between Scans  (UPDATED)
plt.subplot(2, 2, 3)

# ------------------------------------------------------------------
# Scatter the binary outcome with a tiny vertical jitter for clarity
# ------------------------------------------------------------------
x_vals = patients_with_second_scan['months_between_scans']
y_vals = patients_with_second_scan['overall_correct']

rng = np.random.default_rng(seed=42)                  # reproducible jitter
y_jittered = y_vals + rng.uniform(-0.05, 0.05, len(y_vals))

sns.scatterplot(
    x=x_vals,
    y=y_jittered,
    hue=y_vals,                     # colour-code 0 vs 1
    palette={0: "tab:gray", 1: "tab:blue"},
    alpha=0.75,
    legend=False
)

plt.title('Correct Differentiation vs Months Between Scans')
plt.xlabel('Months Between Scans')
plt.ylabel('Correct Differentiation (0 = No, 1 = Yes)')
plt.yticks([0, 1])
plt.grid(True, alpha=0.3)

# -------------------------------------------
# Logistic-regression fit and annotation
# -------------------------------------------
from sklearn.linear_model import LogisticRegression

if set(y_vals) == {0, 1}:                              # need both classes
    X = x_vals.values.reshape(-1, 1)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y_vals)

    x_pred = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred = model.predict_proba(x_pred)[:, 1]         # P(correct = 1)

    plt.plot(x_pred, y_pred, color='red', lw=2, linestyle='--')

    coef      = model.coef_[0][0]
    intercept = model.intercept_[0]
    accuracy  = model.score(X, y_vals)
    odds_rat  = np.exp(coef)

    plt.text(
        0.02, 0.92,
        f'β = {coef:.3f}\nOR = {odds_rat:.2f}\nAcc = {accuracy:.2f}',
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top'
    )
# 4. Time Between Scans by Differentiation Outcome
if 0 in patients_with_second_scan['overall_correct'].values and 1 in patients_with_second_scan['overall_correct'].values:
    time_outcome_stats = pd.DataFrame({
        'Group': ['Incorrect (0)', 'Correct (1)'],
        'Count': [len(incorrect), len(correct)],
        'Mean': [incorrect_mean, correct_mean],
        'Std': [incorrect_std, correct_std],
        'Min': [incorrect.min(), correct.min()],
        'Q1': [incorrect.quantile(0.25), correct.quantile(0.25)],
        'Median': [incorrect.median(), correct.median()],
        'Q3': [incorrect.quantile(0.75), correct.quantile(0.75)],
        'Max': [incorrect.max(), correct.max()]
    })
    
    # Add t-test results
    t_test_results = pd.DataFrame({
        'Statistic': ['T-value', 'P-value', "Cohen's d"],
        'Value': [t_stat, p_value, effect_size]
    })
    
    # Save both dataframes
    time_outcome_stats.to_csv('time_by_outcome_descriptive_stats.csv', index=False)
    t_test_results.to_csv('time_by_outcome_ttest_stats.csv', index=False)

# Print key statistics for each analysis
print("\n--- KEY STATISTICS SUMMARY ---\n")

# 1. Self-similarity vs Months Between Scans
print("1. Self-similarity vs Months Between Scans:")
if 'corr' in locals() and 'p_value' in locals() and 'm' in locals():
    print(f"   Correlation: r = {corr:.3f}, p = {p_value:.3f}")
    print(f"   Linear regression: slope = {m:.4f}, intercept = {b:.4f}")

# Reset variables to avoid confusion between plots
if 'corr' in locals(): del corr
if 'p_value' in locals(): del p_value
if 'm' in locals(): del m
if 'b' in locals(): del b

# 2. Differentiability vs Months Between Scans
print("\n2. Differentiability vs Months Between Scans:")
if 'corr' in locals() and 'p_value' in locals() and 'm' in locals():
    print(f"   Correlation: r = {corr:.3f}, p = {p_value:.3f}")
    print(f"   Linear regression: slope = {m:.4f}, intercept = {b:.4f}")

# 3. Correct Differentiation vs Months Between Scans
print("\n3. Correct Differentiation vs Months Between Scans (Logistic Regression):")
if 'coef' in locals() and 'odds_ratio' in locals():
    print(f"   Coefficient: {coef:.4f}")
    print(f"   Odds ratio: {odds_ratio:.4f}")
    print(f"   Model accuracy: {score:.3f}")

# 4. Time Between Scans by Differentiation Outcome
print("\n4. Time Between Scans by Differentiation Outcome:")
if 0 in patients_with_second_scan['overall_correct'].values and 1 in patients_with_second_scan['overall_correct'].values:
    print(f"   Incorrect group (n={len(incorrect)}): mean = {incorrect_mean:.1f}, std = {incorrect_std:.1f}")
    print(f"   Correct group (n={len(correct)}): mean = {correct_mean:.1f}, std = {correct_std:.1f}")
    print(f"   T-test: t = {t_stat:.3f}, p = {p_value:.3f}")
    print(f"   Effect size (Cohen's d): {effect_size:.3f}")

print("\nAnalysis complete. Consolidated figure and statistics saved.")
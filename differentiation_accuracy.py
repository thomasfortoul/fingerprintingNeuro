# Import additional libraries needed for statistical analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.patches import Patch

# Load the previously saved data
accuracy_df = pd.read_csv('subject_differentiation_accuracy.csv', index_col=0)
diff_df = pd.read_csv('all_differentiability_scores.csv', index_col=0)
combined_df = pd.read_csv('combined_differentiability_accuracy.csv', index_col=0)

# 1. Enhanced Statistical Analysis
# Calculate comprehensive statistics for each group
metrics = ['row_correct', 'col_correct', 'overall_correct']
group_stats = {}

for group in ['healthy', 'patient']:
    group_data = accuracy_df[accuracy_df['group'] == group][metrics]
    
    # Calculate statistics
    group_stats[group] = {
        'mean': group_data.mean() * 100,
        'std': group_data.std() * 100,
        'median': group_data.median() * 100,
        'min': group_data.min() * 100,
        'max': group_data.max() * 100,
        'count': len(group_data)
    }

# Create a comprehensive statistical summary
stat_summary = pd.DataFrame()
for group, stats_dict in group_stats.items():
    for stat_name, stat_values in stats_dict.items():
        if isinstance(stat_values, pd.Series):
            for col, val in stat_values.items():
                stat_summary.loc[f"{group}_{col}", stat_name] = val
        else:
            stat_summary.loc[f"{group}", stat_name] = stat_values

# Save the statistical summary
stat_summary.to_csv('differentiation_statistical_summary.csv')
print("Statistical Summary:")
print(stat_summary)

# 2. Box and Whisker Plot for Accuracy Metrics
plt.figure(figsize=(12, 7))
accuracy_melt = pd.melt(accuracy_df, 
                       id_vars=['group'], 
                       value_vars=metrics,
                       var_name='Metric', 
                       value_name='Accuracy')
accuracy_melt['Accuracy'] = accuracy_melt['Accuracy'] * 100  # Convert to percentage

# Create box plot
sns.boxplot(x='Metric', y='Accuracy', hue='group', data=accuracy_melt, 
            palette={'healthy': 'skyblue', 'patient': 'salmon'})

# Add individual data points
sns.stripplot(x='Metric', y='Accuracy', hue='group', data=accuracy_melt, 
              dodge=True, alpha=0.5, jitter=True, size=6, 
              palette={'healthy': 'darkblue', 'patient': 'darkred'})

# Rename the x-axis labels for clarity
plt.xticks(ticks=[0, 1, 2], labels=['Run1→Run2', 'Run2→Run1', 'Overall'])
plt.xlabel('Differentiation Direction', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Distribution of Differentiation Accuracy by Group', fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Group')
plt.tight_layout()
plt.savefig('differentiation_boxplot.png', dpi=300)
plt.close()

# 3. Kernel Density Estimation (KDE) for Differentiability Scores
plt.figure(figsize=(12, 6))
# Plot KDE curves
sns.kdeplot(data=combined_df[combined_df['group'] == 'healthy'], x='differentiability', 
            fill=True, alpha=0.5, label='CU', color='skyblue')
sns.kdeplot(data=combined_df[combined_df['group'] == 'patient'], x='differentiability', 
            fill=True, alpha=0.5, label='AD', color='salmon')

# Add median lines
healthy_median = combined_df[combined_df['group'] == 'healthy']['differentiability'].median()
patient_median = combined_df[combined_df['group'] == 'patient']['differentiability'].median()

plt.axvline(healthy_median, color='blue', linestyle='--', label=f'CU Median ({healthy_median:.3f})')
plt.axvline(patient_median, color='red', linestyle='--', label=f'AD Median ({patient_median:.3f})')

plt.xlabel('Differentiability Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Differentiability Scores by Group', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('differentiability_kde.png', dpi=300)
plt.close()

# 4. Statistical Tests for Group Differences
# Perform t-tests to compare groups
ttest_results = {}
for metric in metrics + ['differentiability']:
    healthy_data = combined_df[combined_df['group'] == 'healthy'][metric]
    patient_data = combined_df[combined_df['group'] == 'patient'][metric]
    
    if metric in metrics:
        # Convert binary to percentage for accuracy metrics
        healthy_data = healthy_data * 100
        patient_data = patient_data * 100
    
    t_stat, p_value = stats.ttest_ind(healthy_data, patient_data, equal_var=False)
    ttest_results[metric] = {
        'healthy_mean': healthy_data.mean(),
        'patient_mean': patient_data.mean(),
        'mean_diff': healthy_data.mean() - patient_data.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Convert to DataFrame
ttest_df = pd.DataFrame(ttest_results).T
ttest_df.to_csv('group_difference_tests.csv')
print("\nStatistical Test Results for Group Differences:")
print(ttest_df)

# 5. Correlation Analysis
# Calculate correlations between differentiability and accuracy metrics
correlation_results = {}
for group in ['all', 'healthy', 'patient']:
    if group == 'all':
        group_data = combined_df
    else:
        group_data = combined_df[combined_df['group'] == group]
    
    correlations = {}
    for metric in metrics:
        corr, p_value = stats.pearsonr(group_data['differentiability'], group_data[metric])
        correlations[metric] = {
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    correlation_results[group] = correlations

# Create correlation summary DataFrame
corr_summary = pd.DataFrame()
for group, metrics_dict in correlation_results.items():
    for metric, values in metrics_dict.items():
        for stat_name, stat_val in values.items():
            corr_summary.loc[f"{group}_{metric}", stat_name] = stat_val

corr_summary.to_csv('differentiability_accuracy_correlations.csv')
print("\nCorrelation Analysis Results:")
print(corr_summary)

# 6. Visualize Group-level Accuracy with Error Bars
plt.figure(figsize=(14, 8))
bar_width = 0.35
index = np.arange(len(metrics))

# Calculate means and standard errors for each group
healthy_means = [group_stats['healthy']['mean'][m] for m in metrics]
patient_means = [group_stats['patient']['mean'][m] for m in metrics]
healthy_stderr = [group_stats['healthy']['std'][m] / np.sqrt(group_stats['healthy']['count']) for m in metrics]
patient_stderr = [group_stats['patient']['std'][m] / np.sqrt(group_stats['patient']['count']) for m in metrics]

# Create bar plot with error bars
plt.bar(index - bar_width/2, healthy_means, bar_width, label='CU',
        color='skyblue', yerr=healthy_stderr, capsize=7, alpha=0.8)
plt.bar(index + bar_width/2, patient_means, bar_width, label='AD',
        color='salmon', yerr=patient_stderr, capsize=7, alpha=0.8)

# Add p-value stars
for i, metric in enumerate(metrics):
    p_value = ttest_results[metric]['p_value']
    if p_value < 0.001:
        sig_symbol = '***'
    elif p_value < 0.01:
        sig_symbol = '**'
    elif p_value < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    
    y_pos = max(healthy_means[i], patient_means[i]) + 5
    plt.text(i, y_pos, sig_symbol, ha='center', fontsize=12)

plt.xlabel('Differentiation Direction', fontsize=12)
plt.ylabel('Differentiation Accuracy (%)', fontsize=12)
plt.title('Group Differentiation Accuracy with Standard Error', fontsize=14)
plt.xticks(index, ['Run1→Run2', 'Run2→Run1', 'Overall'])
plt.legend()

# Add significance legend
legend_elements = [
    Patch(facecolor='white', edgecolor='black', label='* p < 0.05'),
    Patch(facecolor='white', edgecolor='black', label='** p < 0.01'),
    Patch(facecolor='white', edgecolor='black', label='*** p < 0.001'),
    Patch(facecolor='white', edgecolor='black', label='ns: not significant')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 105)  # Set y-axis limit to accommodate p-value symbols
plt.tight_layout()
plt.savefig('group_accuracy_comparison.png', dpi=300)
plt.close()

# 7. Scatter Plot with Regression Line for Differentiability vs Accuracy
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.figure(figsize=(12, 10))

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
axes = axes.flatten()

# Remap group labels in advance
combined_df['Group'] = combined_df['group'].map({'healthy': 'CU', 'patient': 'AD'})
combined_df['accuracy_mean'] = combined_df[metrics].mean(axis=1) * 100

# Updated color scheme
colors = {'CU': '#1f77b4', 'AD': '#ff7f0e'}
metric_names = {'row_correct': 'Run1→Run2', 'col_correct': 'Run2→Run1', 'overall_correct': 'Overall'}

# Plot each metric in a separate subplot
for i, metric in enumerate(metrics):
    ax = axes[i]
    combined_df[f'{metric}_pct'] = combined_df[metric] * 100

    for group in ['CU', 'AD']:
        group_data = combined_df[combined_df['Group'] == group]

        # Jitter for binary classification display
        jitter = np.random.normal(0, 2, size=len(group_data))
        y_vals = group_data[f'{metric}_pct'] + jitter

        # Scatter points
        ax.scatter(group_data['differentiability'], y_vals,
                   alpha=0.7, color=colors[group], label=group, s=80)

        # Regression line
        x = group_data['differentiability']
        y = group_data[metric]
        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            ax.plot(np.sort(x), 100 * (slope * np.sort(x) + intercept),
                    color=colors[group], linestyle='--',
                    label=f'{group}: r={r_value:.2f}, p={p_value:.3f}')

    ax.set_xlabel('Differentiability Score', fontsize=12)
    ax.set_ylabel(f'{metric_names[metric]} Differentiation Accuracy (%)', fontsize=12)
    ax.set_title(f'Differentiability vs {metric_names[metric]} Accuracy', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    ax.set_ylim(-5, 105)

# Fourth plot: Overall average accuracy vs differentiability
ax = axes[3]
for group in ['CU', 'AD']:
    group_data = combined_df[combined_df['Group'] == group]

    ax.scatter(group_data['differentiability'], group_data['accuracy_mean'],
               alpha=0.7, color=colors[group], label=group, s=80)

    x = group_data['differentiability']
    y = group_data[metrics].mean(axis=1)
    if len(x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        ax.plot(np.sort(x), 100 * (slope * np.sort(x) + intercept),
                color=colors[group], linestyle='--',
                label=f'{group}: r={r_value:.2f}, p={p_value:.3f}')

ax.set_xlabel('Differentiability Score', fontsize=12)
ax.set_ylabel('Average Accuracy (%)', fontsize=12)
ax.set_title('Differentiability vs Average Accuracy', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='lower right')
ax.set_ylim(-5, 105)

plt.tight_layout()
plt.savefig('differentiability_vs_accuracy_detailed.png', dpi=300)
plt.close()




##### SIGMOID
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import log_loss

# Load the combined data
combined_df = pd.read_csv('combined_differentiability_accuracy.csv', index_col=0)

# Prepare data for logistic regression (using overall_correct as target)
X = combined_df['differentiability'].values.reshape(-1, 1)
y = combined_df['overall_correct'].values

# Fit logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X, y)

# Get predictions and probabilities
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Create a grid of points for the sigmoid curve
x_range = np.linspace(min(X) - 0.1, max(X) + 0.1, 1000).reshape(-1, 1)
sigmoid_curve = model.predict_proba(x_range)[:, 1]
# Map group labels
combined_df['Group'] = combined_df['group'].map({'healthy': 'CU', 'patient': 'AD'})

# Plot the sigmoid function
plt.figure(figsize=(12, 8))

# Plot actual data points, color-coded by group with jitter
jitter = np.random.normal(0, 0.02, size=len(y))
jitter_y = y + jitter

for group, color in zip(['CU', 'AD'], ['skyblue', 'salmon']):
    group_mask = combined_df['Group'] == group
    plt.scatter(X[group_mask], jitter_y[group_mask], 
                c=color, s=80, alpha=0.7, edgecolors='k',
                label=f'{group} (jittered)')

# Plot the fitted sigmoid curve
plt.plot(x_range, sigmoid_curve, 'b-', linewidth=2, label='Sigmoid function')

# Add vertical decision boundary line
threshold = -model.intercept_[0] / model.coef_[0][0]
plt.axvline(x=threshold, color='red', linestyle='--', 
            label=f'Decision boundary: {threshold:.3f}')

# Final plot formatting
plt.xlabel('Differentiability Score', fontsize=14)
plt.ylabel('Probability of Successful Differentiation', fontsize=14)
plt.title('Logistic Regression: Differentiability vs Differentiation Accuracy', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig("logistic_sigmoid_plot.png", dpi=300)
plt.close()

# Add groups and predictions to legend
handles, labels = plt.gca().get_legend_handles_labels()
healthy_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', 
                          markersize=10, label='CU')
patient_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', 
                          markersize=10, label='AD')
plt.legend(handles=handles + [healthy_patch, patient_patch], fontsize=12)

plt.tight_layout()
plt.savefig('sigmoid_classifier.png', dpi=300)
plt.close()

# Additional analyses with statsmodels for detailed statistics
# Fit the model using statsmodels for more detailed statistics
X_sm = sm.add_constant(X)  # Add intercept term
logit_model = Logit(y, X_sm)
result = logit_model.fit()

# Calculate various metrics
accuracy = accuracy_score(y, y_pred)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('sigmoid_classifier_roc.png', dpi=300)
plt.close()

# Plot calibration curve (reliability diagram)
plt.figure(figsize=(10, 8))
fraction_of_positives, mean_predicted_value = calibration_curve(y, y_prob, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability", fontsize=14)
plt.ylabel("Fraction of positives", fontsize=14)
plt.title("Calibration curve (Reliability diagram)", fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('sigmoid_classifier_calibration.png', dpi=300)
plt.close()

# Calculate log loss
log_loss_value = log_loss(y, y_prob)

# Group-wise analysis
group_metrics = {}
for group in ['healthy', 'patient']:
    group_data = combined_df[combined_df['group'] == group]
    group_X = group_data['differentiability'].values.reshape(-1, 1)
    group_y = group_data['overall_correct'].values
    
    if len(np.unique(group_y)) < 2:
        # Skip if all observations are the same class
        group_metrics[group] = "Insufficient class variation for logistic regression"
        continue
    
    group_model = LogisticRegression(random_state=42)
    group_model.fit(group_X, group_y)
    
    group_y_pred = group_model.predict(group_X)
    group_y_prob = group_model.predict_proba(group_X)[:, 1]
    
    group_X_sm = sm.add_constant(group_X)
    try:
        group_logit = Logit(group_y, group_X_sm)
        group_result = group_logit.fit(disp=0)
        group_metrics[group] = {
            'coef': group_model.coef_[0][0],
            'intercept': group_model.intercept_[0],
            'threshold': -group_model.intercept_[0] / group_model.coef_[0][0] if group_model.coef_[0][0] != 0 else "N/A",
            'accuracy': accuracy_score(group_y, group_y_pred),
            'log_loss': log_loss(group_y, group_y_prob, eps=1e-15),
            'p_values': group_result.pvalues.tolist() if hasattr(group_result, 'pvalues') else None,
            'conf_intervals': group_result.conf_int().values.tolist() if hasattr(group_result, 'conf_int') else None
        }
    except:
        group_metrics[group] = "Model fitting failed"

# Print detailed statistics
print("\n===== LOGISTIC REGRESSION MODEL PARAMETERS =====")
print(f"Coefficient: {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Decision threshold (differentiability score): {threshold:.4f}")
print(f"Log-likelihood: {result.llf:.4f}")
print(f"Pseudo R-squared (McFadden's): {result.prsquared:.4f}")
print(f"AIC: {result.aic:.4f}")
print(f"BIC: {result.bic:.4f}")
print(f"Log Loss: {log_loss_value:.4f}")

print("\n===== CONFIDENCE INTERVALS =====")
print(result.conf_int())

print("\n===== P-VALUES =====")
print(result.pvalues)

print("\n===== CLASSIFICATION PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
print(f"Specificity (True Negative Rate): {specificity:.4f}")
print(f"Precision (Positive Predictive Value): {precision:.4f}")
print(f"Negative Predictive Value: {npv:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y, y_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y, y_pred))

print("\n===== GROUP-WISE ANALYSIS =====")
for group, metrics in group_metrics.items():
    print(f"\n{group.upper()} GROUP:")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {metrics}")

print("\n===== STATSMODELS SUMMARY =====")
print(result.summary())

# Save all results to a file
with open('sigmoid_classifier_results.txt', 'w') as f:
    f.write("===== LOGISTIC REGRESSION MODEL PARAMETERS =====\n")
    f.write(f"Coefficient: {model.coef_[0][0]:.4f}\n")
    f.write(f"Intercept: {model.intercept_[0]:.4f}\n")
    f.write(f"Decision threshold (differentiability score): {threshold:.4f}\n")
    f.write(f"Log-likelihood: {result.llf:.4f}\n")
    f.write(f"Pseudo R-squared (McFadden's): {result.prsquared:.4f}\n")
    f.write(f"AIC: {result.aic:.4f}\n")
    f.write(f"BIC: {result.bic:.4f}\n")
    f.write(f"Log Loss: {log_loss_value:.4f}\n\n")
    
    f.write("===== CONFIDENCE INTERVALS =====\n")
    f.write(str(result.conf_int()) + "\n\n")
    
    f.write("===== P-VALUES =====\n")
    f.write(str(result.pvalues) + "\n\n")
    
    f.write("===== CLASSIFICATION PERFORMANCE =====\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Sensitivity (True Positive Rate): {sensitivity:.4f}\n")
    f.write(f"Specificity (True Negative Rate): {specificity:.4f}\n")
    f.write(f"Precision (Positive Predictive Value): {precision:.4f}\n")
    f.write(f"Negative Predictive Value: {npv:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"AUC-ROC: {roc_auc:.4f}\n\n")
    
    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(classification_report(y, y_pred) + "\n\n")
    
    f.write("===== CONFUSION MATRIX =====\n")
    f.write(str(confusion_matrix(y, y_pred)) + "\n\n")
    
    f.write("===== GROUP-WISE ANALYSIS =====\n")
    for group, metrics in group_metrics.items():
        f.write(f"\n{group.upper()} GROUP:\n")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
        else:
            f.write(f"  {metrics}\n")
    
    f.write("\n===== STATSMODELS SUMMARY =====\n")
    f.write(str(result.summary()))

print("\nAll results saved to 'sigmoid_classifier_results.txt'")
import numpy as np
from scipy import stats

# Sample dataset: exam scores
scores = [70, 85, 90, 65, 78, 92, 88, 75, 80, 95]

# Descriptive statistics
mean_score = np.mean(scores)
median_score = np.median(scores)
mode_score = stats.mode(scores)[0][0]
std_dev = np.std(scores)
variance = np.var(scores)

print("Mean:", mean_score)
print("Median:", median_score)
print("Mode:", mode_score)
print("Std Dev:", std_dev)
print("Variance:", variance)

# Inferential statistics: 95% confidence interval
conf_int = stats.t.interval(alpha=0.95, df=len(scores)-1, loc=mean_score, scale=stats.sem(scores))
print("95% Confidence Interval:", conf_int)
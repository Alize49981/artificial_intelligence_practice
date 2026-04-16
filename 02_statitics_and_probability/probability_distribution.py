import numpy as np
from scipy.stats import bernoulli, binom, norm

# Bernoulli distribution: coin toss
p = 0.5  # probability of success
coin = bernoulli(p)
print("P(Heads) =", coin.pmf(1))  # PMF = probability mass function

# Binomial distribution: 3 heads in 5 tosses
binom_dist = binom(n=5, p=0.5)
print("P(3 heads) =", binom_dist.pmf(3))

# Normal distribution: mean=0, std=1
normal_dist = norm(loc=0, scale=1)
print("P(X ≤ 1) =", normal_dist.cdf(1))  # CDF = probability X ≤ 1
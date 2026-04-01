import numpy as np

# Discrete random variable: number of heads in 3 coin tosses
outcomes = [0,1,2,3]
probabilities = [0.125, 0.375, 0.375, 0.125]

# Expected value
expected_value = np.dot(outcomes, probabilities)
print("Expected Value:", expected_value)

# Variance
variance = np.dot((np.array(outcomes)-expected_value)**2, probabilities)
print("Variance:", variance)

#Naive Bayes Probability
# Given probabilities
P_Spam = 0.2
P_BuyNow_given_Spam = 0.5
P_BuyNow = 0.1

# Bayes theorem
P_Spam_given_BuyNow = (P_BuyNow_given_Spam * P_Spam) / P_BuyNow
print("P(Spam | 'Buy now'):", P_Spam_given_BuyNow)
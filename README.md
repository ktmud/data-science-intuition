# DS Intuition - Data Science Interview Questions

Build your intuition on important Data Science/Statistics/Machine Learning concepts. Use this material as a study guide for interviews or exams.

All answers here are written to be a quick response to interview questions so that you can impress them with your intuitive understanding of the concepts.

----

## Statistics


### What is the Central Limit Theorem and why is it important?

Given a sufficiently large sample from a population with finite variance, the sum/mean of the samples would tend to a normal distribution, regardless of the distribution of the population.

It is important because it simplifies problems in statistics by allowing us to work on an approximately normal distribution. 

### What is hypothesis testing?

We hypothesis there is no difference between two groups, and call it the "null hypothesis", then we look for evidence to reject the null hypythosis.

The evidence is based on how likely we will see the same sample statistics again if the null hypothesis is true.

### What P-value?

P-value is the probablity of observing data at least as favorable to the alternative hypothesis as our current data, when null hypothesis is true.

**In the context of A/B testing:** P-value is the probablity of seeing experiment group being different to control group to the same or more extreme extent as showing in the data, when there is actually no difference between control and experiment group.

**In the context of linear regression:** Each term has a p-value, where the null hypothesis is that the coefficient should be 0.

### What is Statistical Power?


### What is Confidence Interval?

For a 95% confidence interval, if we repeat the same sampling and estimation process 100 times, the confidence interval would include the true parameter 95 of the times.

**General notes:** a lot of concepts in frequentist inference involves "take infinite samples" or "read the same measurement in the long run". This is directly related to the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem). You should always call out the fact that the confidence of our estimation is based on a repetitive sampling process.

### What is R-squared?

R-sqaured tells how much the variance of the target outcome is explained by the model. The higher R-squared, the better the model explains the data.

<p align="center"><img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B130%7DR%5E2%20%3D%201%20-%20SSE/SST%2C"></p>

where SSE is the squared sum of the error terms (i.e., sum of squares of residuals, predict value - true value), SST is the squared sum of the data variance (total sum of squares, data points - sample mean).

SST is total variance, SSE is remaining unexplained variance.

More features fit the data better, therefore will see smaller SSE. It's better to use the adjusted R-squared,

<p align="center"><img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B130%7D%5Cbar%7BR%7D%5E2%20%3D%201-%5Cfrac%7BSS_%7Bres%7D/df_e%7D%7BSS_%7Btot%7D/df_t%7D%20%3D%20R%5E2%20-%20%281%20-%20R%5E2%29%5Cfrac%7Bp%7D%7Bn-p-1%7D%2C"></p>

which penalizes more complex models.

### What are the assumptions required for linear regression?


### What is a statistical interaction?


### What is selection bias?


### What is an example of a dataset with a non-Gaussian distribution?

### What is the difference between frequentist statistics and Bayesian statistics?

In frequentist statistics, parameters are fixed constants. We calculate the likelihood ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DP%28X%7C%5Ctheta%29), the probability of seeing the obsered data points given the parameter of interest.

In Bayesian statistics, parameters are a random variable with certain distribution. We capture our priori uncertainty about the parameter in the form of a prior distribution, ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DP%28%5Ctheta%29), then update our belief according to the data we observed, so to get a posterior distribution of the parameter, ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DP%28%5Ctheta%7CX%29). Bayesian' Theorem tells us ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DP%28%5Ctheta%7CX%29%20%5Cpropto%20P%28X%7C%5Ctheta%29%20%5Ccdot%20P%28%5Ctheta%29).

----

## Machine Learning

### Explain the relationship between type I Error, type II error, precision, recall, sensitivity and specificity.

- Type I error: False Positive
  - Incorrectly reject the null hypothesis when null is true
  - Type I error rate = false positive rate = significant level = ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5Calpha) = 1 - specificity
  - Perfect specificity gives zero Type I error.
- Type II error: False Negative
  - Failed to reject the null when null is false (alternative hypothesis is true）
  - Type II error rate = false negative rate = ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5Cbeta)
  - Power = sensitivity = recall = ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D1%20-%20%5Cbeta) = the probaility of detecting the effect when there is indeed an effect.
  - Perfect sensitivity gives zero Type II error.



### What's the difference between normalization and standardization?

Normalization "normalize" values to a range of [0, 1], by substract the minimal value then divide by value range.

<p align="center"><img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B130%7Dx%27%20%3D%20%5Cfrac%7Bx%20-%20x_%7B%5Cmin%7D%7D%7Bx_%7B%5Cmax%7D%20-%20x_%7B%5Cmin%7D%7D"></p>

Standardization "standardize" values to multiples of standard deviation by removing the mean then divide by standard deviation.

<p align="center"><img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B130%7Dx%27%20%3D%20%5Cfrac%7Bx%20-%20%5Cmu%7D%7B%5Csigma%7D"></p>

### What are the methods to reduce model complexity in linear regression?

Reducing model complexity is equivalent to introducing sparsity when there are too many features. We do this mostly for combating overfitting. Possible methods include:

- **Lasso regression**: L1 regularization, large ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5Clambda) would make some coefficients to be 0.
- **Manually select** features using expert knowledge.
- **Univariate selection**: only select features whose correlation coefficient with the target variable exceed certain threshold.
- **Step-wise methods**: forward selection or backward selection.

Lasso is prefered because it can be automated, considers all features simultaneously, and can be controlled via lambda.

### What's the difference between forward selection and backward selection?


### Compare the advantages and disadvantages of different ML methods.


Linear Regression

**Pros:**


**Cons:**

### What is PCA?

Principal Component Analysis: find k vectors (principal components) onto which to project n-dimensional data, so as to minimize the projection error.

Steps to compute PCA:
1. Do mean normalization on the data.
2. Computate covariance matrix ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5CSigma).
3. Compute eigenvectors with Singular Value Decomposition: ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5CSigma%20%3D%20USV%27).
4. The first ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7Dk) columns of matrix ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DU) is the principal components.
5. ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7Dz%20%3D%20U_%7Breduced%7D%27%20x) is the reduced dimensions.



## Business Case Study

## Probability and Brain Teaser

### How to get fair results from a biased coin?

John von Neumann gave the following procedure:

1. Toss the coin twice.
2. If the results match, start over, forgetting both results.
3. If the results differ, use the first result, forgetting the second.

This is because the probability of getting head then tail is the same as getting tail then head. By excluding the other two outcomes of two indepedent tosses---getting both heads and getting both tails, we are left with only two outcomes with equal probability.

### Pick 3 cards from a deck of cards labeled from 1 to 100, what’s the probability of getting Card 1 < Card 2 < Card 3?

The three cards are equivalent to 3 random numbers from [1, 100] without replacement. Getting Card 1 < Card 2 < Card 3 is equivalent to getting the numbers in a specific order. There are in total ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DP%283%2C%203%29%20%3D%206) ways of arranging 3 numbers. Therefore the probability of Card 1 < Card 2 < Card 3 is ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D1/6).

### Toss 3 fair dices one by one, get 3 numbers x, y, z, what is the probability of x < y < z?

Event space: ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D6%20%5Ctimes%206%20%5Ctimes%206%20%3D%20216). Two of x, y, z are equal: ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D3%20%5Ctimes%206%20%5Ctimes%205); all three are equal: ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D6).

The remaning ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D216%20-%203%20%5Ctimes%206%20%5Ctimes%205%20-%206%20%3D%20120) outcomes  are basically arranging 3 different numbers, all from the same discrete uniform distribution ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5B1%2C%206%5D), in different orders. Similar to previous question, there are ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D120/6%20%3D%2020) ways of getting ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7Dx%20%3C%20y%20%3C%20z). The probability is ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D20%20/%20216%20%3D%209.26%5C%25).




---

## Credit

Resources consulted while creating this document:

- Interpretable Machine Learning, https://christophm.github.io/interpretable-ml-book/
- An Introduction to Statistical Learning, http://www-bcf.usc.edu/~gareth/ISL/


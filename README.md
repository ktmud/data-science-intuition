# DS Intuition - Data Science Interview Questions

Build your intuition on important Data Science/Statistics/Machine Learning concepts. Use this material as a study guide for interviews or exams.

All answers here are written to be a quick response to interview questions so that you can impress them with your intuitive understanding of the concepts.

<!--ts-->
   * [DS Intuition - Data Science Interview Questions](#ds-intuition---data-science-interview-questions)
   * [Statistics and Inference](#statistics-and-inference)
      * [Basic Concepts](#basic-concepts)
         * [What is the Central Limit Theorem and why is it important?](#what-is-the-central-limit-theorem-and-why-is-it-important)
         * [What is P-value?](#what-is-p-value)
         * [What is Statistical Power?](#what-is-statistical-power)
         * [What is Confidence Interval?](#what-is-confidence-interval)
         * [What is a statistical interaction?](#what-is-a-statistical-interaction)
         * [What is selection bias?](#what-is-selection-bias)
      * [Hypothesis Test](#hypothesis-test)
         * [What is hypothesis testing?](#what-is-hypothesis-testing)
         * [What is the difference between t-test, z-test, f-test, and Chi-squared test?](#what-is-the-difference-between-t-test-z-test-f-test-and-chi-squared-test)
         * [What is paired samples t-test, and how is it different from unpaired t-test?](#what-is-paired-samples-t-test-and-how-is-it-different-from-unpaired-t-test)
      * [Experiment Design](#experiment-design)
         * [How do you decide when to stop an experiment?](#how-do-you-decide-when-to-stop-an-experiment)
      * [Bayesian Statistics](#bayesian-statistics)
         * [What is the difference between frequentist statistics and Bayesian statistics?](#what-is-the-difference-between-frequentist-statistics-and-bayesian-statistics)
      * [Linear Regression](#linear-regression)
         * [What are the assumptions required for linear regression?](#what-are-the-assumptions-required-for-linear-regression)
         * [What is R-squared?](#what-is-r-squared)
         * [How to deal with multicollinearity in linear regression?](#how-to-deal-with-multicollinearity-in-linear-regression)
         * [What is the relationship between Multiple Regression, General Linear Model and Generalized Linear Model?](#what-is-the-relationship-between-multiple-regression-general-linear-model-and-generalized-linear-model)
      * [Advanced Statistics](#advanced-statistics)
         * [What is Propensity Score Matching?](#what-is-propensity-score-matching)
   * [Machine Learning](#machine-learning)
      * [Metrics](#metrics)
         * [Explain the relationship between type I Error, type II error, precision, recall, sensitivity and specificity.](#explain-the-relationship-between-type-i-error-type-ii-error-precision-recall-sensitivity-and-specificity)
         * [What is ROC curve?](#what-is-roc-curve)
         * [What is AUC?](#what-is-auc)
      * [Modelling](#modelling)
         * [Compare the advantages and disadvantages of different ML methods.](#compare-the-advantages-and-disadvantages-of-different-ml-methods)
            * [Linear Regression](#linear-regression-1)
            * [Decision Tree](#decision-tree)
            * [Bagging (Random Forest)](#bagging-random-forest)
            * [Boosting (Gradient Boosting Tree, AdaBoosting)](#boosting-gradient-boosting-tree-adaboosting)
            * [Neutral Networks](#neutral-networks)
         * [What is the difference between normalization and standardization?](#what-is-the-difference-between-normalization-and-standardization)
         * [What are the methods to reduce model complexity in linear regression?](#what-are-the-methods-to-reduce-model-complexity-in-linear-regression)
         * [What's the difference between forward selection and backward selection?](#whats-the-difference-between-forward-selection-and-backward-selection)
         * [What are the methods for dimension reduction?](#what-are-the-methods-for-dimension-reduction)
         * [What is PCA?](#what-is-pca)
      * [Neutral Networks / Deep Learning](#neutral-networks--deep-learning)
         * [What are the techniques to reduce overfitting in Deep Learning?](#what-are-the-techniques-to-reduce-overfitting-in-deep-learning)
      * [What activation functions do you know?](#what-activation-functions-do-you-know)
   * [Product/Business](#productbusiness)
   * [SQL](#sql)
      * [Concepts](#concepts)
         * [What is the difference between ON clause and WHERE clause in joins?](#what-is-the-difference-between-on-clause-and-where-clause-in-joins)
      * [Coding Excercise](#coding-excercise)
   * [Brain Teaser](#brain-teaser)
      * [Probability](#probability)
         * [How to get fair results from a biased coin?](#how-to-get-fair-results-from-a-biased-coin)
         * [Pick 3 cards from a deck of cards labeled from 1 to 100, what’s the probability of getting Card 1 &lt; Card 2 &lt; Card 3?](#pick-3-cards-from-a-deck-of-cards-labeled-from-1-to-100-whats-the-probability-of-getting-card-1--card-2--card-3)
         * [Toss 3 fair dices one by one, get 3 numbers x, y, z, what is the probability of x &lt; y &lt; z?](#toss-3-fair-dices-one-by-one-get-3-numbers-x-y-z-what-is-the-probability-of-x--y--z)
      * [Other](#other)
   * [Credit](#credit)

<!-- Added by: jesse, at:  -->

<!--te-->

----

# Statistics and Inference

## Basic Concepts

### What is the Central Limit Theorem and why is it important?

Given a sufficiently large sample from a population with finite variance, the sum/mean of the samples would tend to a normal distribution, regardless of the distribution of the population.

It is important because it simplifies problems in statistics by allowing us to work on an approximately normal distribution. 

### What is P-value?

P-value is the probablity of observing data at least as favorable to the alternative hypothesis as our current data, when null hypothesis is true.

**In the context of A/B testing:** P-value is the probablity of seeing experiment group being different to control group to the same or more extreme extent as showing in the data, when there is actually no difference between control and experiment group.

**In the context of linear regression:** Each term has a p-value, where the null hypothesis is that the coefficient should be 0.

### What is Statistical Power?


### What is Confidence Interval?

For a 95% confidence interval, if we repeat the same sampling and estimation process 100 times, the confidence interval would include the true parameter 95 of the times.

**General notes:** a lot of concepts in frequentist inference involves "take infinite samples" or "read the same measurement in the long run". This is directly related to the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem). You should always call out the fact that the confidence of our estimation is based on a repetitive sampling process.

### What is a statistical interaction?


### What is selection bias?


## Hypothesis Test

### What is hypothesis testing?

We hypothesis there is no difference between two groups, and call it the "null hypothesis", then we look for evidence to reject the null hypythosis.

The evidence is based on how likely we will see the same sample statistics again if the null hypothesis is true.

### What is the difference between t-test, z-test, f-test, and Chi-squared test?

- Z-test is based on standard normal distribution and variance of the population. Z score, a.k.a, standard score, is basically how much different is a sample mean (or an observation) is from the population mean in units of population standard error (![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5Csqrt%7B%5Cmathrm%7Bmean%5C%20variance%7D%20/%20N%7D)):
  
  <p align="center"><img alt="" src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B130%7Dz%20%3D%20%5Cfrac%7B%5Cbar%7BX%7D%20-%20%5Cmu%7D%7BSE%7D%20%3D%20%5Cfrac%7B%5Cbar%7BX%7D%20-%20%5Cmu%7D%7B%5Csigma%20/%20%5Csqrt%7Bn%7D%7D%2C"></p>

  where ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5Csigma) is the population standard deviation.
  
  When sample size is large, we can assume the sample mean and variance are close enough to that of the population, then we can use z-test.
- Student's t-test, on the other hand, is based on Student's t-distribution, which would have different shapes in accordance to degree of freedom (sample size).

  <p align="center"><img alt="" src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B130%7Dt%20%3D%20%5Cfrac%7B%5Cbar%7BX%7D%20-%20%5Cmu%7D%7BSE_%7BX%7D%7D%20%3D%20%5Cfrac%7B%5Cbar%7BX%7D%20-%20%5Cmu%7D%7Bs%20/%20%5Csqrt%7Bn%7D%7D%2C"></p>

  where ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7Ds) is the sample standard deviation.

  When sample size is large, student's t-distribution is very close to standard normal distribution. Therefore, in real world application, t-test is normally preferred over z-test.


### What is paired samples t-test, and how is it different from unpaired t-test?

Paried samples t-test typically consists of a sample of matched pairs of similar units, or one group of units has been tested twice. E.g., before and after a treatment. Such a "repeated measures" test compares measurements within subjects, rather than across subjects, which will generally increase the statistical power, or reduce the effects of confounders (variables related to both the outcome and the treatment assignment).

## Experiment Design

### How do you decide when to stop an experiment?


## Bayesian Statistics

### What is the difference between frequentist statistics and Bayesian statistics?

In frequentist statistics, parameters are fixed constants. We calculate the likelihood ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DP%28X%7C%5Ctheta%29), the probability of seeing the obsered data points given the parameter of interest.

In Bayesian statistics, parameters are a random variable with certain distribution. We capture our priori uncertainty about the parameter in the form of a prior distribution, ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DP%28%5Ctheta%29), then update our belief according to the data we observed, so to get a posterior distribution of the parameter, ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DP%28%5Ctheta%7CX%29). Bayesian' Theorem tells us ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DP%28%5Ctheta%7CX%29%20%5Cpropto%20P%28X%7C%5Ctheta%29%20%5Ccdot%20P%28%5Ctheta%29).

## Linear Regression

### What are the assumptions required for linear regression?

### What is R-squared?

R-sqaured tells how much the variance of the target outcome is explained by the model. The higher R-squared, the better the model explains the data.

<p align="center"><img alt="" src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B130%7DR%5E2%20%3D%201%20-%20SSE/SST%2C"></p>

where SSE is the squared sum of the error terms (i.e., sum of squares of residuals, predict value - true value), SST is the squared sum of the data variance (total sum of squares, data points - sample mean).

SST is total variance, SSE is remaining unexplained variance.

More features fit the data better, therefore will see smaller SSE. It's better to use the adjusted R-squared,

<p align="center"><img alt="" src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B130%7D%5Cbar%7BR%7D%5E2%20%3D%201-%5Cfrac%7BSS_%7Bres%7D/df_e%7D%7BSS_%7Btot%7D/df_t%7D%20%3D%20R%5E2%20-%20%281%20-%20R%5E2%29%5Cfrac%7Bp%7D%7Bn-p-1%7D%2C"></p>

which penalizes more complex models.

### How to deal with multicollinearity in linear regression?

Multicollinearity is when multiple predictors correlate with each other to a substantial degree.

In addition to manually identify which variables could be correlated and leave only the most important ones in regression, a couple of methods can be used to automatiacally reduce collinearity:

1. Ridge or Lasso Regression: add L2/L1 regularization term to penalize complex models.
2. Principal Component Regression: predict from factors underlying the predictors (extracted with X'X matrix).
3. Canonical Correlation: predict factors underlying responses (extracted with Y'Y matrix) from factors underlying the predictors.
4. Partial Least Squares Regression: represent predict functions by factors extracted from the Y'XX'Y matrix.

PLS can be used to:

1. Regress when n < p.
2. Identify outliers.
3. Make EDA plots to select predictors.

### What is the relationship between Multiple Regression, General Linear Model and Generalized Linear Model?

## Advanced Statistics

### What is Propensity Score Matching?

Propensity score is normally used in analyzing the effect of biased treatment assignment, e.g., how much do new clinics in poor villages affect the mortality rate? how much more expensive are waterfront properties than regular homes?

The basic idea is:

1. Create a new control group by finding the most similar control observation for every treatment observation based on some selection variable (background characteristics).
2. Compute the treatment effect by comparing the average outcome in the treatment and the new control group.

The trick is in Step 1, in which we use Logistic Regression and all data available to estimate the probability that an observation receives the treatment. Then we use the probability to match pairs of similar observations.

----

# Machine Learning

## Metrics

### Explain the relationship between type I Error, type II error, precision, recall, sensitivity and specificity.

- **Type I error: False Positive**, incorrectrly reject the null hypothesis when null is true
  - Perfect specificity gives zero Type I error.
  - Type I error rate = ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5Calpha) <br>
    = false positive rate <br>
    = significant level <br>
    = 1 - specificity <br>
- **Type II error: False Negative**, failed to reject the null when null is false (alternative is true）
  - Perfect sensitivity gives zero Type II error.
  - Type II error rate = ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5Cbeta) <br>
    = false negative rate <br>
    = 1 - true positive rate <br>
    = 1 - sensitivity / recall / power <br>
    = 1 - the probaility of detecting the effect when there is indeed an effect.

**True positive** means the prediction of being positive is true. Therefore "true positive rate" means of all observations predicted to be positive, how many of them are actually positive.

- True positive + false negative = all known positive observations. 
- True positive + false positive = all assumed positive observations.
- False positive + true negative = all known negative observations.

### What is ROC curve?

Receiver operating characteristics (ROC) curve is a plot of true positive rate against false positive rate at various threshold settings, normally with the x-axis being the false positive rate (1 - sensitivity), and the y-axis being the true positive rate (sensitivity).

We make a prediction ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7Dp%20%5Cin%20%5B0%2C%201%5D), using different threshold ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7Dq%20%5Cin%20%280%2C%201%29), we say all observations with ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7Dp%20%3C%20q) are negative, and others positive.

### What is AUC?

AUC is the probability that a randomly chosen observation from the positive class is greater than a randomly chosen observation from the negative class.

## Modelling

### Compare the advantages and disadvantages of different ML methods.

#### Linear Regression

- Pros:
- Cons:

#### Decision Tree

- Pros:
  - Easy to explain and interpretable
  - Categorical variable support
  - Fast
- Cons:
  - High variance
  - Poor additive modeling

#### Bagging (Random Forest)

Bagging = Bootstrap Aggregation: bootstrap to pick a sample of training data with replacement, train separate submodels, aggregate submodels' predictions by taking the majority vote.

Random Forest: use a subset of features to grow trees.

- Pros:
  - Decrease in **variance** and better accuracy
  - Resilient to outliers
  - Handles missing values well
  - Free validation set
  - Parallel training possible
- Cons:
  - Increase in bias
  - Harder to interpret
  - More expensive to train

#### Boosting (Gradient Boosting Tree, AdaBoosting)

AdaBoosting: up-weight misclassified data points at each step.

- Pros:
  - Decrease in **bias** and better accuracy
  - Additive modeling
- Cons:
  - Increase in variance
  - Prone to overfitting

#### Neutral Networks

- Pros:
  - Handles unstructured data (CV, NLP) well

**Cons:**

### What is the difference between normalization and standardization?

Normalization "normalize" values to a range of [0, 1], by substract the minimal value then divide by value range.

<p align="center"><img alt="" src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B130%7Dx%27%20%3D%20%5Cfrac%7Bx%20-%20x_%7B%5Cmin%7D%7D%7Bx_%7B%5Cmax%7D%20-%20x_%7B%5Cmin%7D%7D"></p>

Standardization "standardize" values to multiples of standard deviation by removing the mean then divide by standard deviation.

<p align="center"><img alt="" src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B130%7Dx%27%20%3D%20%5Cfrac%7Bx%20-%20%5Cmu%7D%7B%5Csigma%7D"></p>

### What are the methods to reduce model complexity in linear regression?

Reducing model complexity is equivalent to introducing sparsity when there are too many features. We do this mostly for combating overfitting. Possible methods include:

- **Lasso regression**: L1 regularization, large ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5Clambda) would make some coefficients to be 0.
- **Manually select** features using expert knowledge.
- **Univariate selection**: only select features whose correlation coefficient with the target variable exceed certain threshold.
- **Step-wise methods**: forward selection or backward selection.

Lasso is prefered because it can be automated, considers all features simultaneously, and can be controlled via lambda.

### What's the difference between forward selection and backward selection?

### What are the methods for dimension reduction?

### What is PCA?

Principal Component Analysis: the process of projecting high dimensional data to lower dimensional vectors in a way that minimizes the projection error. Typically done by matrix decomposition.

Steps to compute PCA:

1. Run mean normalization on the data.
2. Computate covariance matrix ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5CSigma).
3. Run Singular Value Decomposition: ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5CSigma%20%3D%20USV%27), in which diagonal entries of ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DS) are known as the singular value of ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%5CSigma).
4. The first ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7Dk) columns of matrix ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7DU) is the principal components.
5. ![-](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7Dz%20%3D%20U_%7Breduced%7D%27%20x) is the reduced dimensions.

## Neutral Networks / Deep Learning

### What are the techniques to reduce overfitting in Deep Learning?

- L2 and L2 Regularization
    - Cost = Loss + Regularization term
- Dropouts
    - Randomly drop some nodes during each training iteration
    - Similar to ensemble methods
- Data Augmentation
    - Flip, shift, scale, rotate, shear images...
- Early Stopping

## What activation functions do you know?


# Product/Business

For business case and product analysis type of questions, the most important thing is to demonstrate a clear structure in your answers. You'd need: 1. a clear framework or issue tree to tackle the problem; 2. a clear hypothesis whenever applicable.

In data science interviews, there are mainly three types of business case/product sense problems:

1. Make a binary choice---whether a feature is good, whether to enter a market...
2. Predict a numeric value---sales, demand, CTR...
3. Find out why---why certain metrics dropped, why some users perform worse than others...

It's rare for a Data Science interviewer to ask you how to design a product feature.

If it is a binary choice question, you should always state your hypothesis first, then look for evidence to support or reject your hypothesis.

# SQL

## Concepts

### What is the difference between ON clause and WHERE clause in joins?

## Coding Excercise

# Brain Teaser

## Probability 

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

## Other


---

# Credit

Resources consulted while creating this document:

- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
- Wikipedia and Stackoverflow.
- [GitHub Markdown TOC Generation](https://github.com/ekalinin/github-markdown-toc)

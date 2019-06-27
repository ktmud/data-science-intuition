# DS Intuition - Data Science Interview Questions

Build your intuition on important Data Science/Statistics/Machine Learning concepts. Use this material as a study guide for interviews or exams.

All answers here are written to be a quick response to interview questions so that you can impress the interviewer with your intuitive understanding of the concepts.

<!--ts-->
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

- Z-test is based on standard normal distribution and variance of the population. Z score, a.k.a, standard score, is basically how much different is a sample mean (or an observation) is from the population mean in units of population standard error ($\sqrt{\mathrm{mean\ variance} / N}$):
  
  $$z = \frac{\bar{X} - \mu}{SE} = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}},$$

  where $\sigma$ is the population standard deviation.
  
  When sample size is large, we can assume the sample mean and variance are close enough to that of the population, then we can use z-test.
  
- Student's t-test, on the other hand, is based on Student's t-distribution, which would have different shapes in accordance to degree of freedom (sample size).

  $$t = \frac{\bar{X} - \mu}{SE_{X}} = \frac{\bar{X} - \mu}{s / \sqrt{n}},$$

  where $s$ is the sample standard deviation.

  When sample size is large, student's t-distribution is very close to standard normal distribution. Therefore, in real world application, t-test is normally preferred over z-test.


### What is paired samples t-test, and how is it different from unpaired t-test?

Paried samples t-test typically consists of a sample of matched pairs of similar units, or one group of units has been tested twice. E.g., before and after a treatment. Such a "repeated measures" test compares measurements within subjects, rather than across subjects, which will generally increase the statistical power, or reduce the effects of confounders (variables related to both the outcome and the treatment assignment).

## Experiment Design

### How do you decide when to stop an experiment?


## Bayesian Statistics

### What is the difference between frequentist statistics and Bayesian statistics?

In frequentist statistics, parameters are fixed constants. We calculate the likelihood $P(X|\theta)$, the probability of seeing the obsered data points given the parameter of interest.

In Bayesian statistics, parameters are a random variable with certain distribution. We capture our priori uncertainty about the parameter in the form of a prior distribution, $P(\theta)$, then update our belief according to the data we observed, so to get a posterior distribution of the parameter, $P(\theta|X)$. Bayesian' Theorem tells us $P(\theta|X) \propto P(X|\theta) \cdot P(\theta)$.

## Linear Regression

### What are the assumptions required for linear regression?

### What is R-squared?

R-sqaured tells how much the variance of the target outcome is explained by the model. The higher R-squared, the better the model explains the data.

$$R^2 = 1 - SSE/SST,$$

where SSE is the squared sum of the error terms (i.e., sum of squares of residuals, predict value - true value), SST is the squared sum of the data variance (total sum of squares, data points - sample mean).

SST is total variance, SSE is remaining unexplained variance.

More features fit the data better, therefore will see smaller SSE. It's better to use the adjusted R-squared,

$$\bar{R}^2 = 1-\frac{SS_{res}/df_e}{SS_{tot}/df_t} = R^2 - (1 - R^2)\frac{p}{n-p-1},$$

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

### What are the methods to deal with nonlinearities in regression?

### What to do when ...

1. data is not i.i.d. (independent and identically distributed)?
   - mixed models, generalized estimating equations
2. model has heteroscedastic errors (different variance for different values, e.g. prediction errors are larger for more expensive houses)?
   - robust regression
3. outliers strongly influence my model?
   - robust regression
4. I want to predict time-to-event but event is rare?
   - parametric survival models, cox regression, survival analysis.
5. I need to predict ordered categories (e.g., school grades)?
   - proportional odds model.
6. my outcome is count?
   - Use [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression) or negative binomial regression.
   - When count zero is very frequent, use zero-inflated Poisson regression or hurdle model.
7. I have missing data?
   - Multiple imputation

Above list is copied from the [Interpretable ML](https://christophm.github.io/interpretable-ml-book/extend-lm.html) book.

----

# Machine Learning

## Metrics

### Explain the relationship between type I Error, type II error, precision, recall, sensitivity and specificity.

- **Type I error: False Positive**, incorrectrly reject the null hypothesis when null is true
  - Perfect specificity gives zero Type I error.
  - Type I error rate = $\alpha$ <br>
    = false positive rate <br>
    = significant level <br>
    = 1 - specificity <br>
- **Type II error: False Negative**, failed to reject the null when null is false (alternative is true）
  - Perfect sensitivity gives zero Type II error.
  - Type II error rate = $\beta$ <br>
    = false negative rate <br>
    = 1 - true positive rate <br>
    = 1 - sensitivity / recall / power <br>
    = 1 - the probaility of detecting the effect when there is indeed an effect.

**True positive** means the prediction of being positive is true. Therefore "true positive rate" means of all observations that are actually positive, how many of them are predicted positive.

- True positive + false negative = all known positive observations. 
- True positive + false positive = all assumed positive observations.
- False positive + true negative = all known negative observations.

### What is ROC curve?

Receiver operating characteristics (ROC) curve is a plot of true positive rate against false positive rate at various threshold settings, normally with the x-axis being the false positive rate (1 - sensitivity), and the y-axis being the true positive rate (sensitivity).

We make a prediction $p \in [0, 1]$, using different threshold $q \in (0, 1)$, we say all observations with $p < q$ are negative, and others positive.

### What is AUC?

AUC is the probability that a randomly chosen observation from the positive class is greater than a randomly chosen observation from the negative class.

## Modelling

### Compare the advantages and disadvantages of different ML methods.

#### Linear Regression

- Pros:
    - Very easy to interpret
    - Mathematically straightforward and optimal weights are guaranteed
    - With weights you get confidence intervals, tests, and solid statistical theory
- Cons:
    - Nonlinearities and interactions, if there's any, have to be manually added
    - Linearity assumption can be to restrictive, resulting in subpar predictive performance
    - Inpterpretation of a weight can be unintuitive when it depends on other features

#### Decision Tree

- Pros:
  - Easy to explain and interpretable
  - Support categorical variable out-of-the-box
  - No need to transform features
  - Capture interactions between features well
- Cons:
  - Unstable, i.e., high variance
  - Poor additive modeling
  - Fail to deal with linear relationship, resulting in lack of smoothness

A caveat is that number of terminal nodes increase quickly with depth and deep trees are difficult to interpret (because of too many decision rules).

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
- Cons:


### What is the difference between normalization and standardization?

Normalization "normalize" values to a range of [0, 1], by substract the minimal value then divide by value range.

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Standardization "standardize" values to multiples of standard deviation by removing the mean then divide by standard deviation.

$$x' = \frac{x - \mu}{\sigma}$$

### What are the methods to reduce model complexity in linear regression?

Reducing model complexity is equivalent to introducing sparsity when there are too many features. We do this mostly for combating overfitting. Possible methods include:

- **Lasso regression**: L1 regularization, large $\lambda$ would make some coefficients to be 0.
- **Manually select** features using expert knowledge.
- **Univariate selection**: only select features whose correlation coefficient with the target variable exceed certain threshold.
- **Step-wise methods**: forward selection or backward selection.

Lasso is prefered because it can be automated, considers all features simultaneously, and can be controlled via lambda.

### What's the difference between forward selection and backward selection?

### Why do we not use linear regression for classification?

1. A linear model does not output probabilities.
2. The prediction can be below zero or above one, there is no meaningful threshold to distinguish one class from the other.
3. It cannot extend to multi-class classification problems.

### How to interpret logistic regression?

$$
P(y_i=1) = \frac{1}{1 + e^{-\hat{y_i}}}, y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + ...
$$

We wrap a linear model with the logistic function, so to squeeze the output between 0 and 1. Inverse the logistic function, we get the logit function:

$$
\hat{y}_i = \log \left( \frac{P(y_i=1)}{1 - P(y_i=1)} \right)
$$

Which is equivalent to:

$$
\log \left( \frac{P(y_i=1)}{P(y_i=0)} \right) = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + ...
$$

The term inside the log function is called "odds" (probability of event divided by probability of no event), thus we can say **logistic regression is a linear model for log odds**.

> A change in a feature by one unit changes the odds ratio (multiplicative) by a factor of $\exp(\beta_i)$.

We can also say

> A change in $x_i$ by one unit increases the log odds ratio by the value of the corresponding weight. 

[Details here](https://christophm.github.io/interpretable-ml-book/logistic.html).


### What are the methods for dimension reduction?

1. Remove columns with too many missing values
2. Remove columns with constant value or low variance
3. Reduce highly correlated columns, pick the one with the largest variance
4. Use domain knowledge to select an appropriate set of features
5. Rank features based on their importance measured from information gain or linear regression
6. Use PCA
7. Use Lasso/Ridge regression 


### What is PCA?

Principal Component Analysis: the process of projecting high dimensional data to lower dimensional vectors in a way that minimizes the projection error. Typically done by matrix decomposition.

Steps to compute PCA:

1. Run mean normalization on the data.
2. Computate covariance matrix $\Sigma$.
3. Run Singular Value Decomposition: $\Sigma = USV'$, in which diagonal entries of $S$ are known as the singular value of $\Sigma$.
4. The first $k$ columns of matrix $U$ is the principal components.
5. $z = U_{reduced}' x$ is the reduced dimensions.

### What is an identify function? A hinge function?

### How to handle umbalanced data?

1. Undersample the larger classes or oversample the smaller classes
2. For oversampling, we could do bootstraping (sample with replacement) or SMOTE (Synthetic Minority Over-sampling Technique):
    take a random sample and its random closest neighbor within k ranks, nudge each feature towards the neighbor a little bit.
3. Penalize classification by increasing weights to the minority class


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

## What is hidden in Hidden Markov Model?

Markov models are Bayesian networks. In 


# Product/Business

For business case and product analysis type of questions, the most important thing is to demonstrate a clear structure in your answers. You'd need: 1. a clear framework or issue tree to tackle the problem; 2. a clear hypothesis whenever applicable.

In data science interviews, there are mainly three types of business/product sense problems:

1. Decision making: make a binary choice, e.g., whether a feature is good, whether to enter a market...
2. Forecasting: predict a numeric value, sales, demand, CTR...
3. Reasoning: find out whycertain metrics dropped, why some users perform worse than others...

It's rare for a Data Science interviewer to ask you how to design a product feature.

## Decision Making

For decision making questions, especially binary choices, you should always state your hypothesis first, then look for evidence to support or reject your hypothesis. Use the following template:

> My hypothesis is that we should [enter this market]. To test my hypothesis, I 'd like to [look at a couple of factors/track the following metrics]: ...

## Forecasting


# SQL

## Concepts

### What is the difference between ON clause and WHERE clause in joins?

## Coding Excercise

# Brain Teaser

## Probability 

### How to get fair results from a biased coin?

John von Neumann gave the following procedure:

1. Toss the coin twice.
2. If the results match, start over, forget both results.
3. If the results differ, use the first result, forget the second.

This is because the probability of getting head then tail is the same as getting tail then head. By excluding the other two outcomes of two indepedent tosses---getting both heads and getting both tails, we are left with only two outcomes with equal probability.

### Pick 3 cards from a deck of cards labeled from 1 to 100, what’s the probability of getting Card 1 < Card 2 < Card 3?

The three cards are equivalent to 3 random numbers from [1, 100] without replacement. Getting Card 1 < Card 2 < Card 3 is equivalent to getting the numbers in a specific order. There are in total $P(3, 3) = 6$ ways of arranging 3 numbers. Therefore the probability of Card 1 < Card 2 < Card 3 is $1/6$.

### Toss 3 fair dices one by one, get 3 numbers x, y, z, what is the probability of x < y < z?

Event space: $6 \times 6 \times 6 = 216$. Two of x, y, z are equal: $3 \times 6 \times 5$; all three are equal: $6$.

The remaning $216 - 3 \times 6 \times 5 - 6 = 120$ outcomes  are basically arranging 3 different numbers, all from the same discrete uniform distribution $[1, 6]$, in different orders. Similar to previous question, there are $120/6 = 20$ ways of getting $x < y < z$. The probability is $20 / 216 = 9.26\%$.

## Other


---

# Credit

Resources consulted while creating this document:

- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
- Wikipedia and Stackoverflow.
- [GitHub Markdown TOC Generation](https://github.com/ekalinin/github-markdown-toc)

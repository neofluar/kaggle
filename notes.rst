===================
1. The Fundamentals
===================

1.1. What
=========

ML is the science of programming computers so they can learn from data.

A computer program is said to learn from experience E with respect to some task T and some perfomance measure P, if its perfomans on T, as measured by P, improves with E.

1.2. Why
========

ML is great for:

- Problems for which existing solutions require a lot of fine-tuning or long list of rules;
- Complex problems for which using traditional approach yields no good solution;
- Fluctuating environment: a ML system can adapt to new data;
- Getting insights about complex problems and large amount of data.

Examples of applications:

- Analyzing images of products
- Detecting tumors in brain scans
- Automatically classifying new articles
- Flagging offensive comments
- Summarizing documents
- Creating chatbots
- Forecasting company's revenue
- Voice recognition
- Detecting credit card fraud
- Segmenting clients
- Recommending products

1.3. Types
==========

- By amount and type of supervision they get during training:
  Supervised/Unsupervised/Semisupervised/Reinforcment
- By whether or not they can learn incrementally on the fly:
  Online (Incremental) / Batch learning
- By comparing new data to known data, or by detecting patterns and building a predictive model
  Instance / Model based learning

Supervised
----------

The training set includes the desired solutions, called labels (classification and regression).

- k-Nearest Neighbors 
- Linear Regression
- Logistic Regression
- Support Vector Machines
- Decision Trees
- Random Forest
- Neural Networks (except autoencoders and Boltzman machine)

Unsupervised
------------

The training set is unlabeled.

- Clustering (detecting groups of similar datapoints):

  - K-Means
  - DBSCAN
  - Hierarchical Cluster Analysis (HCA)

- Anomaly/Novelty detection:

  - One-class SVM
  - Isolation Forest

- Visualization and Dimensionality reduction (simplify data representation):

  - Principal Component Analisis (PCA)
  - Kernel PCA
  - Locally Linear Embedding (LLE)
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)

Assosiation Rule Learning
-------------------------

Discovering hidden relations between data attributes:

- Apriori
- Eclat

Semisupervised
--------------

The training set is partially labeled.

- Deep Belief Networks (DBNs) are based on unsupervised component Restricted Boltzman Machines (RBMs),
  stacked on top of one another. The whole system is fine-tuned using supervised learning.

Reinforcment Learning
---------------------

An agent can observe the Environment, select and perform actions, and get rewards in return. It must learn the best strategy, called Policy, to get the most reward over time.

Batch Learning
--------------

A system must be trained using all available data, offline learning. To incorporate new data we need to learn the system from scratch on the full dataset. Thus it cannot adapt to rapidly changing data. Usually it takes a lot of time, power, memory and computing resources. 

Incremental Learning
--------------------

Train the system incrementally by feeding it data instances sequentially, either individually or in small groups. This is a more reactive solution. Each learning step is fast and cheap, so the system can learn about new data on the fly. To control how fast the system should adapt to changing data the learning rate is used.

Instance-Based Learning
-----------------------

The system learns the examples by heart, then generalizes to new cases by using a similarity measure to compare them to the learned examples.

Model-Based Learning
--------------------

The system builds a model of these examples and then uses that model to make predictions.

1.4. Summary So Far
===================

1. You studied the data.
2. You selected a model.
3. You trained it on the training data (i.e. the learning algorithm searched for the model 
   parameter values that minimize a cost function).
4. You applied the model to make predictions on new cases (inference).

1.5. Main Challenges
====================

- Data

  - insufficiant quantity of training data
  - nonrepresentative training data
  - poor-quality data
  - irrelevant features

- Model

  - overfitting (more training data, less features, simplify the model: regularization)
  - underfitting (a more powerful model, reduce regularization, better features)

No Free Lunch Theorem
---------------------

If you make absolutely no assumption about the data, then there is no reason to prefer one model over any other.

================================
2. End-to-End Regression Example
================================

2.1. Main steps
===============

1. Frame the problem and look at the big picture
2. Get the data
3. Explore the data to get insights
4. Prepare the data for ML algorithms
5. Explore many different models and shortlist the best ones
6. Fine-tune your models
7. Present your solution
8. Launch, monitor, and maintain your system

2.2. Frame the Problem
======================

1. Define the objective in business terms
2. How will your solution be used?
3. What are the current solutions/workarounds (if any)?
4. How should you frame this problem (supervised/unsupervised, online/offline, etc.)?
5. How should performance be measured?
6. Is the performance measure aligned with the business objective?
7. What would be the minimum performance needed to reach the business objective?
8. What are comparable problems? Can you reuse experience or tools?
9. Is human expertise available?
10. How would you solve the problem manually?
11. List the assumptions you (and others) have made so far
12. Verify assumptions if possible
 
2.3. Get the Data
=================
 
1. Take care of your credentials and access authorizations to get the data
2. Familiarize yourself with the data scheme
3. Load the data and take a quick look at the data structure (number and types of attributes, missing values, etc.)
4. Visualize attribute distributions if possible
5. Create a test set as early as possible: avoid data snooping bias.

2.4. Create a Test Set
======================

Random methods with fixed seed based on indicies or unique, immutable ids make updating your dataset not trivial.

Stratified sampling by the most valuable feature in the dataset. The feature should not have too many strata, and each stratum should be large enough.

2.5. Discover and Visualize the Data
====================================

1. Use different scatter plots
2. Look for linear correlations between attributes using
4. Zoom in on the distinct correlation plots to see data quirks and anomalies if any
5. Experiment with feature engineering (combine some attributes) using common sence, then check the correlation agains the new    attributes

2.6. Prepare the Data for ML Algorithms
=======================================

1. Separate the predictors and the labels.
2. Deal with missing values:
  
  2.1. Get rid of the corresponding samples
  
  2.2. Get rid of the whole attribute
  
  2.3. Set the values to some value (zero, mean, median, etc.)
  
3. Encode categorical attributes using ordinal numbers. Bear in mind that ML algorithms will assume that 2 nearby values are      more similar than 2 distant values. That is good for ordered categories, but it can be not your case, then use *one-hot        encoding*. If a categorical attribute has a large number of possible categories, then one-hot encoding will result in a        large number of input features. This may slow down training and degrade performance. If this happens, you may want to          replace the categorical input with useful numeric features related to the categories. Alternatively, you could replace each    category with a learnable, low-dimensional vector called an *embedding*.
4. Scale input features:

  4.1. *Min-max scaling* (aka *normalization*) is the simplest: values are shifted and rescaled so they end up ranging from 0          to 1. We do this by subtracting the min value and dividing by the max minus the min.
  
  4.2. *Standardization*: subtract the mean and divide by standard deviation. Unlike min-max scaling, standardization does not        bound values to a specific range. However, standardization is much less affected by outliers.

5. Create a custom transformer to automate both transformation of numerical and categorical attributes.

2.7. Select and Train a Model
=============================

Train a simple model, then evaluate it on the training set. If a typical prediction error on the training set is large, the model underfits the training data. It means that the features do not provide enough information to make good predictions, or that the model is not powerful enough.

To fix underfitting:

- Select a more powerfull model
- Feed the model with better features
- Reduce the constraints on the model (regularization)
  
If a typical prediction error on the training set is small (or zero), that may be a sign of the model overfits the training set. You need another way to evaluate such a model using the training set. Do not touch the test set yet! Make use of *K-fold cross-validation* on the training set only. Notice that cross-validation allows you to get not only an estimate of the performance of your model, but also a measure of how good this estimate is (i.e. its standard deviation). If the score on the training set is much lower than on the validation sets, that still means the model overfits the training set.

To fix overfitting:

- Simplify the model
- Constrain the model (regularization)
- Get a lot more training data

Try out many other models from various categories of ML algorithms, without spending too much time tweaking the hyperparameters. The goal is to shortlist 2-5 promissing models.

2.8. Fine-Tune Your Model
=========================

A few things to do

1. Grid Search is fine when you explore relatively few combinations
2. Random Search is fine when the hyperparameters search space is large
3. Combine the models that perform better *ensemble methods*

You will often get good insights on the problem by inspecting the best model. You may want to try dropping some of the less important features. After tweaking your model for a while, you eventually have a system that performs sufficiently well. Now it is time to evalute it on the test set. If you did a lot of hyperparameters tuning, the performance will usually be slightly worse than what you measured using cross-validation. *Resist the temptation to tweak hyperparameters to make the numbers look good on the test set; the improvements would be unlikely to generalize on the new data!*

Present your solution to the stake holders. Highlight what you have learned, what worked and what did not, what assumptions were made, and what your system's limitations are.

2.9. Launch, Monitor, and Maintain Your System
==============================================

1. Deploy the model to your production environment (website, web service, cloud)
2. Write monitoring code to check your system's live performance at regular intervals and trigger alerts when it drops
3. If the data keeps evolving, you will need to update your dataset and retrain the model regulary
4. Evaluate input data quality constantly
5. Keep backups of every model you create and every version of the dataset

=================
3. Classification
=================

3.1. Binary Classification
==========================

A good way to evaluate a model's generalization performance is to use *cross-validation* that can be performed with different scoring strategies. If a model performs well on the training data but generalizes poorly accordingly to the cross-validation metrics, then your model is overfitting. If it performs poorly on both, then it is underfitting.

Accuracy
--------

Accuracy is generally not the preffered performance measure for classifiers, especially when you are dealing with a skewed dataset. A much better way to evaluate the performance of a classifier is to look at the *confusion matrix*. 

Confusion Matrix
----------------

The general idea is to count the number of times instances of class A are classified as class B. Each row in the matrix represents an *actual* class, while each column represents a *predicted* class.

+---------------------+---------------------+--------------------+
|                     |  Predicted Negative | Predicted Positive |
+=====================+=====================+====================+
| **Actual Negative** | TN                  | FP                 |
+---------------------+---------------------+--------------------+
| **Actual Positive** | FN                  | TP                 |
+---------------------+---------------------+--------------------+

A perfect classifier would have onlu true positives and true negatives values.

Precision and Recall
--------------------

*Precision = TP / (TP + FP)* is the accuracy of the positive predictions (*specificity*).  

*Recall = TP / (TP + FN)* is the ratio of actual positive instances that are correctly detected (*sensitivity*).  

When the model claims an image represents positive class, it is correct only 100xP% of the time. Moreover, it only detects 100xR% of the actual positives.  

It is often convenient to combine precision and recall into a single metric called *F1 score*. 

F1 Score
--------

It is the harmonic mean of precision and recall that gives much more weight to low values. As a result, the classifier will only get a high F1 score if both recall and precision are high.  

F1 = 2 / (1/P + 1/R)  

Precision/Recall Trade-off
--------------------------

Increasing precision reduces recall, and vice versa. The key concept is a moving decision threshold. Increasing the threshold increases precision and reduces recall. Conversely, lowering the threshold increases recall and reduses precision. We can observe it by controling the threshold manualy. We can plot precision and recall against all possible threshold values to select a good trade-off. 

The PR Curve
------------

Another way to do that is to plot precision directly against recall and choose an arbitrary balance point according to our task in hands. But remember that increasing the threshold allows you to get any precision value you want. But a high-precision classifier is not very useful if its recall is too low.

The ROC Curve
-------------

The *reciver operating characteristic* (ROC) curve is another common tool used with binary classifiers. It is very similar to the PR curve, but instead of plotting precision versus recall, the ROC curve plots of the *true positive rate* (aka recall) against the *false positive rate* for all possible thresholds. The FPR is the ratio of negative instances that are incorrectly classified as positive. It is equal to 1 minus the *true negative rate*, which is the ratio of negative instances that are correctly classified as negative. One way to compare classifiers is to measure the *area under the curve* (AUC).

PRC or ROC?
-----------

Prefer the PR curve whenever the positive class is rare or whenever you care more about the false positives than the false negatives. Overwise, use the ROC curve.


3.2. Multiclass Classification
==============================

Whereas binary classifiers distinguish between 2 classes, *multyclass classifiers* can distinguish between more than 2 classes. Some algorithms (such as SGDClassifier, Random Forest, and naive Bayes classifiers) are capable of handling multiple classes natively. Others (such as Logistic Regression or Support Vector Machine classifiers) are strictly binary. However, there are various strategies that you can use to perform multiclass classification with multiple binary classifiers.

- *One-vs-Rest* (or *One-vs-All*): each class gets its own binary classifier. Select the class whose classifier outputs the       highest score

- *One-vs-One*: each class pair gets its own classifier (if there is N classes, then you train Nx(N-1)/2 binary classifiers.      Select a class that won the most duels

OvO has much more classifiers to train. The main advantage of OvO is that each classifier needs to be trained on the part of the training set for the 2 classes that it must distinguish. Some algorithms (such as Support Vector Machine classifier) scale poorly with the size of the training set. For this algorithms OvO is preffered because it is faster to train many classifiers on small training sets than to train few classifiers on large training sets. For most binary classification algorithms, however, OvR is preffered.

3.3. Error Analysis
===================

We will assume that you have found a promising model and you want to find ways to improve it. One way to do this is to analyze the types of errors it makes.

Look at the confusion matrix. It is often more convinient to look at an image representation of the confusion matrix. But      first, divide its values by the number of images in the corresponding class so you can compare error rates instead of          absolute numbers of errors. Fill the diagonal with 0s to keep errors only, and plot the result. Analyzing the confusion matrix often gives you insights into ways to improve your classifier. Try to gather more images of the most misclassified classes. Or engineer new features that would help the classifier. Or preprocess images to make some patterns (such as closed loops) stand out more.

3.4. Multilabel Classification
==============================

In some cases you may want your classifier to output multiple classes for each instance. Such a a classification system that outputs multiple binary tags is called a *multilabel classification* system. In general you need only create 2 or more label sets and pass them to an algorithm which supports multilabel classification such as `KNeighborClassifier`.  

There are several ways to evaluate a multilabel classifier, and selecting the right metric really depends on your task. One approach is to measure F1 score for each individual label (or any other classifier metric), then simply average them. This assumes that all labels are equally important, which may not be the case. You can assign a weight to each label.

3.5. Multioutput Classification
===============================

*Multioutput-multiclass* classification is simply a generaluzation of multilabel classification where each label can be multiclass (i.e., it can have more than 2 possible values). To illustrate this, we can build a system that removes noise from images. Notice that the classifier's output is multilabel (one label per pixel) and each label can have multiple values (pixel intensity ranges from 0 to 255).

==================
4. Training Models
==================

4.1. Linear Regression
======================

There are 2 very different ways to train it:

- Using a direct "closed-form" equation that directly computes the model parameters that best fit the model to the training set   (i.e., the model parameters that minimize the cost function over the training set
- Using an iterative optimization approach called *Gradien Descent* that gradually tweaks the model parameters to minimize the   cost function over the training set

More generally, a linear model makes predictions by simply computing a weighted sum of the input features, plus a constant called the *bias term* (*intersept term*)

y_pred = THETA_0 + THETA_1*x_1 + THETA_2*x_2 + ... + THETA_n*x_n

This can be written using a vectorized form  

y_pred = **THETA** **x**

- **THETA** is the model's *parameter vector*, containing the bias term THETA_0 and the feature weights THETA_1 to THETA_n
- **x** is the instance's *feature vector* from x_1 to x_n, containong x_0 = 1

The Normal Equation
-------------------

**THETA_BEST** = (**X**.T **X**) ^ -1 **y**

- **THETA_BEST** is the value of **THETA** that minimize the cost function
- **y** is the vector of target values

`LinearRegression` from `Scikit-learn` is based on the *pseudoinverse* of **X**. The pseudoinverse is computed using a standard matrix factorization technique called *Singular Value Decomposition* (*SVD*) that can decompose the training set matrix **X** into the matrix multiplication of 3 matrices. This approach is more efficient than computing the Normal Equation, plus it handles edge cases nicely.

Computational Complexity
------------------------

The Normal Equation computes the inverse of **X**.T **X**, which is (n + 1)x(n + 1) matrix (where n is the number of features). The *computational complexity* of inverting such a matrix is typically about O(n^2.4) to O(n^3). The SVD approach is about O(n^2). But both the Normal Equation and the SVD approach get very slow when the number of features grows large (e.g., 100.000). On the positive side, both are linear with regard to the number of instances in the training set O(m). In both cases, predictions are very fast: the computational complexity is linear with regard both the number of instances in the test set and n.  

Now we will look at a very different way to train a LR model, which is better suited for cases where there are a large number of features or too mane training instances to fit in memory.

4.2. Gradient Descent
=====================

*Gradient Descent* is a generic optimization alogorithm capable of finding optimal solution to a wide range of problems. The general idea of GD is to tweak parameters iteratively in order to minimize a cost function. It measures the local gradient of the error function with regard to the parameter vector **THETA**, and it goes in the direction of descending gradient. The size of a step is determined by a *learning rate* (*eta*) hyperparameter. Once the gradienrt is zero, you have reached a minimum (local or global).  

The MSE cost function for a LR model happens to be a *convex* function. This implies that there are no local minima, just one global minimum. When using GD, you should ensure that all features have a similar scale. Or else it will take much longer to converge.

Batch Gradient Descent
----------------------

To implement GD, you need to compute the gradient of the cost function with regard to each model parameter *THETA_j*. in other words, you need to calculate how much the cost function will change if you change *THETA_j* just a little bit. This is called a *partial derivative*. The formula involves calculations of the full training srt **X**, at each gradient step. This is why the algorithm is called *Batch GD*: it uses the whole batch of training data at every step.  

As a result it is terribly slow on very large training sets. However, GD scales well with number of features; training a LR model when there are hundreds of thousands of features is much faster using GD than using the Normal Equation or SVD decomposition.

Stochastic Gradient Descent
---------------------------

Opposite to BGD that uses the whole training set on each step, *Stochastic Gradient Descent* picks a random instance in the training set and computes forward pass and the gradients based only on that single instance. It makes the algorithm mush faster. It also makes possible to train on huge training sets.  

On the other hand, due to its stochastic nature, this algorithm is much less regular than BGD: instead of gentlly decreasing until it reaches the minimum, the cost function will bounce up and down, decreasing only on average. Over time it will end up very close to the minimum, but once it gets there it will continue to bounce around, never settling down. When the cost function is very irregular, this can actually help the algorithm jump out of local minima, so SGD has a better chances of finding the global minimum than BGD does. Randomness is good to escape from local minima, but bad because it means that the algorithm can never settle at the minimum. One solution is to gradually reduce the learning rate according to a *learning schedule*.

Mini-batch Gradient Descent
---------------------------

*Mini-batch Gradient Descent* computes the gradients on small random sets of instances called *mini-batches*. The performance boost comes from hardware optimization of matrix operations, especially using GPU. The algorithm's progress in parametre space is less erratic than with SGD, especially with fairly large mini-batches.  

*Comparision of algorithms for Linear Regression*

+---------------------+---------+---------+---------+---------+---------+
| Algorithm           | Large m | Large n | Scaling | Hparams | Scikit  |
+=====================+=========+=========+=========+=========+=========+
| Normal Equation     | Fast    | Slow    | No      | 0       | N/A     |
+---------------------+---------+---------+---------+---------+---------+
| SVD                 | Fast    | Slow    | No      | 0       | LR      |
+---------------------+---------+---------+---------+---------+---------+
| Batch GD            | Slow    | Fast    | Yes     | 2       | SGDR    |
+---------------------+---------+---------+---------+---------+---------+
| Stochastic GD       | Fast    | Fast    | Yes     | >= 2    | SGDR    |
+---------------------+---------+---------+---------+---------+---------+
| Mini-batch GD       | Fast    | Fast    | Yes     | >= 2    | SGDR    |
+---------------------+---------+---------+---------+---------+---------+

4.3. Polinomial Regression
==========================

You can use a linear model to fit nonlinear data. A simple way to do it is to add powers of each feature and their combinations up to ceratain degree as new features, then train a linear model on this extended set of features. This technique is called *Polinomial Regression*. So features *a* and *b* with degree 2 become *a*, *ab*, *b*, *a^2*, *b^2*.

4.4. Learning Curves
====================

Early we used cross-validation to get an estimate of a model's generalization performance (under/over-fitting). Another way to tell is to look at the *learning curves*: these are plots of the model's performance on the training set and the validation set as a function of the training set size (or the training iteration).  

If both curves have reached a plateau and they are close and fairly high, the model is underfitting the data. Adding new instances to the training set can not improve errors. You need to use a more complex model or come up with better features.  

If both curves are much lower and there is a clear gap between them, the model is overfitting the data. Adding new instances to the training set would help the validation error reach the training error.  

The Bias/Variance Trade-off
---------------------------

A model's generalzation errors can be expressed as the sum of 3 errors:

- *Bias* is due to wrong assumptions, such as assuming that the data is linear when it ia actually quadratic. A high-bias model   is most likely to *underfit* the training data
- *Variance* is due to the model's excessive sensitivity to small variations in the training data. A model with many degrees of   freedom is likely to have high variance and thus *overfit* the training data
- *Irreducible error* is due to the noiseness of the data itself.The only way to reduce it is to clean up the data

Increasing a model's complexity will typically increase its variance and reduce its bias. And vise versa.

4.5. Regularized Linear Models
==============================

As we saw earlier, a good way to reduce overfitting is to *regularize* the model (i.e., to constrain it). A simple way to regularize a polynomial model is to reduce the number of polynomial degrees. For a linear model, regularization is typically achieved by constraining the weights of the model. It is important to scale the data befor performing almost all regularized models.

Ridge Regression
----------------

*Ridge Resression* is a regularized version of Linear Regression: a *regularization term* equal to half of the weighted sum of squared weights is added to the cost function: 0.5 * *alpha* * sum(**THETA**)^2, which is 0.5 * *alpha* * L2norm(**THETA**). This forces the algorithm to not only fit the data but also keep the model weights as small as possible. Increasing alpha leads to flatter predictions, thus reducing the model's variance but increasing bias.

Lasso Regression
----------------

*Lasso Regression* is a regularized version of Linear Regression: a *regularization term* equal to weighted sum of eights: *alpha* * sum(**THETA**), which is *alpha* * L1norm(**THETA**). The Lasso tends to eliminate the weights of the least important features (i.e., set them to 0). In other words, Lasso Regression automatically performs feature selection and outputs a *sparce model*.

Elastic Net
-----------

*Elastic Net* is a middle ground between Ridge Regression and Lasso Regression. The regularization term is just a mix of their terms, and you can control the mix ratio: *r* * LassoTerm + (1 - *r*) * RidgeTerm.  

It is always preferable to have a little bit of regularization, so generally you should avoid plain Linear Regression. Ridge is a good default, but if you suspect that only a few features are useful, you should prefer Lasso or ElasticNet. In general, Elastic Net is preffered over Lasso when several features are stronglt correlated.

Early Stopping
--------------

A very different way to regularize iterative learning algorithms such as GD is to stop training as soon as the validation error reaches a mininmum. This is called *early stopping*.

4.6. Logistic Regression
========================

*Logistic Regression* - a binary classifier with a threshold 0.5 that is commonly used to estimate probability that an instance belongs to a particular class. Just like a Linear Regression model, a Logistic Regression model computes a weighted sum of the input features (plus a bias term) and outputs the *logistic* of this result

``p_pred = sigmoid(*x*.T **THETA**)``

Once the Logistic Regression model has estimated the probability, it can make a binary prediction with 0.5 probability threshold. Notice thet sigmoid(*t*) < 0.5 when *t* < 0, and sigmoid(*t*) >= 0.5 when *t* >= 0, where *t* = *x*.T **THETA**. The score *t* is often called the *logit*. The name comes from the fact that the logit function, defined as logit(*p*) = log(*p* / (1 - *p*)), is the inverse of the logistic function. Indeed, if you compute the logit of the estimated probability *p*, you will find that the result is *t*. The logit is also called the *log-odds*.  

The cost function for a single training instance **x**:   

c(**THETA**) = -log(*p_pred*) if *y* = 1, -log(1 - *p_pred*) if *y* = 0  

The Logistic Regression cost function over all training set:  

J(**THETA**) = -sum(*yi**log*(*p_predi*) + (1 - *yi*)*log*(1 - *p_predi*)) / m  

There is no closed-form equation to compute **THETA_BEST**, but this cost function is convex, so GD is guaranteed to find the global minimum.  

A *decision boundary* is a value of a feature where probability is equal to 50%.

4.7. Softmax Regression
=======================

*Softmax Regression*, or *Multinomial Logistic Regression* is a multiclass classifier. When given an instance **x**, the Softmax Regression model first computes a score sk(**x**) = **x**.T **THETA_k** for each class *k*, then estimates the probability of each class by applying the *softmax function* to the scores that computes the exponential of every score, then normalizes them. Each class has its own dedicated parameter vector **THETA_k**. The scores are generally colled *logits* or *log-odds*. The Softmax Classifier predicts the class with the highest estimated probability (which is simply the class with the highest score). The cost function is called *cross-entropy*. You can compute the gradient vector for every class, then use GD to find parameter matrix **THETA** that minimizes the cost function. *Decision boundaries* are linear as it is still a linear model.

=========================
5. Support Vector Machine
=========================

5.1. Linear SVM Classification
==============================

If the 2 classes can be separated easily with a straight line (they are *linearly separable*), the decision boundary of an SVM classifier not only separates the 2 classes but also stays as far away from the closest training instances as possible. These instances are called the *support vectors*. This type of classification is called *large margin classification*.  

SVM are sensitive to the feature scales.

Soft Margine Classification
---------------------------

If we strictly impose that all instances must be of the street and on the right side, this is called *hard margin classification*. It only works if the data is linearly separable, and it is sensitive to outliers. Also it will probably not generalize well. To avoide these issues, use *soft margin classification*: find a good balance between keeping the street as wide as possible and limiting the margin violations. Although margin violations are bad, it's usually better to have a few of them. The model will probably generalize better.

5.2. Nonlinear SVM Classification
=================================

If the dataset is not linearly separable, then one approach is to add more features, such as polinomial features. In some cases this may result in a linearly separable dataset.

Polynomial Kernel
-----------------

The *kernel trick* makes it possible to get the same result as if you had added many polynomial features, even with very high-degree polinomials, without actually having to add them.


Similarity Features
-------------------

Another technique to tackle nonlinear problems is to add features computed using *similarity function*, which measures how much each instance resembles a particular *landmark*. *Gaussian Radial Basis Function* (RBF) is an example of this function. Landmark can be created at the location of every instance in the dataset.

Selecting a Kernel
------------------

Always try the linear kernel first, especially if the training set is large.If it is not, you should also try the Gaussian RBF kernel.

Computational Complexity
------------------------

+---------------------+---------+-------------+---------+--------------+
| Class               | Time    | Out-of-Core | Scaling | Kernel trick |
+=====================+=========+=============+=========+==============+
| Linear SVC          | O(mn)   | No          | Yes     | No           | 
+---------------------+---------+-------------+---------+--------------+
| SGDClassifier       | O(mn)   | Yes         | Yes     | No           | 
+---------------------+---------+-------------+---------+--------------+
| SVC                 | O(m^2n) | No          | Yes     | Yes          | 
+---------------------+---------+-------------+---------+--------------+

5.3. SVM Regression
===================

The SVM algorithm also supports linear and nonlinear regression. It tries to fit as many instances as possible on the street while limiting margin violations.

==================
6. Decision Trees
==================

*Decision Trees* are *non-parametric* algorithm, that are simple to interpret, easy to use, versatile.  

The greedy recursive algorithm works by first splitting the training set into 2 subsets using a single feature *k* and its threshold *tk*. it searches for the pair (*k*, *tk*) that produces the purest subsets weighted by their size. It stops recursing once it reaches the maximum depth, or if it cannot find a split that will reduce impurity.  

The training complexity is O(*n* *m* *log*(*m*)), the testing - O(*log*(*m*)).  

Decision Trees manages classification and regression tasks (and find anomalities) and don't require feature scaling or centering.  

Decision Trees make very few assumptions about the training data. If left unconstrained, the tree will adapt itself to the training data most likely overfitting it. Increasing ``min_`` hyperparameters or reducing ``max_`` will regularize the model.  

Disadvantages:

- Sensitive to training set rotation (use PCA to fix)
- Sensitive to small variations in the training data
- Stochastic nature

=======================================
7. Ensemble Learning and Random Forests
=======================================

If you aggregate the predictions of a group of predictors, you will often get better predictions than with the best individual predictor. A group of predictors is called an *ensemble*. Ensemble methods work best when the predictors are as independent as possible.

7.1. Voting Classifiers
=======================

One approach can be to train several different classifiers and aggregate the predictions of each classifier to predict the class that gets the most votes. A majority-vote classifier is called a *hard voting* classifier. If all classifiers in the ensemble are able to predict class probabilities, then we can predict the class with the highest class probability, averaged over all the individuals classifiers. This is called *soft voting*.

7.2. Bagging and Pasting
========================

Another approach is to use the same training algorithm for every predictor in the ensemble and train them on different random subsets of the training set. The typical size of such a subset is equal to the size of the training set. *Bagging* (*bootstrap aggregating*) and *pasting* are methods to select samples for these subsets. Bagging allows sample repetition for each individual algorithm, and pasting doesn't. Generally, the net result is that the ensemble has a similar bias but a lower variance than a single predictor trained in the original training set.

Out-of-Bag Evaluation
---------------------

With bagging, some instances may be sampled several times for any given predictor, while others may not be sampled at all. This means that a part of the training instances are never used to train a particular predictor. They are called *out-of-bag* instances. We can use them to evaluate the predictor without the need for a validation set. And it is possib;e to evaluate the ensemble itself by averaging out the oob evaluations of each predictor.

7.3. Random Patches and Random Subspaces
========================================

The `BaggingClassifer` class supports sampling the features as well allowing feature sampling instead of instance sampling. Thus, each predictor in an ensemble will be trained on a random subset of the input features. This way we trade a bit more bias for a lower variance.

7.4. Random Forests
===================

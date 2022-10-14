===================
1. The Fundamentals
===================

What
====

ML is the science of programming computers so they can learn from data.

A computer program is said to learn from experience E with respect to some task T and some perfomance measure P, if its perfomans on T, as measured by P, improves with E.

Why
====

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

Types
=====

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

Summary So Far
==============

1. You studied the data.
2. You selected a model.
3. You trained it on the training data (i.e. the learning algorithm searched for the model 
   parameter values that minimize a cost function).
4. You applied the model to make predictions on new cases (inference).

Main Challenges
===============

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

Main steps
==========

1. Frame the problem and look at the big picture
2. Get the data
3. Explore the data to get insights
4. Prepare the data for ML algorithms
5. Explore many different models and shortlist the best ones
6. Fine-tune your models
7. Present your solution
8. Launch, monitor, and maintain your system

Frame the Problem
=================

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
 
Get the Data
============
 
1. Take care of your credentials and access authorizations to get the data
2. Familiarize yourself with the data scheme
3. Load the data and take a quick look at the data structure (number and types of attributes, missing values, etc.)
4. Visualize attribute distributions if possible
5. Create a test set as early as possible: avoid data snooping bias.

Create a Test Set
=================

Random methods with fixed seed based on indicies or unique, immutable ids make updating your dataset not trivial.

Stratified sampling by the most valuable feature in the dataset. The feature should not have too many strata, and each stratum should be large enough.

Discover and Visualize the Data
===============================

1. Use different scatter plots
2. Look for linear correlations between attributes using
4. Zoom in on the distinct correlation plots to see data quirks and anomalies if any
5. Experiment with feature engineering (combine some attributes) using common sence, then check the correlation agains the new    attributes

Prepare the Data for ML Algorithms
==================================

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

Select and Train a Model
========================

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

Fine-Tune Your Model
====================

A few things to do

1. Grid Search is fine when you explore relatively few combinations
2. Random Search is fine when the hyperparameters search space is large
3. Combine the models that perform better *ensemble methods*

You will often get good insights on the problem by inspecting the best model. You may want to try dropping some of the less important features. After tweaking your model for a while, you eventually have a system that performs sufficiently well/ Now it is time to evalute it on the test set. if you did a lot of hyperparameters tuning, the performance will usually be slightly worse than what you measured using cross-validation. *Resist the temptation to tweak hyperparameters to make the numbers look good on the test set; the improvements would be unlikely to generalize on the new data!*

Present your solution to the stake holders. Highlight what you have learned, what worked and what did not, what assumptions were made, and what your system's limitations are.

Launch, Monitor, and Maintain Your System
=========================================

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

A good way to evaluate a model is to use cross-validation that can be performed with different scoring strategies.  

Accuracy
--------

Accuracy is generally not the preffered performance measure for classifiers, especially when you are dealing with a skewed dataset. A much better way to evaluate the performance of a classifier is to look at the *confusion matrix*. 

Confusion Matrix
----------------

The general idea is to count the number of times instances of class A are classified as class B. Each row in the matrix represents an *actual* class, while each column represents a *predicted* class.

| --- | Predicted Negative | Predicted Positive |
| --- | --- | --- |
| **Actual Negative** | TN  | FP  |
| **Actual Positive** | FN  | TP  |

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

4.4. Multilabel Classification
==============================

In some cases you may want your classifier to output multiple classes for each instance. Such a a classification system that outputs multiple binary tags is called a *multilabel classification* system. In general you need only create 2 or more label sets and pass them to an algorithm which supports multilabel classification such as `KNeighborClassifier`.  

There are several ways to evaluate a multilabel classifier, and selecting the right metric really depends on your task. One approach is to measure F1 score for each individual label (or any other classifier metric), then simply average them. This assumes that all labels are equally important, which may not be the case. You can assign a weight to each label.

4.5. Multioutput Classification
===============================

*Multioutput-multiclass* classification is simply a generaluzation of multilabel classification where each label can be multiclass (i.e., it can have more than 2 possible values). To illustrate this, we can build a system that removes noise from images. Notice that the classifier's output is multilabel (one label per pixel) and each label can have multiple values (pixel intensity ranges from 0 to 255).

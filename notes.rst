================
The Fundamentals
================

What
====

ML is the science of programming computers so they can learn from data.

A computer program is said to learn from experience E with respect to some task T and some perfomance measure P, if its perfomans on T, as measured by P, improves with E.

Why
===

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

**Supervised**

The training set includes the desired solutions, called labels (classification and regression).

- k-Nearest Neighbors 
- Linear Regression
- Logistic Regression
- Support Vector Machines
- Decision Trees
- Random Forest
- Neural Networks (except autoencoders and Boltzman machine)

**Unsupervised**

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

**Assosiation rule learning**

Discovering hidden relations between data attributes:

- Apriori
- Eclat

**Semisupervised**

The training set is partially labeled.

- Deep Belief Networks (DBNs) are based on unsupervised component Restricted Boltzman Machines (RBMs),
  stacked on top of one another. The whole system is fine-tuned using supervised learning.

**Reinforcment Learning**

An agent can observe the Environment, select and perform actions, and get rewards in return. It must learn the best strategy, called Policy, to get the most reward over time.

**Batch Learning**

A system must be trained using all available data, offline learning. To incorporate new data we need to learn the system from scratch on the full dataset. Thus it cannot adapt to rapidly changing data. Usually it takes a lot of time, power, memory and computing resources. 

**Incremental Learning**

Train the system incrementally by feeding it data instances sequentially, either individually or in small groups. This is a more reactive solution. Each learning step is fast and cheap, so the system can learn about new data on the fly. To control how fast the system should adapt to changing data the learning rate is used.

**Instance-Based Learning**

The system learns the examples by heart, then generalizes to new cases by using a similarity measure to compare them to the learned examples.

**Model-Based Learning**

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

**No Free Lunch Theorem**

If you make absolutely no assumption about the data, then there is no reason to prefer one model over any other.

==================
End-to-End Example
==================

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


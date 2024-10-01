# Decision Tree Classifier and Regressor
## Decision Trees
Decision trees are versatile machine learning algorithms that can perform both classification and regression tasks. (Scroll [below](#note) to get the basic idea of classification and regression in case unfamiliar with)\
Decision trees are also the fundamental components of random forests, which are among the most powerful machine learning algorithms available today. \
I've tried to implement Decision Tree Classifier and Regressor using the CART (Classification and Regression Trees) algorithm that is used by `scikit-learn` \
CART builds a tree-like structure consisting of nodes and branches. The nodes represent different decision points and the branches the possible outcomes of that decision. \
The leaf nodes contain a predicted class label or value for the target variable. \
The CART algorithm uses a greedy approach to split data (Gini Impurity (or) Entropy for Classification and Mean Squared Error for Regression)

## Decision Tree Classifier
In order to build a decision tree classifier, we need to find something called _Gini Impurity_ (or) _Entropy_ for each potential split in data and then we split the data based on the _Gini Impurity_ (we try to split such that the Gini Impurity is minimum) then we recursively build the tree, i.e., repeat the process of creating branches and leaves until a stopping condition is met (like maximum depth, minimum samples per node, or perfect purity).

For prediction, traverse the tree to predict the class for each sample.

>[!NOTE]
>Gini Impurity is a metric used to classify data <br>
>Gini Impurity varies from 0 to 1, where 0 depicts that all the elements are allied to a certain class or only one class exists and a Gini Impurity close to 1 means a high level of impurity where each class consists of a small fraction of elements. A value of $1- 1/n$ occurs when the elements are uniformly distributed into _n_ classes and each class has a probability of $1/n$ <br>
>Mathematically the Gini Impurity is given by <br> 
> $Gini \space Impurity =  1 - \displaystyle\sum_{i=0}^n p_i ^ 2$


## Decision Tree Regressor
In order to build a decision tree regressor, calculate Mean Squared Error (MSE) (this is the metric for regression tasks), where we evaluate the MSE for splits and choose the one that minimizes the error. Then, split the data (similar to the classification case, we split based on the feature values and thresholds). Now, recursively build the tree.

For regression, the prediction is the mean value of the samples in the leaf nodes.


### NOTE
> _Classifier_: Used to classify data into one or more categories. The goal is to assign the input data to specific, predefined categories. Output is typically a label or a class from a set of predefined options

> _Regressor_: Used to predict a continous output by finding the corelations between dependent and independent variables. The output is a real valued number that can vary within a range

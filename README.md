# logistic-reg-from-scratch

In this project, we will implement the softmax regression for multi-class classification using the MNIST dataset. 

The dataset contains 60K training samples, and 10K test samples. We split 10K from the training samples for validation. Thus, we have 50K training samples, 10K validation samples, and 10K test samples. The target label is among {0, 1, …, 9}.

We will implement stochastic gradient descent (SGD) for cross-entroy loss of softmax as the learning algorithm. The measure of success will be the accuracy (i.e., the fraction of correct predictions).

After reaching the maximum number of iterations, we pick the epoch that yields the best validation performance (the lowest risk), and test the model on the test set.

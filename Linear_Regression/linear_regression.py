import numpy as np 


class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):

        self.lr = lr #learning rate
        self.n_iters = n_iters #iteration of our gradient descent
        self.weights = None 
        self.bias = None 

    def fit(self, X, y):
        #implement gradient_Descent method in order to reduce the error
        n_samples, n_features = X.shape
        #initialize weights with 0 
        self.weights = np.zeros(n_features)
        self.bias = 0 

        for _ in range(self.n_iters):
            
            #calculate the y_predicted in order to calculate the derivative of the weights to reduce the error
            y_predicted = np.dot(X, self.weights) + self.bias

            #derivative_weights = 1/N * sum(x_i*(predicted_y - actual_y))
            #derivative_bias = 1/N * sum(predicted_y - actual_y)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted- y)

            #update rules 
            #new_weights = old_weights - learning rate * derivative_Weights
            #new_bias = old_bias - learning rate * derivative_bias


            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias 
        return y_predicted
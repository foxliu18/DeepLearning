import numpy as np


class LineRegression:

    def __init__(self):
        self.data = None
        self.theta = None
        self.labels = None

    def train(self, alpha, num_iterations = 500):
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        num_examples = self.data.shape[0]
        predictions = LineRegression.hypothesis(self.data, self.theta)
        delta = predictions - self.labels
        theta = self.theta
        theta = theta -alpha * (1/num_examples)*(np.dot(delta.T,  self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        num_examples = self.data.shape[0]
        delta = LineRegression.hypothesis(data, self.theta) - labels
        cost = (1/2)* np.dot(delta.T, delta)
        return cost[0][0]


    @staticmethod
    def hypothesis(self, data, theta):
        predictions = np.dot(data, theta)
        return predictions

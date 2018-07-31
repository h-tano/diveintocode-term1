class LinearModel():

    def __init__(self,X):
        theta = np.random.rand(X.shape[1],1)

    def compute_cost(self,X, y, theta):
        h_x_y = np.dot(X, theta) - y
        m = len(X)
        J_theta = (1/(2 * m)) *  (h_x_y * h_x_y).sum()

        return J_theta

    def gradient_descent(self,X, y, theta, iterations, alpha):
        """
        args:
          alpha: Step size/Learning rate
          iterations: No. of iterations(Number of iterations)
        """
        m = len(X)
        past_costs =[]

        for i in range(iterations):
            h_x_y = np.dot(X , theta) - y
            theta = theta -  (alpha/m) * np.dot(X.T , h_x_y)
            past_costs.append(compute_cost(X, y, theta))

        return past_costs

    def plot_learning_curve(self,iterations,past_costs):
        plt.plot(range(0,iterations),past_costs)
        plt.show()

        return None

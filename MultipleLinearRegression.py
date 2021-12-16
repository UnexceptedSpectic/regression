import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Theta(Enum):
    SLOPE = 0
    INTERCEPT = 1


class MultipleLinearRegression:

    def __init__(self, data_filepath: str, ref_r_var_ind: int, transpose_data: bool = False, use_test_data: bool = False):
        # pyplot config
        plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
        # Load data
        if use_test_data:
            x = np.array([i for i in range(250)])
            y = -4.756 * x + -3.214 + np.random.normal(0, 100, 250)
            self.data = pd.DataFrame(np.array([x, y]))
        else:
            self.data = pd.read_csv(data_filepath, header=0, index_col=0)
        if transpose_data:
            self.data = self.data.transpose()
        # Data now consists of random variables as columns and samples as rows
        # Each random variable represents a dimension of the data
        self.n_dimensions = np.shape(self.data)[1]
        # Initial cost function parameters
        self.slope_thetas = np.repeat(None, self.n_dimensions - 1)
        self.intercept_theta = np.repeat(None, self.n_dimensions - 1)
        # The index of the data dimension with respect to which the linear models will be optimized
        self.ref_r_var_ind = ref_r_var_ind

    def train_model(self):
        """
        Minimize the cost functions described by the sum of least squares between the model line and each dimension of
        the data, by performing gradient descent. Compare the results with those of the numpy polynomial fit function.
        """
        for r_var_ind in range(1, self.n_dimensions):
            # Model regression via numerical solution.
            self.minimize_thetas(r_var_ind=r_var_ind)
            # Compare to numpy solution.
            fit = np.polyfit(self.data.iloc[:, self.ref_r_var_ind], self.data.iloc[:, r_var_ind], 1)
            print('Numpy solution:\nintercept: %s\tslope: %s' % (fit[1], fit[0]))
            # Compare to sklearn solution
            self.plot_sklearn_model(r_var_ind=r_var_ind)

    def minimize_thetas(self, r_var_ind: int):
        """
        Minimize the cost function between a reference random variable and a target random variable, using several
        different initializers as approaches.
        """
        # Attempt three different approaches defined by variable intercept theta starting points. The top-down approach
        # involves setting the intercept theta to the maximum of the data being trained against, the bottom-up approach
        # involves using the respective minimum, and the last approach involves starting at the average of the two.
        y_max = float(np.max(self.data.iloc[:, r_var_ind]))
        y_min = float(np.min(self.data.iloc[:, r_var_ind]))
        td_thetas_cost = [y_max, float(0), None]
        bu_thetas_cost = [y_min, float(0), None]
        m_y_thetas_cost = [np.mean([y_max, y_min]), float(0), None]
        for approach_ind, (int_theta, slope_theta, cost) in enumerate([td_thetas_cost, bu_thetas_cost, m_y_thetas_cost]):
            for theta_being_optimized in [Theta.SLOPE, Theta.INTERCEPT] * 30:
                # Track how many optimization steps are required to minimize the cost function.
                n_iterations = 0
                # Initialize variables for cost function minimization
                cost = 0
                previous_cost = 1000
                alpha = 0.0000001
                alpha_optimized = False
                while np.abs(cost - previous_cost) > 1e-3:
                    n_iterations += 1
                    previous_cost, gradient = self.get_cost_and_gradient(
                        r_var_ind=r_var_ind, slope_theta=slope_theta, approach_ind=approach_ind, int_theta=int_theta
                    )
                    if theta_being_optimized == Theta.SLOPE:
                        gradient = np.dot(gradient, self.data.iloc[:, self.ref_r_var_ind])
                    elif theta_being_optimized == Theta.INTERCEPT:
                        gradient = np.sum(gradient)
                    if not alpha_optimized:
                        # Define the ideal starting ratio of gradient descent step to cost function cost
                        if theta_being_optimized == Theta.SLOPE:
                            step_to_cost = 1e-7
                        elif theta_being_optimized == Theta.INTERCEPT:
                            step_to_cost = 1e-5
                        alpha = np.abs(previous_cost * step_to_cost / gradient)
                        alpha_optimized = True
                    step = alpha * gradient
                    if theta_being_optimized == Theta.SLOPE:
                        slope_theta -= step
                    elif theta_being_optimized == Theta.INTERCEPT:
                        int_theta -= step
                    cost, _ = self.get_cost_and_gradient(
                        r_var_ind, slope_theta=slope_theta, int_theta=int_theta, approach_ind=approach_ind, plot=False
                    )
                    if cost > previous_cost:
                        # Revert to previous values
                        cost = previous_cost
                        if theta_being_optimized == Theta.SLOPE:
                            slope_theta += step
                        elif theta_being_optimized == Theta.INTERCEPT:
                            int_theta += step
                        break
                    # self.get_cost_and_gradient(r_var_ind, slope_theta=slope_theta, int_theta=int_theta, plot=True)
                if approach_ind == 0:
                    td_thetas_cost = int_theta, slope_theta, cost
                elif approach_ind == 1:
                    bu_thetas_cost = int_theta, slope_theta, cost
                elif approach_ind == 2:
                    m_y_thetas_cost = int_theta, slope_theta, cost
                # print('%s number %s found to be %s after %s iterations'
                #       % (theta_being_optimized, r_var_ind, int_theta, n_iterations))
            self.get_cost_and_gradient(
                r_var_ind, slope_theta=slope_theta, int_theta=int_theta, approach_ind=approach_ind, plot=True
            )
        print('Summary of the different starting intercept theta approaches:')
        print('max(y):\t\tintercept: %s\tslope: %s\tcost: %s' % td_thetas_cost)
        print('min(y):\t\tintercept: %s\tslope: %s\tcost: %s' % bu_thetas_cost)
        print('avg(min_max):\tintercept: %s\tslope: %s\tcost: %s)' % m_y_thetas_cost)
        best_result = sorted([td_thetas_cost, bu_thetas_cost, m_y_thetas_cost], key=lambda x: x[2])[0]
        self.intercept_theta[r_var_ind - 1] = best_result[0]
        self.slope_thetas[r_var_ind - 1] = best_result[1]
        print('Best result:\nintercept: %s\tslope: %s\tcost: %s)' % best_result)

    def get_cost_and_gradient(self, r_var_ind: int, slope_theta: float, int_theta: float, approach_ind: int,
                              plot: bool = False) -> (int, int):
        """
        Determine the cost and gradient of the model between a reference dimension defined by self.ref_r_var_ind and
        another dimension as defined by the input r_var_ind. The cost function is the sum of least squares, scaled by
        (1/[2m]), where m is the dimensionality of the data.
        """
        # Find the difference between the model and the actual data. The model is defined as existing between a
        # reference random variable as determined by self.ref_r_var_ind and another random variable, as determined by
        # the input parameter r_var_ind. Modeled Xn[] = slope_theta_n * Xref[] + intercept_theta_n
        ref_x = self.data.iloc[:, self.ref_r_var_ind]
        model = slope_theta * ref_x + int_theta
        actual = self.data.iloc[:, r_var_ind]
        model_actual_diff = model - actual
        cost = np.sum(model_actual_diff * model_actual_diff, axis=0) / self.n_dimensions / 2
        # The pre-gradient associated with the cost function
        gradient = model_actual_diff / self.n_dimensions
        # Plot the line defined by the model
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(self.data.iloc[:, self.ref_r_var_ind], self.data.iloc[:, r_var_ind], color='grey')
            ax.set(
                title='Raw Data vs Regression Model (Approach %s)' % (approach_ind + 1),
                xlabel='Reference Random Variable (%s)' % (self.ref_r_var_ind + 1),
                ylabel='Random Variable %s' % (r_var_ind + 1)
            )
            ax.plot(self.data.iloc[:, self.ref_r_var_ind], model, color='red')
            fig.show()
        return cost, gradient

    def plot_sklearn_model(self, r_var_ind: int):
        x = np.array([self.data.iloc[:, self.ref_r_var_ind], self.data.iloc[:, r_var_ind]]).T
        y = np.dot(x, np.array([0, 1]))
        reg = LinearRegression().fit(x, y)
        fig, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1], color='grey')
        ax.set(
            title='Sklearn Regression Model',
            xlabel='Random Variable 1',
            ylabel='Random Variable %s' % (r_var_ind + 1)
        )
        ax.plot(x[:, 0], reg.predict(x), color='red')
        fig.show()


if __name__ == '__main__':
    mlr = MultipleLinearRegression('./data/high_dim.csv', ref_r_var_ind=0, transpose_data=True, use_test_data=True)
    # Summarize input data
    print(mlr.data.head(3))
    # Find the best-fitting regression line, using alpha as the learning rate
    mlr.train_model()

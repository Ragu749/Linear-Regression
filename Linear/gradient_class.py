import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# This class performs linear regression when data is provided in the format discussed in the documentation.
# See Documentation for more information


def error_gradient(func):
    # This is a decorator for the gradient function. The gradient_single function returns the gradient of
    # the error function at a particular data point. This wrapper sums for all data points.
    def wrapper(*args):

        beta_shape = args[1].shape
        data_shape = args[0].shape
        grad = np.zeros((beta_shape[0],  beta_shape[1]))

        for i in range(data_shape[0]):
            grad += func(args[0][i], args[1])

        return grad

    return wrapper


@error_gradient
def gradient_single(data, beta):

    # This function produces the gradient of the error function at the data point given by 'data'
    # It used the coefficients beta in order to calculate the error function and returns an array
    # denoting the gradient

    beta_shape = beta.shape
    gradient = np.ones((beta_shape[0], beta_shape[1]))

    temp_array = np.dot(beta[:, : beta_shape[1] - 1], data[:beta_shape[1] - 1])

    for i in range(beta_shape[0]):
        for j in range(beta_shape[1]):
            if j != beta_shape[1] - 1:
                gradient[i, j] = temp_array[i]
                gradient[i, j] += beta[i, beta_shape[1] - 1] - data[beta_shape[1] - 1 + i]
                gradient[i, j] *= 2*data[j]
            elif j == beta_shape[1] - 1:
                gradient[i, j] = temp_array[i]
                gradient[i, j] += beta[i, beta_shape[1] - 1] - data[beta_shape[1] - 1 + i]
                gradient[i, j] *= 2

    return gradient


class GradientClass:

    def __init__(self, data=None, shape_dep=None, tolerance=0.00001, time_step=0.0001):

        # Constructor stores the data and shape of array as well as the no. of dependent and independent
        # variables.
        # Also rescales the data between 0 and 1 and stores the factors and the rescaled array

        if data is None:
            print('Please provide Data required to initialise regression!')
        elif shape_dep is None:
            print('No input for the dependent and independent variables has been provided!')
        elif shape_dep[0] + shape_dep[1] != data.shape[1]:
            print("Error! The number of independent and dependent variables does not match the data.")
            sys.exit()
        else:
            self.shape = data.shape
            rescale_array = np.copy(data)
            rescale_factors = np.full((2, self.shape[1]), -1)

            for i in range(self.shape[1]):
                min_val = np.amin(data[:, i])
                rescale_array[:, i] -= np.ones(self.shape[0]) * min_val
                max_val = np.amax(rescale_array[:, i])
                rescale_array[:, i] *= 1 / max_val
                rescale_factors[0, i] = min_val
                rescale_factors[1, i] = max_val

            self.rescaled_data = rescale_array
            self.rescale_factors = rescale_factors
            self.shape_dep = shape_dep
            self.data = data
            self.final_regression = 0
            self.scaled_regression = 0

            self.tolerance = tolerance
            self.time_step = time_step

    def __repr__(self):

        # Outputs the size of the data and the number of dependent and independent variables

        output = 'This is an ' + str(self.shape[0]) + ' x ' + str(self.shape[1]) + ' array '
        output += 'with ' + str(self.shape_dep[1]) + ' independent and ' + str(self.shape_dep[0])
        output += ' dependent variables'

        return output

    def scaled_data_out(self):

        # Outputs the re-scaled data to file path\rescaled.dat

        np.savetxt(os.getcwd() + r'\rescaled.dat', self.rescaled_data)

    def starting_point(self):

        # Method to provide a default starting point for the gradient descent
        # Tries to determine the slope of the data and fixes an appropriate starting point

        start = np.zeros((self.shape_dep[0], self.shape_dep[1] + 1))
        start[:, self.shape_dep[1]] = 0.5

        for dep in range(self.shape_dep[0]):
            index = np.argmax(self.rescaled_data[:, self.shape_dep[1] + dep])
            for i in range(self.shape_dep[1]):
                if self.rescaled_data[index, i] > 0.5:
                    start[dep - 1, i] = 1
                else:
                    start[dep - 1, i] = -1

        return start

    def scaled_grad_descent(self, beta=None):

        # Performs the gradient descent on the error function to find the linear coefficients
        # for the scaled data
        # N.B can specify starting point - if not uses default starting point

        if beta is None:
            beta = self.starting_point()

        grad = gradient_single(self.rescaled_data, beta)
        n = 10000

        temp_array = np.copy(beta)

        for i in range(n):

            correction = grad * self.time_step
            convergence = np.absolute(correction)
            temp_array -= correction

            if np.amax(convergence) < self.tolerance:
                break

            grad = gradient_single(self.rescaled_data, temp_array)
            if i == n - 1:
                print('Finished Iterations:')

        self.scaled_regression = temp_array
        return temp_array

    def regression(self, beta=None):

        # Re-scales the scaled regression back to the full data set and returns the final array of
        # coefficients
        # N.B can specify starting point - if not uses default starting point

        scaled = self.scaled_grad_descent(beta)
        reg = np.zeros_like(scaled)
        res = self.rescale_factors

        for dep in range(self.shape_dep[0]):
            for i in range(self.shape_dep[1]):
                reg[dep, self.shape_dep[1]] -= scaled[dep, i] * res[0, i] / res[1, i]
                reg[dep, i] = scaled[dep, i] / res[1, i] * res[1, self.shape_dep[1] + dep]

            reg[dep, self.shape_dep[1]] += scaled[dep, self.shape_dep[1]]
            reg[dep, self.shape_dep[1]] *= res[1, self.shape_dep[1] + dep]
            reg[dep, self.shape_dep[1]] += res[0, self.shape_dep[1] + dep]

        self.final_regression = reg
        return reg

    def polynomial(self, dep, beta=None):

        # Function that uses the linear regression to output a string showing the polynomial representaiton
        # for the dependent variable 'dep'
        # N.B can specify starting point - if not uses default starting point

        if isinstance(self.final_regression, int):
            self.regression(beta)

        poly = "y_" + str(dep) + ' = ' + str(self.final_regression[dep - 1, self.shape_dep[1]]) + ' '

        for i in range(self.shape_dep[1]):
            temp = '{:+}'.format(self.final_regression[dep - 1, i]) + 'x_{}'.format(i + 1) + ' '
            poly += temp[0] + ' ' + temp[1:]

        print(poly)

    def plot_graphs(self, dep=None, beta=None):

        # If the system has one independent variable, this function plots the linear fit against the data
        # where the choice of dependent variable is specified by 'dep'
        # N.B can specify starting point - if not uses default starting point

        assert dep is not None, 'Please specify dependent'
        assert self.shape_dep[1] == 1, 'The system is multidimensional so we cannot plot'

        if isinstance(self.final_regression, int):
            self.regression(beta)

        temp_reg = self.final_regression

        x_var = [self.data[j, 0] for j in range(self.shape[0])]

        y_var = [self.data[j, dep] for j in range(self.shape[0])]
        fit = [temp_reg[dep - 1, self.shape_dep[1]] for j in range(self.shape[0])]

        for j in range(self.shape[0]):
            fit[j] += np.dot(temp_reg[dep - 1, :self.shape_dep[1]], self.data[j, :self.shape_dep[1]].transpose())

        plt.plot(x_var, fit)
        plt.scatter(x_var, y_var)

        plt.tight_layout()
        plt.savefig('plots.pdf')

    def scaled_plot_graphs(self, dep=None, beta=None):

        # If the system has one independent variable, this function plots the scaled linear fit against the
        # scaled data where the choice of dependent variable is specified by 'dep'
        # This is mainly for error testing
        # N.B can specify starting point - if not uses default starting point

        assert dep is not None, 'Please specify dependent'
        assert self.shape_dep[1] == 1, 'The system is multidimensional so we cannot plot'

        if isinstance(self.scaled_regression, int):
            self.scaled_grad_descent(beta)

        temp_reg = self.scaled_regression
        fig = plt.figure(figsize=(15, 15))

        x_var = [self.rescaled_data[j, 0] for j in range(self.shape[0])]

        y_var = [self.rescaled_data[j, dep] for j in range(self.shape[0])]
        fit = [temp_reg[dep - 1, self.shape_dep[1]] for j in range(self.shape[0])]

        for j in range(self.shape[0]):
            data_temp = self.rescaled_data[j, :self.shape_dep[1]].transpose()
            fit[j] += np.dot(temp_reg[dep - 1, :self.shape_dep[1]], data_temp)

        ax = fig.add_subplot(1, 1, 1)
        ax.axis('equal')
        ax.plot(x_var, fit)
        ax.scatter(x_var, y_var)

        plt.tight_layout()
        plt.savefig("scaled_plots.pdf")
        # plt.show()

    def error(self, beta=None):

        # Outputs the error for each dependent variable to a file path\error.dat
        # N.B can specify starting point - if not uses default starting point

        if isinstance(self.final_regression, int):
            self.regression(beta)

        reg = self.final_regression
        temp = np.zeros(self.shape_dep[0])

        for i in range(self.shape[0]):
            temp += np.dot(reg[:, : self.shape_dep[1]], self.data[i, : self.shape_dep[1]].transpose())
            temp += reg[:, self.shape_dep[1]].transpose()
            temp -= self.data[i, self.shape_dep[1]:]
            temp = np.absolute(temp)

        np.savetxt(os.getcwd() + r'\error.dat', temp/self.shape[0])

    def error_scaled(self, beta=None):

        # Outputs the error for the scaled data to a file path\scaled_error.dat
        # N.B can specify starting point - if not uses default starting point

        if isinstance(self.scaled_regression, int):
            self.scaled_grad_descent(beta)

        reg = self.scaled_regression
        temp = np.zeros(self.shape_dep[0])

        for i in range(self.shape[0]):
            temp += np.dot(reg[:, : self.shape_dep[1]], self.rescaled_data[i, : self.shape_dep[1]].transpose())
            temp += reg[:, self.shape_dep[1]].transpose()
            temp -= self.rescaled_data[i, self.shape_dep[1]:]
            temp = np.absolute(temp)

        np.savetxt(os.getcwd() + r'\scaled_error.dat', temp/self.shape[0])



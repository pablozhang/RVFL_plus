import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, chi2_kernel, polynomial_kernel, \
                                        additive_chi2_kernel, laplacian_kernel
from sklearn.preprocessing import OneHotEncoder

class KRVFL_plus(object):
    """
    The code of kernel random vector functional link-plus (KRVFL+) network is implemented in Python. If you would like
    to use it in your researches, please cite the paper "Zhang, Peng-Bo, and Zhi-Xin Yang. A new learning paradigm for 
    random vector functional-link network: RVFL+. Neural Networks 122 (2019) pp.94-105".

    Parameters
    ----------
    kernel_name : default='rbf', the kernel function is used in KRVFL+. Please select a kernel function from the below
    set: {'rbf', 'linear', 'add_chi2', 'chi2', 'poly', 'laplace'}, and other kernel function is not supported now. In
         the later, we will add more kernel functions.
    type: default='classification', classification or regression,


    Example
    --------
    model = KRVFL_plus()
    model.fit(train_x, addition_x, train_y)
    y_hat = model.predict(train_x, test_x)
    """

    def __init__(self, kernel_name = 'rbf', type = 'classification'):

        self.kernel_name = kernel_name
        self.type = type

    def _generate_kernel_matrix(self, x, kernel_name, y = None):

        if y is None:
            if kernel_name == 'rbf':
                return rbf_kernel(x)
            elif kernel_name == 'linear':
                return linear_kernel(x)
            elif kernel_name == 'add_chi2':
                return additive_chi2_kernel(x)
            elif kernel_name == 'chi2':
                return chi2_kernel(x)
            elif kernel_name == 'poly':
                return polynomial_kernel(x)
            elif kernel_name == 'laplace':
                return laplacian_kernel(x)
        else:

            assert x.shape[1] == y.shape[1]
            if kernel_name == 'rbf':
                return rbf_kernel(x, y)
            elif kernel_name == 'linear':
                return linear_kernel(x,y)
            elif kernel_name == 'add_chi2':
                return additive_chi2_kernel(x, y)
            elif kernel_name == 'chi2':
                return chi2_kernel(x, y)
            elif kernel_name == 'poly':
                return polynomial_kernel(x, y)
            elif kernel_name == 'laplace':
                return laplacian_kernel(x, y)


    def _transform_label(self, y):
        enc = OneHotEncoder(handle_unknown='ignore')
        try:
            target = enc.fit_transform(y).toarray()
            print('the label can be transformed directly using onehotencoder')
        except:
            target = enc.fit_transform(y.reshape(-1, 1)).toarray()
            print('the label must be reshaped before being transformed')
        return target

    def fit(self, train_x, addition_x, train_y, C = 0.1, gamma = 1000):
        """
        Params:
        ---------
        :param train_x: a NumofSamples * NumofFeatures matrix, training data
        :param addition_x: a NumofSamples * NumofPFeatures matrix, the privileged information data
        :param train_y: training label
        :param C: default=0.1, the penelty parameter
        :param gamma: default = 1000, the trade-off parameter balancing between training data and privileged data
        """
        omega1 = self._generate_kernel_matrix(train_x, 'linear')
        omega2 = self._generate_kernel_matrix(train_x, self.kernel_name)
        omega_p_1 = self._generate_kernel_matrix(addition_x, 'linear')
        omega_p_2 = self._generate_kernel_matrix(addition_x, self.kernel_name)
        if self.type == 'classification':
            one_hot_target = self._transform_label(train_y)
        elif self.type == 'regression':
            one_hot_target = train_y
        else:
            raise Exception("The type is not supported now! please select classification or regression.")
        self.beta = np.linalg.inv(omega1 + omega2 + 1/gamma*(omega_p_1 + omega_p_2) + np.eye(train_x.shape[0])/C).\
            dot(one_hot_target - C / gamma * (omega_p_1 + omega_p_2).dot(np.ones(one_hot_target.shape)))


    def predict(self, train_x, test_x):
        """

        :param train_x:  a NumofSamples * NumofFeatures matrix, training data, building the kernel matrix
        :param test_x: a NumofTestSamples * NumofFeatures matrix, test data
        :return: y_hat, the predicted labels
        """
        omega_test_1 = self._generate_kernel_matrix(test_x, kernel_name= 'linear', y = train_x)
        omega_test_2 = self._generate_kernel_matrix(test_x, kernel_name= self.kernel_name, y = train_x)
        y_hat_temp = (omega_test_1 + omega_test_2).dot(self.beta)
        if self.type == 'classification':
            y_hat = np.argmax(y_hat_temp, axis=1)
            return y_hat
        else:
            return y_hat_temp

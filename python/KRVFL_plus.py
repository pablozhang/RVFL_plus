import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, chi2_kernel, polynomial_kernel, \
                                        additive_chi2_kernel, laplacian_kernel
from sklearn.preprocessing import OneHotEncoder

class KRVFL_plus(object):
    """
    The code of kernel random vector functional link-plus (KRVFL+) network is implemented in Python. If you would like
    to use it in your researches, please cite the paper "Zhang, Peng-Bo, and Yang, Zhi-Xin. A new learning paradigm for
    random vector functional-link network: RVFL+. Neural Networks 122 (2020) pp.94-105".

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
        self.kernel_dict = {"rbf": lambda x, y = None: rbf_kernel(x, y),
                            "linear": lambda x, y = None: linear_kernel(x, y),
                            "add_chi2": lambda x, y = None: additive_chi2_kernel(x, y),
                            "chi2": lambda x, y = None: chi2_kernel(x, y),
                            "poly": lambda x, y = None: polynomial_kernel(x, y),
                            "laplace": lambda x, y = None: laplacian_kernel(x, y)}

    def _transform_label(self, y):
        enc = OneHotEncoder(handle_unknown='ignore')
        try:
            target = enc.fit_transform(y).toarray()
            print('the label can be transformed directly using onehotencoder')
        except:
            target = enc.fit_transform(y.reshape(-1, 1)).toarray()
            print('the label must be reshaped before being transformed')
        return target

    def _softmax(self, x):

        max_value = np.max(x,axis=1,keepdims=True)
        e_x = np.exp(x - max_value) 
        sum_value = np.sum(e_x,axis=1,keepdims=True)
        y = e_x / sum_value
        return y

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
        omega1, omega2 = self.kernel_dict["linear"](train_x), self.kernel_dict[self.kernel_name](train_x)
        omega_p_1, omega_p_2 = self.kernel_dict["linear"](addition_x), self.kernel_dict[self.kernel_name](addition_x)
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
        y_hat = self.predict_proba(train_x, test_x)
        if self.type == 'classification':
            y_hat = np.argmax(y_hat, axis=1)
            return y_hat
        else:
            return y_hat

    def predict_proba(self, train_x, test_x):

        omega_test_1,  omega_test_2 = self.kernel_dict["linear"](test_x, train_x), \
                                      self.kernel_dict[self.kernel_name](test_x, train_x)
        y_hat_temp = (omega_test_1 + omega_test_2).dot(self.beta)
        if self.type == "classification":

            y_hat_prob = self._softmax(y_hat_temp)
            return y_hat_prob
        else:
            return y_hat_temp

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from numpy.linalg import multi_dot


class RVFL_plus(object):
    """
    The code of random vector functional link-plus (RVFL+) network is implemented in Python. If you would like
    to use it in your researches, please cite the paper ''

    Parameters
    ----------

    hidden_node : default = 100, the number of enhancement node between the input layer and the hidden layer
    random_type :default = 'uniform', please select random type from "uniform" or "gaussian"
    scale_value : default=1.0, a positive value for initialing reasonably weights and biases in the first hidden layer
    activation_name : default = 'sigmoid', please select an activation function from the below set: {'sigmoid', 'tanh',
                     'sin', 'hardlim', 'softlim', 'gaussianRBF', 'multiquadricRBF', 'inv_multiquadricRBF', 'tribas',
                       'inv_tribas'}
    type: default='classification', classification or regression

    Example
    --------
    model = RVFL_plus()
    model.fit(train_x, addition_x, train_y)
    y_hat = model.predict(test_x)
    """
    def __init__(self, hidden_node=100, random_type="uniform",
                 scale_value=1.0, activation_name="sigmoid", type="classification"):

        self.hidden_node = hidden_node
        self.random_type = random_type
        self.scale_value = scale_value
        self.activation_name = activation_name
        self.type = type


    def _generate_randomlayer(self, X_train, X_addition):
        num_samples, num_feas = X_train.shape
        _, num_pfeas = X_addition.shape
        # np.random.seed(0)
        if self.random_type == 'uniform':
            weights = (np.random.rand(num_feas, num_samples) * 2.0 - 1.0) * self.scale_value
            biases = np.random.rand(1, num_samples) * self.scale_value
            pf_weights = (np.random.rand(num_pfeas, num_samples) * 2.0 - 1.0) * self.scale_value

        elif self.random_type == 'gaussian':
            weights = np.random.randn(num_feas, num_samples)
            biases = np.random.randn(1, num_samples)
            pf_weights = np.random.randn(num_feas, num_samples)

        else:
            raise Exception('The random type is not supported now!')

        return weights, biases, pf_weights

    def _activation_function(self, x, activation_name):

        if activation_name == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        elif activation_name == 'sin':
            return np.sin(x)
        elif activation_name == 'tanh':
            return np.tanh(x)
        elif activation_name == 'hardlim':
            return np.array(x > 0.0, dtype=float)
        elif activation_name == 'softlim':
            return np.clip(x, 0.0, 1.0)
        elif activation_name == 'gaussianRBF':
            return np.exp(-pow(x, 2.0))
        elif activation_name == 'multiquadricRBF':
            return np.sqrt(1.0 + pow(x, 2.0))
        elif activation_name == 'inv_multiquadricRBF':
            return 1.0 / (np.sqrt(1.0 + pow(x, 2.0)))
        elif activation_name == 'tribas':
            return np.clip(1.0 - np.fabs(x), 0.0, 1.0)
        elif activation_name == 'inv_tribas':
            return np.clip(np.fabs(x), 0.0, 1.0)

        else:
            raise Exception('The activation function is not supported now!')

    def _transform_label(self, y):
        enc = OneHotEncoder(handle_unknown='ignore')
        try:
            target = enc.fit_transform(y).toarray()
            print('the label can be transformed directly using onehotencoder')
        except:
            target = enc.fit_transform(y.reshape(-1, 1)).toarray()
            print('the label must be reshaped before being transformed')
        return target

    def fit(self,train_x, addition_x, train_y, gamma=1000, C=0.1):

        """
        Params:
        -------
        :param train_x: a NumofSamples * NumofFeatures matrix, training data
        :param addition_x: a NumofSamples * NumofPFeatures matrix, the privileged information data
        :param train_y: training label
        :param gamma: default = 1000, the trade-off parameter balancing between training data and privileged data
        :param C: default=0.1, the penelty parameter

        """

        self.weights, self.biases, self.pf_weights = self._generate_randomlayer(train_x, addition_x)
        train_g = train_x.dot(self.weights) + self.biases
        train_pg = addition_x.dot(self.pf_weights) + self.biases
        H, PH = self._activation_function(train_g, activation_name=self.activation_name), \
                self._activation_function(train_pg, activation_name=self.activation_name)
        H, PH = np.hstack((H, train_x)), np.hstack((PH, addition_x))
        if self.type == 'classification':
            one_hot_target = self._transform_label(train_y)
        elif self.type == 'regression':
            one_hot_target = train_y
        else:
            raise Exception("The type is not supported now! please select classification or regression.")
        part_a = np.linalg.inv(H.dot(H.T) + 1 / gamma * PH.dot(PH.T) + np.eye(train_x.shape[0]) / C)
        part_b = one_hot_target - C / gamma * multi_dot([PH, PH.T, np.ones(one_hot_target.shape)])
        self.beta = multi_dot([H.T, part_a, part_b])

    def predict(self, test_x):
        """
        Params:
        -------
        :param test_x: a NumofTestSamples * NumofFeatures matrix, test data
        :return: y_hat, the predicted labels
        """
        test_g = test_x.dot(self.weights) + self.biases
        test_h = self._activation_function(test_g, activation_name=self.activation_name)
        test_h = np.hstack((test_h, test_x))
        y_hat_temp = test_h.dot(self.beta)
        if self.type == 'classification':
            y_hat = np.argmax(y_hat_temp, axis=1)
            return y_hat
        else:
            return y_hat_temp

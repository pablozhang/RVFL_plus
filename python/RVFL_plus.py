from numpy.linalg import multi_dot
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class RVFL_plus(object):
    """
    The code of random vector functional link-plus (RVFL+) network is implemented in Python. If you would like
    to use it in your researches, please cite the paper "Zhang, Peng-Bo, and Yang, Zhi-Xin. A new learning paradigm
    for random vector functional-link network: RVFL+. Neural Networks, 122 (2020) pp.94-105"
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
    def __init__(self, hidden_node=1000, random_type="uniform",
                 scale_value=1.0, activation_name="sigmoid", task_type="classification"):

        self.hidden_node = hidden_node
        self.random_type = random_type
        self.scale_value = scale_value
        self.activation_name = activation_name
        self.type = task_type
        self._activation_dict = {'sigmoid': lambda x : 1.0 / (1.0 + np.exp(-x)),
                                 "sin": lambda x: np.sin(x),
                                 "tanh": lambda x: np.tanh(x),
                                 "hardlim": lambda x:np.array(x > 0.0, dtype=float),
                                 "softlim":lambda x:np.clip(x, 0.0, 1.0),
                                 "gaussianRBF": lambda x:np.exp(-pow(x, 2.0)),
                                 "multiquadricRBF": lambda x:np.sqrt(1.0 + pow(x, 2.0)),
                                 "inv_multiquadricRBF": lambda x : 1.0 / (np.sqrt(1.0 + pow(x, 2.0))),
                                 "tribas": lambda x: np.clip(1.0 - np.fabs(x), 0.0, 1.0),
                                 "inv_tribas": lambda x: np.clip(np.fabs(x), 0.0, 1.0)}


    def _generate_randomlayer(self, X_train, X_addition):
        num_samples, num_feas = X_train.shape
        _, num_pfeas = X_addition.shape
        # np.random.seed(0)
        if self.random_type == 'uniform':
            weights = (np.random.rand(self.hidden_node, num_feas) * 2.0 - 1.0) * self.scale_value 
            biases = np.random.rand(self.hidden_node, 1) * self.scale_value
            pf_weights = (np.random.rand(self.hidden_node, num_pfeas) * 2.0 - 1.0) * self.scale_value

        elif self.random_type == 'gaussian':
            weights = np.random.randn(self.hidden_node, num_feas)
            biases = np.random.randn(self.hidden_node, 1)
            pf_weights = np.random.randn(self.hidden_node, num_pfeas)

        else:
            raise Exception('The random type is not supported now!')

        return weights, biases, pf_weights

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
        out = np.exp(x)
        return out/ np.sum(out, axis=1, keepdims=True)

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
        train_g = self.weights.dot(train_x.T) + self.biases
        train_pg = self.pf_weights.dot(addition_x.T) + self.biases
        try:
            H, PH = self._activation_dict[self.activation_name](train_g), \
                    self._activation_dict[self.activation_name](train_pg)
        except:
            raise Exception('The activation function is not supported now!')

        H, PH = np.concatenate([H, train_x.T]), np.concatenate([PH, addition_x.T])
        if self.type == 'classification':
            one_hot_target = self._transform_label(train_y)
        elif self.type == 'regression':
            one_hot_target = train_y
        else:
            raise Exception("The type is not supported now! please select classification or regression.")
        part_a = np.linalg.inv(H.T.dot(H) + 1 / gamma * PH.T.dot(PH) + np.eye(train_x.shape[0]) / C)
        part_b = one_hot_target - C / gamma * multi_dot([PH.T, PH, np.ones(one_hot_target.shape)])
        self.beta = multi_dot([H, part_a, part_b])

    def predict(self, test_x):
        """
        Params:
        -------
        :param test_x: a NumofTestSamples * NumofFeatures matrix, test data
        :return: y_hat, the predicted labels
        """
        y_hat = self.predict_proba(test_x)
        if self.type == 'classification':
            y_hat = np.argmax(y_hat, axis=1)
            return y_hat
        else:
            return y_hat

    def predict_proba(self, test_x):
        test_g = self.weights.dot(test_x.T) + self.biases
        test_h = self._activation_dict[self.activation_name](test_g)
        test_h = np.concatenate([test_h, test_x.T])
        y_hat_temp = test_h.T.dot(self.beta)
        if self.type == "classification":
            y_hat_prob = self._softmax(y_hat_temp)
            return y_hat_prob
        else:
            return y_hat_temp

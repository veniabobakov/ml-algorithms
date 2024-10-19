import numpy as np
import pandas as pd

"""
This module created to work with machine learning models.
It's similar with popular module like scikit-learn.    

Classes include:
    - MyLineReg: a class of linear regression.
"""


class MyLineReg:
    """
    Linear regression class.

    Typical use:
        model = MyLineReg()
        model.fit(x, y, True)

    Attributes:
        - __n_iter: A count of gradient iterations.
        - __weights: A numpy array of model weights.
        - __learning_rate: A multiplier of gradient step.

    Methods:
        - get_coef: return weights of a fitting model.
        - fit: fitting a model to find the best weights.
        - predict: return predicted values of the model.
        - my_logging: loging of fitting model.
    """

    def __init__(self, n_iter: int = 100, learning_rate: float = 0.1) -> None:
        self.__n_iter: int = n_iter
        self.__weights: np.array = None
        self.__learning_rate: float = learning_rate

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool | int = False) -> None:
        """
        Fitting model by a samples.

        :param X: features (matrix of samples) for fitting.
        :param y: target variable.
        :param verbose: logging flag.
        :return: None.
        """

        # Adding the bias column:
        X.insert(0, "w0", 1.0)
        self.__weights = np.ones(X.shape[1])

        for step in range(1, self.__n_iter + 1):
            # Predict y:
            head_y = X @ self.__weights

            # Calculate MSE loss function:
            mse = np.mean((head_y - y) ** 2)

            # Calculate gradient of loss function:
            gradient = (2 / len(y)) * (head_y - y) @ X

            # Update model weights:
            self.__weights -= gradient * self.__learning_rate

            self.my_logging(step=step, mse=mse, verbose=verbose)

    def predict(self, X: pd.DataFrame):
        X.insert(0, "w0", 1.0)
        y_head = X @ self.__weights
        return y_head


    def get_coef(self) -> np.array:
        """
        Return the weights of fitting model.

        :return: numpy.array
        """
        return self.__weights[1:]

    def __repr__(self) -> str:
        return f"""
                    Linear regression class.

                    Typical use:
                        model = MyLineReg()
                        model.fit(x, y, True)

                    Attributes:
                        - __n_iter: A count of gradient iterations.
                        - __weights: A numpy array of model weights.
                        - __learning_rate: A multiplier of gradient step.

                    Methods:
                        - get_coef: return weights of a fitting model.
                        - fit: fitting a model to find the best weights.
                """

    def __str__(self) -> str:
        return f"MyLineReg class: n_iter={self.__n_iter}, learning_rate={self.__learning_rate}"

    @staticmethod
    def my_logging(step: int, mse: float, verbose: object = False) -> None:
        """
        Print MSE every n step.

        :param step: the size of step.
        :param mse: the current MSE value.
        :param verbose: a flag of logging.
        :return: None
        """

        if verbose > 0:
            if step == 1:
                print(f"| STARTING MSE IS | {mse} |")
            elif not (step % verbose):
                print(f"| MSE IS | {mse} |")
from timewarpy import preprocess


class UnivariateTS:
    """Core object for processing univariate time series. Use this object
    to set up and save all necessary transformations for pre and post model
    processing of a time series that only contains one dimension outside of time.
    """

    def __init__(self, train_horizon: int, pred_horizon: int, scaler: object = None,
                 roll_increment: int = 0):
        """Initializes the core class. First, this sets the values for how many
        time points should be in the training and forecasting windows. See [here](/#univariate-data)
        for a visual on the training and prediction window lengths. Second, this
        also defines any scaling that needs to occur the variable changing in time.
        Scaling functionality follows the scikit-learn standards for methods needed. See an example
        of a standard scaler [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

        Args:
            train_horizon (int): number of time steps in each training vector
            pred_horizon (int): number of time steps in each prediction vector
            scaler (object, optional): scaling function to use, usage follows scikit-learn. Defaults to None.
            roll_increment (int, optional): how many time sets to skip while rolling windows. Defaults to 0.
        """
        self.__str__ = 'Univariate Time Series Processing Class'
        self.train_horizon = train_horizon
        self.pred_horizon = pred_horizon
        self.roll_increment = roll_increment
        if scaler is not None:
            self.scaler = scaler()
        else:
            self.scaler = None

    def fit(self, df, column):
        """Given a pandas dataframe and column for the univariate
        time series data, this will fit necessary preprocessing to that
        given data column. Currently this is only fitting the scalar to
        the given column in the __init__ function.

        Args:
            df (pandas.DataFrame): univariate time series
            column (str): column to use in the dataframe
        """
        if self.scaler is not None:
            self.scaler.fit(df[column].to_numpy().reshape(-1, 1))

    def transform(self, df, column):
        """Given a pandas dataframe and column for the univariate
        time series data, tranform the data to a neural network friendly
        set of vectors.

        Args:
            df (pandas.DataFrame): univariate time series
            column (str): column to use in the dataframe

        Returns:
            tuple: X (np.array) training vectors, y (np.array) forecasting/prediction vectors
        """
        time_series = self.scaler.transform(df[[column]])
        X, y = preprocess.create_univariate_windows(
            time_series, self.train_horizon, self.pred_horizon
        )
        return X, y

    def fit_transform(self, df, column):
        self.fit(df, column)
        X, y = self.transform(df, column)
        return X, y

    def inverse_transform(self):
        return None

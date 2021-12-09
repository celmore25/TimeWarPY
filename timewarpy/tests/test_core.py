from timewarpy import datasets, core
from sklearn.preprocessing import MinMaxScaler


def test_UnivariateTS_fit():
    df = datasets.load_energy_data()
    TSprocessor = core.UnivariateTS(1680, 240, scaler=MinMaxScaler)
    TSprocessor.fit(df, 'Appliances')
    assert TSprocessor.scaler.data_max_ is not None


def test_UnivariateTS_fit_transform():
    df = datasets.load_energy_data()
    TSprocessor = core.UnivariateTS(1680, 240, scaler=MinMaxScaler)
    TSprocessor.fit(df, 'Appliances')
    X, y = TSprocessor.transform(df, 'Appliances')
    assert X.shape == (17816, 1680, 1)


# def test_UnivariateTS_transform():
#     df = datasets.load_energy_data()
#     TSprocessor = core.UnivariateTS(1680, 240, scaler=MinMaxScaler)
#     TSprocessor.fit(df, 'Appliances')
#     assert TSprocessor.scaler.data_max_ is not None

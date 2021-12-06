from .. import preprocess


def test_load_energy_data():
    assert preprocess.load_energy_data().shape == (19735, 12)

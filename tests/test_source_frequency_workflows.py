"""Source descriptions and Bruker frequencies from independent synthetic headers."""

from mrax.io.bruker_frequency import bruker_imaging_frequency


def test_frequency_uses_nominal_headers_and_supports_visu_only_metadata():
    assert bruker_imaging_frequency({"PVM_FrqWork": [400.15]}, {}, {"BF1": 400.12}) == 400.12
    assert bruker_imaging_frequency({"PVM_FrqWork": [400.15], "PVM_FrqRef": [400.12]}) == 400.12
    assert bruker_imaging_frequency({}, {"VisuAcqImagingFrequency": 400.12}) == 400.12
    assert bruker_imaging_frequency({"PVM_FrqWork": [float("nan"), 400.12]}) == 400.12
    assert bruker_imaging_frequency({"PVM_FrqWork": [False, -1, 0, float("inf")]}) is None

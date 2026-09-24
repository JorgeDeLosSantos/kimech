from importlib.metadata import version

import kimech


def test_runtime_version_matches_distribution_metadata():
    assert kimech.__version__ == version("kimech")


def test_release_version_is_0_5_0():
    assert kimech.__version__ == "0.5.0"

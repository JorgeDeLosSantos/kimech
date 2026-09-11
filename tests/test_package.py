from importlib.metadata import version

import kimech


def test_runtime_version_matches_distribution_metadata():
    assert kimech.__version__ == version("kimech")

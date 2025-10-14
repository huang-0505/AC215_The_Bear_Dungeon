import sys

def test_python_version():
    v = sys.version_info
    assert f"{v.major}.{v.minor}" == "3.11"

def test_libs_import():
    import transformers  # noqa: F401
    import datasets      # noqa: F401
    import numpy as np   # noqa: F401
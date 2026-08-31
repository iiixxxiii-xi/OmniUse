import minicua
from minicua.core.errors import CUAError, StaleElementError


def test_package_imports():
    assert minicua.__version__ == "0.1.0"


def test_core_errors_import():
    assert issubclass(StaleElementError, CUAError)

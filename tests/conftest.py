import pytest


@pytest.fixture(scope="session")
def root_directory(request):
    return str(request.config.rootdir)

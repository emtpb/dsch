from dsch import helpers


def test_backend_module():
    import dsch.backends.npz
    backend_module = helpers.backend_module('npz')
    assert backend_module == dsch.backends.npz

from collections import namedtuple
import importlib
import pytest
from dsch import frontend, schema


backend_data = namedtuple('backend_data', ('module', 'storage_path'))


@pytest.fixture(params=('hdf5', 'mat', 'npz'))
def backend(request, tmpdir):
    backend = backend_data(
        module=importlib.import_module('dsch.backends.' + request.param),
        storage_path=str(tmpdir.join('test_frontend.' + request.param))
    )
    return backend


def test_create(backend):
    schema_node = schema.Bool()
    storage = frontend.create(backend.storage_path, schema_node)
    assert isinstance(storage, backend.module.Storage)
    assert storage.schema_node == schema_node


def test_load(backend):
    schema_node = schema.Bool()
    storage = frontend.create(backend.storage_path, schema_node)
    storage.data.replace(True)
    storage.save()

    new_storage = frontend.load(backend.storage_path)
    assert isinstance(new_storage, backend.module.Storage)
    assert new_storage.data.value is True


def test_load_require_schema(backend):
    schema_node = schema.Bool()
    storage = frontend.create(backend.storage_path, schema_node)
    storage.data.replace(True)
    storage.save()

    schema_hash = storage.schema_hash()
    frontend.load(backend.storage_path, require_schema=schema_hash)


def test_load_validation_fail(backend):
    schema_node = schema.String(max_length=3)
    storage = frontend.create(backend.storage_path, schema_node)
    storage.data.replace('spam')
    storage.save()

    with pytest.raises(schema.ValidationError):
        frontend.load(backend.storage_path)
    # With force=True, no exception must be raised.
    frontend.load(backend.storage_path, force=True)

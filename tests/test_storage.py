from collections import namedtuple
import importlib
import pytest
from dsch import schema


backend_data = namedtuple('backend_data', ('module', 'storage_path'))


@pytest.fixture(params=('hdf5', 'mat', 'npz'))
def backend(request, tmpdir):
    backend = backend_data(
        module=importlib.import_module('dsch.backends.' + request.param),
        storage_path=str(tmpdir.join('test_frontend.' + request.param))
    )
    return backend


class TestStorage:
    @pytest.fixture
    def storage_obj(self, backend):
        schema_node = schema.Bool()
        return backend.module.Storage(storage_path=backend.storage_path,
                                      schema_node=schema_node)

    def test_complete(self, storage_obj):
        assert not storage_obj.complete
        storage_obj.data.value = True
        assert storage_obj.complete

    @pytest.mark.parametrize('schema_node', (
        schema.Bool(),
        schema.Compilation({'spam': schema.Bool(), 'eggs': schema.Bool()}),
        schema.List(schema.Bool()),
    ))
    def test_incomplete_data(self, backend, schema_node):
        storage_obj = backend.module.Storage(storage_path=backend.storage_path,
                                             schema_node=schema_node)
        storage_obj.save(force=True)
        del storage_obj

        storage_obj = backend.module.Storage(storage_path=backend.storage_path,
                                             schema_node=schema_node)
        assert storage_obj.schema_node

    def test_schema_hash(self, storage_obj):
        nominal_hash = ('45d0233242870dd39f632cb5dd78704b'
                        '901db11b9483de5bcc6489b1d3b76235')
        assert storage_obj.schema_hash() == nominal_hash

    def test_validate(self, storage_obj):
        storage_obj.data.value = True
        storage_obj.validate()

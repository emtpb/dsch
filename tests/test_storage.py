from collections import namedtuple
import importlib
import pytest
from dsch import schema, storage


backend_data = namedtuple('backend_data', ('module', 'storage_path'))


@pytest.fixture(params=('hdf5', 'mat', 'npz'))
def backend(request, tmpdir):
    backend = backend_data(
        module=importlib.import_module('dsch.backends.' + request.param),
        storage_path=str(tmpdir.join('test_frontend.' + request.param))
    )
    return backend


class TestStorage:
    def test_complete(self, backend):
        schema_node = schema.Bool()
        storage_obj = backend.module.Storage(storage_path=backend.storage_path,
                                             schema_node=schema_node)
        assert not storage_obj.complete
        storage_obj.data.replace(True)
        assert storage_obj.complete

    @pytest.mark.parametrize('schema_node', (
        schema.Bool(),
        schema.Compilation({'spam': schema.Bool(), 'eggs': schema.Bool()}),
        schema.List(schema.Bool()),
    ))
    def test_incomplete_data(self, backend, schema_node):
        storage_obj = backend.module.Storage(storage_path=backend.storage_path,
                                             schema_node=schema_node)
        storage_obj.save()

    def test_schema_hash(self):
        schema_node = schema.Bool()
        storage_obj = storage.Storage('', schema_node)
        nominal_hash = ('45d0233242870dd39f632cb5dd78704b'
                        '901db11b9483de5bcc6489b1d3b76235')
        assert storage_obj.schema_hash() == nominal_hash

    def test_schema_from_json(self):
        storage_obj = storage.Storage('')
        storage_obj._schema_from_json('{"config": {}, "node_type": "Bool"}')
        assert isinstance(storage_obj.schema_node, schema.Bool)

    def test_schema_to_json(self):
        storage_obj = storage.Storage('', schema.Bool())
        json_data = storage_obj._schema_to_json()
        assert json_data == '{"config": {}, "node_type": "Bool"}'

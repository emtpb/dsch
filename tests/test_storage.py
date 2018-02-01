import importlib
from collections import namedtuple

import pytest

from dsch import schema

backend_data = namedtuple('backend_data', ('module', 'storage_path'))


class TestStorage:
    @pytest.fixture(params=('hdf5', 'inmem', 'mat', 'npz'))
    def storage_obj(self, request, tmpdir):
        if request.param == 'inmem':
            storage_path = '::inmem::'
        else:
            storage_path = str(tmpdir.join('test_frontend.' + request.param))
        backend =  backend_data(
            module=importlib.import_module('dsch.backends.' + request.param),
            storage_path=storage_path
        )
        schema_node = schema.Bool()
        return backend.module.Storage(storage_path=backend.storage_path,
                                      schema_node=schema_node)

    def test_complete(self, storage_obj):
        assert not storage_obj.complete
        storage_obj.data.value = True
        assert storage_obj.complete

    def test_schema_hash(self, storage_obj):
        nominal_hash = ('45d0233242870dd39f632cb5dd78704b'
                        '901db11b9483de5bcc6489b1d3b76235')
        assert storage_obj.schema_hash() == nominal_hash

    def test_validate(self, storage_obj):
        storage_obj.data.value = True
        storage_obj.validate()


class TestFileStorage:
    @pytest.fixture(params=('hdf5', 'mat', 'npz'))
    def backend(self, request, tmpdir):
        backend = backend_data(
            module=importlib.import_module('dsch.backends.' + request.param),
            storage_path=str(tmpdir.join('test_frontend.' + request.param))
        )
        return backend

    def test_file_exists(self, backend):
        schema_node = schema.Bool()
        storage_obj = backend.module.Storage(storage_path=backend.storage_path,
                                            schema_node=schema_node)
        storage_obj.save()
        del storage_obj

        with pytest.raises(FileExistsError):
            # Give schema_node --> request file creation
            storage_obj = backend.module.Storage(
                storage_path=backend.storage_path, schema_node=schema_node)

    def test_file_not_found(self, backend):
        # Omit schema node --> request file loading
        with pytest.raises(FileNotFoundError):
            storage_obj = backend.module.Storage(
                storage_path=backend.storage_path)

    @pytest.mark.parametrize('schema_node', (
        schema.Bool(),
        schema.Compilation({'spam': schema.Bool(), 'eggs': schema.Bool()}),
        schema.List(schema.Bool()),
    ))
    def test_incomplete_data(self, backend, schema_node):
        storage_obj = backend.module.Storage(storage_path=backend.storage_path,
                                             schema_node=schema_node)
        storage_obj.save()
        del storage_obj

        storage_obj = backend.module.Storage(storage_path=backend.storage_path)
        assert storage_obj.schema_node

    def test_missing_optional_data(self, backend):
        schema_node = schema.Compilation({
            'ham': schema.Compilation({'spam': schema.Bool(),
                                       'eggs': schema.Bool()},)
        })
        storage_obj = backend.module.Storage(storage_path=backend.storage_path,
                                             schema_node=schema_node)
        storage_obj.data.ham.spam.value = True
        storage_obj.save()
        del storage_obj

        storage_obj = backend.module.Storage(storage_path=backend.storage_path)
        assert hasattr(storage_obj.data.ham, 'eggs')

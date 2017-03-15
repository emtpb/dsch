import json
import numpy as np
from dsch import schema
from dsch.backends import npz


class TestBool:
    def test_init_from_storage(self):
        data_node = npz.Bool(schema.Bool(), data_storage=np.array([True]))
        assert data_node._storage == np.array([True])
        assert data_node.value is True

    def test_replace(self):
        data_node = npz.Bool(schema.Bool())
        data_node.replace(False)
        assert isinstance(data_node._storage, np.ndarray)
        assert data_node._storage == np.array([False])

    def test_save(self):
        data_node = npz.Bool(schema.Bool())
        data_node.replace(False)
        assert data_node.save() == np.array([False])


class TestCompilation:
    def test_init_from_storage(self):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        data_storage = {'spam': np.array([True]),
                        'eggs': np.array([False])}
        data_node = npz.Compilation(schema_node, data_storage=data_storage)
        assert data_node.spam.value is True
        assert data_node.eggs.value is False

    def test_save(self):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        data_node = npz.Compilation(schema_node)
        data_node.spam.replace(True)
        data_node.eggs.replace(False)
        data_storage = data_node.save()
        assert 'spam' in data_storage
        assert 'eggs' in data_storage
        assert data_storage['spam'] == np.array([True])
        assert data_storage['eggs'] == np.array([False])


class TestStorage:
    def test_load_compilation(self, tmpdir):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        storage_path = str(tmpdir.join('test_load_compilation.npz'))
        test_data = {'spam': np.array([True]),
                     'eggs': np.array([False]),
                     '_schema': schema_data}
        np.savez(storage_path, **test_data)

        npz_file = npz.Storage(storage_path=storage_path)
        assert hasattr(npz_file, 'data')
        assert hasattr(npz_file.data, 'spam')
        assert hasattr(npz_file.data, 'eggs')
        assert isinstance(npz_file.data.spam, npz.Bool)
        assert isinstance(npz_file.data.eggs, npz.Bool)
        assert npz_file.data.spam.value is True
        assert npz_file.data.eggs.value is False

    def test_load_item(self, tmpdir):
        schema_node = schema.Bool()
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        storage_path = str(tmpdir.join('test_load_item.npz'))
        test_data = {'data': True, '_schema': schema_data}
        np.savez(storage_path, **test_data)

        npz_file = npz.Storage(storage_path=storage_path)
        assert hasattr(npz_file, 'data')
        assert isinstance(npz_file.data, npz.Bool)
        assert npz_file.data.value is True

    def test_load_list(self, tmpdir):
        schema_node = schema.List(schema.Bool())
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        storage_path = str(tmpdir.join('test_load_list.npz'))
        test_data = {'data.item_0': True, 'data.item_1': False,
                     '_schema': schema_data}
        np.savez(storage_path, **test_data)

        npz_file = npz.Storage(storage_path=storage_path)
        assert hasattr(npz_file, 'data')
        assert isinstance(npz_file.data, npz.List)
        assert npz_file.data[0].value is True
        assert npz_file.data[1].value is False

    def test_save_compilation(self, tmpdir):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        storage_path = str(tmpdir.join('test_save_compilation.npz'))
        npz_file = npz.Storage(storage_path=storage_path,
                               schema_node=schema_node)
        npz_file.data.spam.replace(True)
        npz_file.data.eggs.replace(False)
        npz_file.save()

        with np.load(storage_path) as file_:
            assert '_schema' in file_
            assert file_['_schema'][()] == json.dumps(schema_node.to_dict(),
                                                      sort_keys=True)
            assert 'spam' in file_
            assert file_['spam'].dtype == 'bool'
            assert file_['spam'][0]
            assert 'eggs' in file_
            assert file_['eggs'].dtype == 'bool'
            assert not file_['eggs'][0]

    def test_save_item(self, tmpdir):
        schema_node = schema.Bool()
        storage_path = str(tmpdir.join('test_save_item.npz'))
        npz_file = npz.Storage(storage_path=storage_path,
                               schema_node=schema_node)
        npz_file.data.replace(True)
        npz_file.save()

        with np.load(storage_path) as file_:
            assert '_schema' in file_
            assert file_['_schema'][()] == json.dumps(schema_node.to_dict(),
                                                      sort_keys=True)
            assert 'data' in file_
            assert file_['data'].dtype == 'bool'
            assert file_['data'][0]

    def test_save_list(self, tmpdir):
        schema_node = schema.List(schema.Bool())
        storage_path = str(tmpdir.join('test_save_list.npz'))
        npz_file = npz.Storage(storage_path=storage_path,
                               schema_node=schema_node)
        npz_file.data.replace([True, False])
        npz_file.save()

        with np.load(storage_path) as file_:
            assert '_schema' in file_
            assert file_['_schema'][()] == json.dumps(schema_node.to_dict(),
                                                      sort_keys=True)
            assert 'data.item_0' in file_
            assert file_['data.item_0'].dtype == 'bool'
            assert file_['data.item_0'][0]
            assert 'data.item_1' in file_
            assert file_['data.item_1'].dtype == 'bool'
            assert not file_['data.item_1'][0]


class TestList:
    def test_init_from_storage(self):
        data_storage = {'item_0': np.array([True]),
                        'item_1': np.array([False])}
        data_node = npz.List(schema.List(schema.Bool()),
                             data_storage=data_storage)
        assert data_node[0].value is True
        assert data_node[1].value is False

    def test_save(self):
        data_node = npz.List(schema.List(schema.Bool()))
        data_node.append(True)
        data_node.append(False)
        data_storage = data_node.save()
        assert len(data_storage) == 2
        assert data_storage['item_0'] == np.array([True])
        assert data_storage['item_1'] == np.array([False])


class TestString:
    def test_init_from_storage(self):
        data_storage = np.array('spam', dtype='U')
        data_node = npz.String(schema.String(),
                               data_storage=data_storage)
        assert data_node._storage == data_storage
        assert data_node.value == 'spam'

    def test_replace(self):
        data_node = npz.String(schema.String())
        data_node.replace('spam')
        assert isinstance(data_node._storage, np.ndarray)
        assert data_node._storage == np.array('spam', dtype='U')

    def test_save(self):
        data_node = npz.String(schema.String())
        data_node.replace('spam')
        assert data_node.save() == np.array('spam', dtype='U')

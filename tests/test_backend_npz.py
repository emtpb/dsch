import json
import numpy as np
from dsch import schema
from dsch.backends import npz


class TestBool:
    def test_load(self):
        schema_node = schema.Bool()
        data_node = npz.Bool(schema_node)
        data_node.load(np.array([True]))
        assert data_node.storage == np.array([True])
        assert data_node.value is True

    def test_replace(self):
        schema_node = schema.Bool()
        data_node = npz.Bool(schema_node)
        data_node.replace(False)
        assert isinstance(data_node.storage, np.ndarray)
        assert data_node.storage == np.array([False])


class TestCompilation:
    def test_load(self):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        data_node = npz.Compilation(schema_node)
        data_storage = {'spam': np.array([True]),
                        'eggs': np.array([False])}
        data_node.load(data_storage)
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


class TestNPZFile:
    def test_load_compilation(self, tmpdir):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        file_name = str(tmpdir.join('test_load_compilation.npz'))
        test_data = {'spam': np.array([True]),
                     'eggs': np.array([False]),
                     '_schema': schema_data}
        np.savez(file_name, **test_data)

        npz_file = npz.NPZFile(file_name=file_name)
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
        file_name = str(tmpdir.join('test_load_item.npz'))
        test_data = {'data': True, '_schema': schema_data}
        np.savez(file_name, **test_data)

        npz_file = npz.NPZFile(file_name=file_name)
        assert hasattr(npz_file, 'data')
        assert isinstance(npz_file.data, npz.Bool)
        assert npz_file.data.value is True

    def test_load_list(self, tmpdir):
        schema_node = schema.List(schema.Bool())
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        file_name = str(tmpdir.join('test_load_list.npz'))
        test_data = {'data.item_0': True, 'data.item_1': False,
                     '_schema': schema_data}
        np.savez(file_name, **test_data)

        npz_file = npz.NPZFile(file_name=file_name)
        assert hasattr(npz_file, 'data')
        assert isinstance(npz_file.data, npz.List)
        assert npz_file.data[0].value is True
        assert npz_file.data[1].value is False

    def test_save_compilation(self, tmpdir):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        npz_file = npz.NPZFile(schema_node=schema_node)
        data_storage = {'spam': {'data': np.array([True])},
                        'eggs': {'data': np.array([False])}}
        npz_file.data.load(data_storage)
        file_name = str(tmpdir.join('test_save_compilation.npz'))
        npz_file.save(file_name)

        with np.load(file_name) as file_:
            assert '_schema' in file_
            assert file_['_schema'][()] == json.dumps(schema_node.to_dict(),
                                                      sort_keys=True)
            assert 'spam.data' in file_
            assert file_['spam.data'].dtype == 'bool'
            assert file_['spam.data'][0]
            assert 'eggs.data' in file_
            assert file_['eggs.data'].dtype == 'bool'
            assert not file_['eggs.data'][0]

    def test_save_item(self, tmpdir):
        schema_node = schema.Bool()
        npz_file = npz.NPZFile(schema_node=schema_node)
        npz_file.data.replace(True)
        file_name = str(tmpdir.join('test_save_item.npz'))
        npz_file.save(file_name)

        with np.load(file_name) as file_:
            assert '_schema' in file_
            assert file_['_schema'][()] == json.dumps(schema_node.to_dict(),
                                                      sort_keys=True)
            assert 'data' in file_
            assert file_['data'].dtype == 'bool'
            assert file_['data'][0]

    def test_save_list(self, tmpdir):
        schema_node = schema.List(schema.Bool())
        npz_file = npz.NPZFile(schema_node=schema_node)
        npz_file.data.replace([True, False])
        file_name = str(tmpdir.join('test_save_list.npz'))
        npz_file.save(file_name)

        with np.load(file_name) as file_:
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
    def test_load(self):
        schema_node = schema.List(schema.Bool())
        data_node = npz.List(schema_node)
        data_storage = {'item_0': np.array([True]),
                        'item_1': np.array([False])}
        data_node.load(data_storage)
        assert data_node[0].value is True
        assert data_node[1].value is False

    def test_save(self):
        schema_node = schema.List(schema.Bool())
        data_node = npz.List(schema_node)
        data_node.append(True)
        data_node.append(False)
        data_storage = data_node.save()
        assert len(data_storage) == 2
        assert data_storage['item_0'] == np.array([True])
        assert data_storage['item_1'] == np.array([False])

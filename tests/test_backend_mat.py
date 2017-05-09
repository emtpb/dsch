import json
import numpy as np
import scipy.io as sio
from dsch import schema
from dsch.backends import mat


class TestCompilation:
    def test_init_from_storage(self):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        data_storage = np.array((np.array([True]), np.array([False])),
                                dtype=[('spam', 'O'), ('eggs', 'O')])
        data_node = mat.Compilation(schema_node, parent=None,
                                    data_storage=data_storage)
        assert data_node.spam.value is True
        assert data_node.eggs.value is False


class TestList:
    def test_init_from_storage(self):
        data_storage = np.zeros((2,), dtype=np.object)
        data_storage[0] = np.array([True])
        data_storage[1] = np.array([False])
        data_node = mat.List(schema.List(schema.Bool()), parent=None,
                             data_storage=data_storage)
        assert data_node[0].value is True
        assert data_node[1].value is False

    def test_save(self):
        data_node = mat.List(schema.List(schema.Bool()), parent=None)
        data_node.append(True)
        data_node.append(False)
        data_storage = data_node.save()
        assert len(data_storage) == 2
        assert data_storage[0] == np.array([True])
        assert data_storage[1] == np.array([False])


class TestStorage:
    def test_complete(self, tmpdir):
        schema_node = schema.Bool()
        storage_path = str(tmpdir.join('test_complete.mat'))
        mat_file = mat.Storage(storage_path=storage_path,
                               schema_node=schema_node)
        assert not mat_file.complete
        mat_file.data.replace(True)
        assert mat_file.complete

    def test_load_compilation(self, tmpdir):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        storage_path = str(tmpdir.join('test_load_compilation.mat'))
        test_data = {'data': {'spam': np.array([True]),
                              'eggs': np.array([False])},
                     'schema': schema_data}
        sio.savemat(storage_path, test_data)

        mat_file = mat.Storage(storage_path=storage_path)
        assert hasattr(mat_file, 'data')
        assert hasattr(mat_file.data, 'spam')
        assert hasattr(mat_file.data, 'eggs')
        assert isinstance(mat_file.data.spam, mat.Bool)
        assert isinstance(mat_file.data.eggs, mat.Bool)
        assert mat_file.data.spam.value is True
        assert mat_file.data.eggs.value is False

    def test_load_item(self, tmpdir):
        schema_node = schema.Bool()
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        storage_path = str(tmpdir.join('test_load_item.mat'))
        test_data = {'data': True, 'schema': schema_data}
        sio.savemat(storage_path, test_data)

        mat_file = mat.Storage(storage_path=storage_path)
        assert hasattr(mat_file, 'data')
        assert isinstance(mat_file.data, mat.Bool)
        assert mat_file.data.value is True

    def test_load_list(self, tmpdir):
        schema_node = schema.List(schema.Bool())
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        storage_path = str(tmpdir.join('test_load_list.mat'))
        data = np.zeros((2,), dtype=np.object)
        data[0] = np.array([True])
        data[1] = np.array([False])
        # test_data = {'data': {'item_0': True, 'item_1': False},
        test_data = {'data': data, 'schema': schema_data}
        sio.savemat(storage_path, test_data)

        mat_file = mat.Storage(storage_path=storage_path)
        assert hasattr(mat_file, 'data')
        assert isinstance(mat_file.data, mat.List)
        assert mat_file.data[0].value is True
        assert mat_file.data[1].value is False

    def test_save_compilation(self, tmpdir):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        storage_path = str(tmpdir.join('test_save_compilation.mat'))
        mat_file = mat.Storage(storage_path=storage_path,
                               schema_node=schema_node)
        mat_file.data.spam.replace(True)
        mat_file.data.eggs.replace(False)
        mat_file.save()

        file_ = sio.loadmat(storage_path, squeeze_me=True)
        assert 'schema' in file_
        assert file_['schema'] == json.dumps(schema_node.to_dict(),
                                             sort_keys=True)
        assert 'data' in file_
        assert 'spam' in file_['data'].dtype.fields
        assert file_['data']['spam']
        assert 'eggs' in file_['data'].dtype.fields
        assert not file_['data']['eggs']

    def test_save_item(self, tmpdir):
        schema_node = schema.Bool()
        storage_path = str(tmpdir.join('test_save_item.mat'))
        mat_file = mat.Storage(storage_path=storage_path,
                               schema_node=schema_node)
        mat_file.data.replace(True)
        mat_file.save()

        file_ = sio.loadmat(storage_path, squeeze_me=True)
        assert 'schema' in file_
        assert file_['schema'] == json.dumps(schema_node.to_dict(),
                                             sort_keys=True)
        assert 'data' in file_
        assert file_['data']

    def test_save_list(self, tmpdir):
        schema_node = schema.List(schema.Bool())
        storage_path = str(tmpdir.join('test_save_list.mat'))
        mat_file = mat.Storage(storage_path=storage_path,
                               schema_node=schema_node)
        mat_file.data.replace([True, False])
        mat_file.save()

        file_ = sio.loadmat(storage_path, squeeze_me=True)
        assert 'schema' in file_
        assert file_['schema'] == json.dumps(schema_node.to_dict(),
                                             sort_keys=True)
        assert 'data' in file_
        assert len(file_['data']) == 2
        assert file_['data'][0]
        assert not file_['data'][1]

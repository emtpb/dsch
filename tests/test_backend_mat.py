import datetime
import json
import numpy as np
import scipy.io as sio
from dsch import schema
from dsch.backends import mat


class TestArray:
    def test_init_from_storage(self):
        data_node = mat.Array(schema.Array(dtype='int'), parent=None,
                              data_storage=np.array([23, 42]))
        assert np.all(data_node._storage == np.array([23, 42]))
        assert np.all(data_node.value == np.array([23, 42]))

    def test_replace(self):
        data_node = mat.Array(schema.Array(dtype='int'), parent=None)
        data_node.replace(np.array([23, 42]))
        assert isinstance(data_node._storage, np.ndarray)
        assert np.all(data_node._storage == np.array([23, 42]))

    def test_save(self):
        data_node = mat.Array(schema.Array(dtype='int'), parent=None)
        data_node.replace(np.array([23, 42]))
        assert np.all(data_node.save() == np.array([23, 42]))


class TestBool:
    def test_init_from_storage(self):
        data_node = mat.Bool(schema.Bool(), parent=None,
                             data_storage=np.array([True]))
        assert data_node._storage == np.array([True])
        assert data_node.value is True

    def test_replace(self):
        data_node = mat.Bool(schema.Bool(), parent=None)
        data_node.replace(False)
        assert isinstance(data_node._storage, np.ndarray)
        assert data_node._storage == np.array([False])

    def test_save(self):
        data_node = mat.Bool(schema.Bool(), parent=None)
        data_node.replace(False)
        assert data_node.save() == np.array([False])


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

    def test_save(self):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        data_node = mat.Compilation(schema_node, parent=None)
        data_node.spam.replace(True)
        data_node.eggs.replace(False)
        data_storage = data_node.save()
        assert 'spam' in data_storage
        assert 'eggs' in data_storage
        assert data_storage['spam'] == np.array([True])
        assert data_storage['eggs'] == np.array([False])


class TestDate:
    def test_init_from_storage(self):
        dt = datetime.date.today()
        data_storage = np.array([dt.year, dt.month, dt.day])
        data_node = mat.Date(schema.Date(), parent=None,
                             data_storage=data_storage)
        assert np.all(data_node._storage == data_storage)
        assert data_node.value == dt

    def test_replace(self):
        data_node = mat.Date(schema.Date(), parent=None)
        dt = datetime.date.today()
        data_node.replace(dt)
        assert isinstance(data_node._storage, np.ndarray)
        assert np.all(data_node._storage == np.array(
            [dt.year, dt.month, dt.day]))

    def test_save(self):
        data_node = mat.Date(schema.Date(), parent=None)
        dt = datetime.date.today()
        data_node.replace(dt)
        assert np.all(data_node.save() == np.array(
            [dt.year, dt.month, dt.day]))


class TestDateTime:
    def test_init_from_storage(self):
        dt = datetime.datetime.now()
        data_storage = np.array([dt.year, dt.month, dt.day, dt.hour,
                                 dt.minute, dt.second, dt.microsecond])
        data_node = mat.DateTime(schema.DateTime(), parent=None,
                                 data_storage=data_storage)
        assert np.all(data_node._storage == data_storage)
        assert data_node.value == dt

    def test_replace(self):
        data_node = mat.DateTime(schema.DateTime(), parent=None)
        dt = datetime.datetime.now()
        data_node.replace(dt)
        assert isinstance(data_node._storage, np.ndarray)
        assert np.all(data_node._storage == np.array(
            [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
             dt.microsecond]))

    def test_save(self):
        data_node = mat.DateTime(schema.DateTime(), parent=None)
        dt = datetime.datetime.now()
        data_node.replace(dt)
        assert np.all(data_node.save() == np.array(
            [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
             dt.microsecond]))


class TestScalar:
    def test_init_from_storage(self):
        data_node = mat.Scalar(schema.Scalar(dtype='int32'), parent=None,
                               data_storage=np.int32(42))
        assert data_node._storage == np.int32(42)
        assert data_node.value == 42

    def test_replace(self):
        data_node = mat.Scalar(schema.Scalar(dtype='int32'), parent=None)
        data_node.replace(42)
        assert isinstance(data_node._storage, np.int32)
        assert data_node._storage == 42

    def test_save(self):
        data_node = mat.Scalar(schema.Scalar(dtype='int32'), parent=None)
        data_node.replace(42)
        assert data_node.save() == np.int32(42)


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


class TestString:
    def test_init_from_storage(self):
        data_storage = np.array('spam', dtype='U')
        data_node = mat.String(schema.String(), parent=None,
                               data_storage=data_storage)
        assert data_node._storage == data_storage
        assert data_node.value == 'spam'

    def test_replace(self):
        data_node = mat.String(schema.String(), parent=None)
        data_node.replace('spam')
        assert isinstance(data_node._storage, np.ndarray)
        assert data_node._storage == np.array('spam', dtype='U')

    def test_save(self):
        data_node = mat.String(schema.String(), parent=None)
        data_node.replace('spam')
        assert data_node.save() == np.array('spam', dtype='U')


class TestTime:
    def test_init_from_storage(self):
        dt = datetime.time(13, 37, 42, 23)
        data_storage = np.array([dt.hour, dt.minute, dt.second,
                                 dt.microsecond])
        data_node = mat.Time(schema.Time(), parent=None,
                             data_storage=data_storage)
        assert np.all(data_node._storage == data_storage)
        assert data_node.value == dt

    def test_replace(self):
        data_node = mat.Time(schema.Time(), parent=None)
        dt = datetime.time(13, 37, 42, 23)
        data_node.replace(dt)
        assert isinstance(data_node._storage, np.ndarray)
        assert np.all(data_node._storage == np.array(
            [dt.hour, dt.minute, dt.second, dt.microsecond]))

    def test_save(self):
        data_node = mat.Time(schema.Time(), parent=None)
        dt = datetime.time(13, 37, 42, 23)
        data_node.replace(dt)
        assert np.all(data_node.save() == np.array(
            [dt.hour, dt.minute, dt.second, dt.microsecond]))

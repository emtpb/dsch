import h5py
import numpy as np
import json
import pytest
from dsch import schema
from dsch.backends import hdf5


@pytest.fixture()
def hdf5file(tmpdir):
    return h5py.File(str(tmpdir.join('hdf5test.h5')))


class TestArray:
    def test_init_from_storage(self, hdf5file):
        hdf5file.create_dataset('test_array', data=np.array([23, 42]))

        schema_node = schema.Array(dtype='int')
        data_node = hdf5.Array(schema_node,
                               data_storage=hdf5file['test_array'])
        assert np.all(data_node._storage == hdf5file['test_array'])
        assert np.all(data_node.value == np.array([23, 42]))

    def test_init_new(self, hdf5file):
        schema_node = schema.Array(dtype='int')
        data_node = hdf5.Array(schema_node, new_params={'name': 'test_array',
                                                        'parent': hdf5file})
        assert 'test_array' in hdf5file
        assert isinstance(hdf5file['test_array'], h5py.Dataset)
        assert hdf5file['test_array'].dtype == 'int'
        assert data_node._storage == hdf5file['test_array']

    def test_replace(self, hdf5file):
        hdf5file.create_dataset('test_array', data=np.array([23, 42]))

        schema_node = schema.Array(dtype='int')
        data_node = hdf5.Array(schema_node,
                               data_storage=hdf5file['test_array'])
        data_node.replace(np.array([23, 42]))
        assert isinstance(data_node._storage, h5py.Dataset)
        assert np.all(data_node._storage == hdf5file['test_array'])
        assert np.all(data_node.value == np.array([23, 42]))


class TestBool:
    def test_init_from_storage(self, hdf5file):
        hdf5file.create_dataset('test_bool', data=True)

        schema_node = schema.Bool()
        data_node = hdf5.Bool(schema_node, data_storage=hdf5file['test_bool'])
        assert data_node._storage == hdf5file['test_bool']
        assert data_node.value is True

    def test_init_new(self, hdf5file):
        schema_node = schema.Bool()
        data_node = hdf5.Bool(schema_node, new_params={'name': 'test_bool',
                                                       'parent': hdf5file})
        assert 'test_bool' in hdf5file
        assert isinstance(hdf5file['test_bool'], h5py.Dataset)
        assert hdf5file['test_bool'].dtype == 'bool'
        assert data_node._storage == hdf5file['test_bool']

    def test_replace(self, hdf5file):
        hdf5file.create_dataset('test_bool', data=True)

        schema_node = schema.Bool()
        data_node = hdf5.Bool(schema_node, data_storage=hdf5file['test_bool'])
        data_node.replace(False)
        assert isinstance(data_node._storage, h5py.Dataset)
        assert data_node._storage == hdf5file['test_bool']
        assert data_node.value is False


class TestCompilation:
    def test_init_from_storage(self, hdf5file):
        test_comp = hdf5file.create_group('test_comp')
        test_comp.create_dataset('spam', data=True)
        test_comp.create_dataset('eggs', data=False)

        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        data_node = hdf5.Compilation(schema_node, data_storage=test_comp)
        assert data_node.spam.value is True
        assert data_node.eggs.value is False

    def test_init_new(self, hdf5file):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        hdf5.Compilation(schema_node,
                         new_params={'name': 'test_comp', 'parent': hdf5file})
        assert 'test_comp' in hdf5file
        assert isinstance(hdf5file['test_comp'], h5py.Group)
        assert 'spam' in hdf5file['test_comp']
        assert 'eggs' in hdf5file['test_comp']


class TestStorage:
    def test_load_compilation(self, tmpdir):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        file_name = str(tmpdir.join('test_load_compilation.hdf5'))
        raw_file = h5py.File(file_name)
        raw_file.attrs['dsch_schema'] = schema_data
        raw_file.create_dataset('spam', data=True)
        raw_file.create_dataset('eggs', data=False)
        raw_file.flush()
        del raw_file

        hdf5_file = hdf5.Storage(storage_path=file_name)
        assert hasattr(hdf5_file, 'data')
        assert hasattr(hdf5_file.data, 'spam')
        assert hasattr(hdf5_file.data, 'eggs')
        assert isinstance(hdf5_file.data.spam, hdf5.Bool)
        assert isinstance(hdf5_file.data.eggs, hdf5.Bool)
        assert hdf5_file.data.spam.value is True
        assert hdf5_file.data.eggs.value is False

    def test_load_item(self, tmpdir):
        schema_node = schema.Bool()
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        file_name = str(tmpdir.join('test_load_item.hdf5'))
        raw_file = h5py.File(file_name)
        raw_file.attrs['dsch_schema'] = schema_data
        raw_file.create_dataset('dsch_data', data=True)
        raw_file.flush()
        del raw_file

        hdf5_file = hdf5.Storage(storage_path=file_name)
        assert hasattr(hdf5_file, 'data')
        assert isinstance(hdf5_file.data, hdf5.Bool)
        assert hdf5_file.data.value is True

    def test_load_list(self, tmpdir):
        schema_node = schema.List(schema.Bool())
        schema_data = json.dumps(schema_node.to_dict(), sort_keys=True)
        file_name = str(tmpdir.join('test_load_list.hdf5'))
        raw_file = h5py.File(file_name)
        raw_file.attrs['dsch_schema'] = schema_data
        data = raw_file.create_group('dsch_data')
        data.create_dataset('item_0', data=True)
        data.create_dataset('item_1', data=False)
        raw_file.flush()
        del raw_file

        hdf5_file = hdf5.Storage(storage_path=file_name)
        assert hasattr(hdf5_file, 'data')
        assert isinstance(hdf5_file.data, hdf5.List)
        assert hdf5_file.data[0].value is True
        assert hdf5_file.data[1].value is False

    def test_save_compilation(self, tmpdir):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        file_name = str(tmpdir.join('test_save_compilation.hdf5'))
        hdf5_file = hdf5.Storage(storage_path=file_name,
                                 schema_node=schema_node)
        hdf5_file.data.spam.replace(True)
        hdf5_file.data.eggs.replace(False)
        hdf5_file.save()

        file_ = h5py.File(file_name, 'r')
        assert 'dsch_schema' in file_.attrs
        assert file_.attrs['dsch_schema'] == json.dumps(schema_node.to_dict(),
                                                        sort_keys=True)
        assert 'spam' in file_
        assert file_['spam'].dtype == 'bool'
        assert file_['spam'].value
        assert 'eggs' in file_
        assert file_['eggs'].dtype == 'bool'
        assert not file_['eggs'].value

    def test_save_item(self, tmpdir):
        schema_node = schema.Bool()
        file_name = str(tmpdir.join('test_save_item.h5'))
        hdf5_file = hdf5.Storage(storage_path=file_name,
                                 schema_node=schema_node)
        hdf5_file.data.replace(True)
        hdf5_file.save()
        del hdf5_file

        file_ = h5py.File(file_name, 'r')
        assert 'dsch_schema' in file_.attrs
        assert file_.attrs['dsch_schema'] == json.dumps(schema_node.to_dict(),
                                                        sort_keys=True)
        assert 'dsch_data' in file_
        assert file_['dsch_data'].dtype == 'bool'
        assert file_['dsch_data'].value

    def test_save_list(self, tmpdir):
        schema_node = schema.List(schema.Bool())
        file_name = str(tmpdir.join('test_save_list.h5'))
        hdf5_file = hdf5.Storage(storage_path=file_name,
                                 schema_node=schema_node)
        hdf5_file.data.replace([True, False])
        hdf5_file.save()

        file_ = h5py.File(file_name, 'r')
        assert 'dsch_schema' in file_.attrs
        assert file_.attrs['dsch_schema'] == json.dumps(schema_node.to_dict(),
                                                        sort_keys=True)
        assert 'dsch_data' in file_
        assert 'item_0' in file_['dsch_data']
        assert file_['dsch_data']['item_0'].dtype == 'bool'
        assert file_['dsch_data']['item_0'].value
        assert 'item_1' in file_['dsch_data']
        assert file_['dsch_data']['item_1'].dtype == 'bool'
        assert not file_['dsch_data']['item_1'].value


class TestList:
    def test_init_from_storage(self, hdf5file):
        test_list = hdf5file.create_group('test_list')
        test_list.create_dataset('item_0', data=True)
        test_list.create_dataset('item_1', data=False)

        schema_node = schema.List(schema.Bool())
        data_node = hdf5.List(schema_node, data_storage=test_list)
        assert data_node[0].value is True
        assert data_node[1].value is False

    def test_init_new(self, hdf5file):
        schema_node = schema.List(schema.Bool())
        hdf5.List(schema_node,
                  new_params={'name': 'test_list', 'parent': hdf5file})
        assert 'test_list' in hdf5file
        assert isinstance(hdf5file['test_list'], h5py.Group)


class TestString:
    def test_init_from_storage(self, hdf5file):
        hdf5file.create_dataset('test_string', data='spam')

        schema_node = schema.String()
        data_node = hdf5.String(schema_node,
                                data_storage=hdf5file['test_string'])
        assert data_node._storage == hdf5file['test_string']
        assert data_node.value == 'spam'

    def test_init_new(self, hdf5file):
        schema_node = schema.String()
        data_node = hdf5.Bool(schema_node, new_params={'name': 'test_string',
                                                       'parent': hdf5file})
        assert 'test_string' in hdf5file
        assert isinstance(hdf5file['test_string'], h5py.Dataset)
        assert data_node._storage == hdf5file['test_string']

    def test_replace(self, hdf5file):
        hdf5file.create_dataset('test_string', data='spam')

        schema_node = schema.String()
        data_node = hdf5.String(schema_node,
                                data_storage=hdf5file['test_string'])
        data_node.replace('eggs')
        assert isinstance(data_node._storage, h5py.Dataset)
        assert data_node._storage == hdf5file['test_string']
        assert data_node.value == 'eggs'

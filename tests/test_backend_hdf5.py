import datetime
import h5py
import numpy as np
import json
import pytest
from dsch import data, schema
from dsch.backends import hdf5


@pytest.fixture()
def hdf5file(tmpdir):
    return h5py.File(str(tmpdir.join('hdf5test.h5')))


@pytest.mark.parametrize('schema_node,valid_data', (
    (schema.Array(dtype='int'), np.array([23, 42])),
    (schema.Bool(), True),
    (schema.Date(), datetime.date.today()),
    (schema.DateTime(), datetime.datetime.now()),
    (schema.Scalar(dtype='int32'), np.int32(42)),
    (schema.String(), 'spam'),
    (schema.Time(), datetime.datetime.now().time()),
))
class TestItemNode:
    @pytest.fixture
    def data_node(self, schema_node, hdf5file):
        new_params = {'name': 'test_item', 'parent': hdf5file}
        return data.data_node_from_schema(schema_node,
                                          module_name='dsch.backends.hdf5',
                                          parent=None, new_params=new_params)

    def test_clear(self, data_node, valid_data, hdf5file):
        data_node.replace(valid_data)
        assert 'test_item' in hdf5file
        data_node.clear()
        assert 'test_item' not in hdf5file

    def test_init_new(self, data_node, valid_data, hdf5file):
        assert 'test_item' not in hdf5file
        assert data_node._dataset_name == 'test_item'
        assert data_node._parent == hdf5file


class TestCompilation:
    def test_init_from_storage(self, hdf5file):
        test_comp = hdf5file.create_group('test_comp')
        test_comp.create_dataset('spam', data=True)
        test_comp.create_dataset('eggs', data=False)

        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        data_node = hdf5.Compilation(schema_node, parent=None,
                                     data_storage=test_comp)
        assert data_node.spam.value is True
        assert data_node.eggs.value is False

    def test_init_new(self, hdf5file):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = hdf5.Compilation(schema_node, parent=None,
                                new_params={'name': 'test_comp',
                                            'parent': hdf5file})
        assert 'test_comp' in hdf5file
        assert isinstance(hdf5file['test_comp'], h5py.Group)
        assert 'spam' in comp._subnodes
        assert 'eggs' in comp._subnodes


class TestList:
    def test_init_from_storage(self, hdf5file):
        test_list = hdf5file.create_group('test_list')
        test_list.create_dataset('item_0', data=True)
        test_list.create_dataset('item_1', data=False)

        schema_node = schema.List(schema.Bool())
        data_node = hdf5.List(schema_node, parent=None,
                              data_storage=test_list)
        assert data_node[0].value is True
        assert data_node[1].value is False

    def test_init_new(self, hdf5file):
        schema_node = schema.List(schema.Bool())
        hdf5.List(schema_node, parent=None,
                  new_params={'name': 'test_list', 'parent': hdf5file})
        assert 'test_list' in hdf5file
        assert isinstance(hdf5file['test_list'], h5py.Group)


class TestStorage:
    def test_complete(self, tmpdir):
        schema_node = schema.Bool()
        file_name = str(tmpdir.join('test_complete.h5'))
        hdf5_file = hdf5.Storage(storage_path=file_name,
                                 schema_node=schema_node)
        assert not hdf5_file.complete
        hdf5_file.data.replace(True)
        assert hdf5_file.complete

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

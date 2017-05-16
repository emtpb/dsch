import datetime
import json
import numpy as np
import pytest
from dsch import data, schema
from dsch.backends import npz


@pytest.mark.parametrize('schema_node,valid_data', (
    (schema.Array(dtype='int'), np.array([23, 42])),
    (schema.Bool(), True),
    (schema.Date(), datetime.date.today()),
    (schema.DateTime(), datetime.datetime.now()),
    (schema.Scalar(dtype='int32'), np.int32(42)),
    (schema.String(), 'spam'),
    (schema.Time(), datetime.datetime.now().time()),
))
def test_save_item_node(schema_node, valid_data):
    data_node = data.data_node_from_schema(schema_node,
                                           module_name='dsch.backends.npz',
                                           parent=None)
    data_node.value = valid_data
    assert np.all(data_node.save() == data_node._storage)


class TestCompilation:
    def test_init_from_storage(self):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        data_storage = {'spam': np.array([True]),
                        'eggs': np.array([False])}
        data_node = npz.Compilation(schema_node, parent=None,
                                    data_storage=data_storage)
        assert data_node.spam.value is True
        assert data_node.eggs.value is False

    def test_save(self):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        data_node = npz.Compilation(schema_node, parent=None)
        data_node.spam.value = True
        data_node.eggs.value = False
        data_storage = data_node.save()
        assert 'spam' in data_storage
        assert 'eggs' in data_storage
        assert data_storage['spam'] == np.array([True])
        assert data_storage['eggs'] == np.array([False])


class TestList:
    def test_init_from_storage(self):
        data_storage = {'item_0': np.array([True]),
                        'item_1': np.array([False])}
        data_node = npz.List(schema.List(schema.Bool()), parent=None,
                             data_storage=data_storage)
        assert data_node[0].value is True
        assert data_node[1].value is False

    def test_save(self):
        data_node = npz.List(schema.List(schema.Bool()), parent=None)
        data_node.append(True)
        data_node.append(False)
        data_storage = data_node.save()
        assert len(data_storage) == 2
        assert data_storage['item_0'] == np.array([True])
        assert data_storage['item_1'] == np.array([False])


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
        npz_file.data.spam.value = True
        npz_file.data.eggs.value = False
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
        npz_file.data.value = True
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


def test_inflate_dotted():
    input = {
        'foo.foo.foo': 23,
        'foo.bar.foo': 42,
        'foo.bar.baz': 1337,
        'baz.bar.foo': 9000
    }
    output = npz._inflate_dotted(input)
    assert output == {
        'foo': {'foo': {'foo': 23}, 'bar': {'foo': 42, 'baz': 1337}},
        'baz': {'bar': {'foo': 9000}}
    }


def test_flatten_dotted():
    input = {
        'foo': {'foo': {'foo': 23}, 'bar': {'foo': 42, 'baz': 1337}},
        'baz': {'bar': {'foo': 9000}}
    }
    output = npz._flatten_dotted(input)
    assert output == {
        'foo.foo.foo': 23,
        'foo.bar.foo': 42,
        'foo.bar.baz': 1337,
        'baz.bar.foo': 9000
    }

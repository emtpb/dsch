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

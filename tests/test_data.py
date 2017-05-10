from collections import namedtuple
import datetime
import h5py
import importlib
import numpy as np
import pytest
from dsch import data, schema


backend_data = namedtuple('backend_data', ('module', 'new_params'))


@pytest.fixture(params=('hdf5', 'mat', 'npz'))
def backend(request, tmpdir):
    if request.param == 'hdf5':
        hdf5file = h5py.File(str(tmpdir.join('hdf5test.h5')))
        new_params = {'name': 'test_data', 'parent': hdf5file['/']}
    elif request.param in ('mat', 'npz'):
        new_params = None
    return backend_data(module=importlib.import_module('dsch.backends.' +
                                                       request.param),
                        new_params=new_params)


class ItemNodeTestBase:
    @pytest.fixture()
    def data_node(self, backend):
        data_node_class = getattr(backend.module, self.class_name)
        return data_node_class(self.schema_node, parent=None,
                               new_params=backend.new_params)

    def test_clear(self, data_node):
        data_node.replace(self.valid_data)
        data_node.clear()
        assert data_node._storage is None

    def test_complete(self, data_node):
        assert not data_node.complete
        data_node.replace(self.valid_data)
        assert data_node.complete

    def test_empty(self, data_node):
        assert data_node.empty
        data_node.replace(self.valid_data)
        assert not data_node.empty

    def test_init_new(self, data_node):
        assert data_node.schema_node == self.schema_node
        assert data_node._storage is None

    def test_replace(self, data_node):
        data_node.replace(self.valid_data)
        assert np.all(data_node.value == self.valid_data)

    def test_roundtrip(self, backend):
        data_node_class = getattr(backend.module, self.class_name)
        data_node1 = data_node_class(self.schema_node, parent=None,
                                     new_params=backend.new_params)
        data_node1.replace(self.valid_data)
        storage = data_node1._storage
        data_node2 = data_node_class(self.schema_node, parent=None,
                                     data_storage=storage)
        assert np.all(data_node2.value == self.valid_data)

    def test_validate(self, data_node):
        data_node.replace(self.valid_data)
        data_node.validate()

    def test_value(self, data_node):
        data_node.replace(self.valid_data)
        assert isinstance(data_node.value, type(self.valid_data))


class TestArray(ItemNodeTestBase):
    class_name = 'Array'
    schema_node = schema.Array(dtype='int')
    valid_data = np.array([23, 42])

    def test_getitem(self, data_node):
        data_node.replace(self.valid_data)
        for idx, item in enumerate(self.valid_data):
            assert data_node[idx] == item
        assert np.all(data_node[()] == self.valid_data)

    def test_resize(self, data_node):
        data_node.replace(np.array([42]))
        assert data_node.value.shape == (1,)
        data_node.resize((5,))
        assert data_node.value.ndim == 1
        assert data_node.value.shape == (5,)

    def test_setitem(self, data_node):
        data_node.replace(np.array([5, 23, 42]))
        data_node[0] = 1
        assert np.all(data_node.value == np.array([1, 23, 42]))
        data_node[1:] = np.array([2, 3])
        assert np.all(data_node.value == np.array([1, 2, 3]))
        data_node[()] = np.array([42, 23, 5])
        assert np.all(data_node.value == np.array([42, 23, 5]))

    def test_validate_depends_on(self, backend):
        comp = backend.module.Compilation(schema.Compilation({
            'time': schema.Array(dtype='float'),
            'voltage': schema.Array(dtype='float', depends_on='time'),
        }), parent=None, new_params=backend.new_params)
        comp.time.replace(np.array([0, 1, 2], dtype='float'))
        comp.voltage.replace(np.array([2, 3, 5], dtype='float'))
        comp.voltage.validate()


class TestBool(ItemNodeTestBase):
    class_name = 'Bool'
    schema_node = schema.Bool()
    valid_data = True


@pytest.mark.parametrize('schema_subnode,valid_subnode_data', (
    (schema.Array(dtype='int32'), np.array([23, 42])),
    (schema.Bool(), True),
    (schema.Date(), datetime.date.today()),
    (schema.DateTime(), datetime.datetime.now()),
    (schema.Scalar(dtype='int32'), np.int32(42)),
    (schema.String(), 'spam'),
    (schema.Time(), datetime.time(13, 37, 42, 23)),
))
class TestCompilation:
    @pytest.fixture()
    def data_node(self, backend, schema_subnode):
        schema_node = schema.Compilation({'spam': schema_subnode,
                                          'eggs': schema_subnode})
        data_node = backend.module.Compilation(schema_node, parent=None,
                                               new_params=backend.new_params)
        return data_node

    def test_clear(self, data_node, valid_subnode_data):
        data_node.spam.replace(valid_subnode_data)
        data_node.eggs.replace(valid_subnode_data)
        data_node.clear()
        assert data_node.spam._storage is None
        assert data_node.eggs._storage is None

    def test_complete(self, data_node, valid_subnode_data):
        assert not data_node.complete
        data_node.spam.replace(valid_subnode_data)
        assert not data_node.complete
        data_node.eggs.replace(valid_subnode_data)
        assert data_node.complete

    def test_complete_optionals(self, data_node, valid_subnode_data):
        data_node.schema_node.optionals.append('eggs')
        assert not data_node.complete
        data_node.spam.replace(valid_subnode_data)
        assert data_node.complete

    def test_empty(self, data_node, valid_subnode_data):
        assert data_node.empty
        data_node.spam.replace(valid_subnode_data)
        assert not data_node.empty

    def test_dir(self, data_node, valid_subnode_data):
        assert 'spam' in dir(data_node)
        assert 'eggs' in dir(data_node)

    def test_getattr(self, data_node, valid_subnode_data):
        assert data_node.spam == data_node._subnodes['spam']
        assert data_node.eggs == data_node._subnodes['eggs']

    def test_init(self, data_node, valid_subnode_data):
        assert hasattr(data_node, 'spam')
        assert hasattr(data_node, 'eggs')
        assert 'spam' in data_node._subnodes
        assert 'eggs' in data_node._subnodes

    def test_replace(self, data_node, valid_subnode_data):
        data_node.replace({'spam': valid_subnode_data,
                           'eggs': valid_subnode_data})
        if isinstance(valid_subnode_data, np.ndarray):
            assert np.all(data_node.spam.value == valid_subnode_data)
            assert np.all(data_node.eggs.value == valid_subnode_data)
        else:
            assert data_node.spam.value == valid_subnode_data
            assert data_node.eggs.value == valid_subnode_data

    def test_replace_compilation_in_compilation(self, backend, schema_subnode,
                                                valid_subnode_data):
        schema_node = schema.Compilation({
            'inner': schema.Compilation({'spam': schema_subnode,
                                         'eggs': schema_subnode})
        })
        data_node = backend.module.Compilation(schema_node, parent=None,
                                               new_params=backend.new_params)
        data_node.replace({'inner': {'spam': valid_subnode_data,
                                     'eggs': valid_subnode_data}})
        if isinstance(valid_subnode_data, np.ndarray):
            assert np.all(data_node.inner.spam.value == valid_subnode_data)
            assert np.all(data_node.inner.eggs.value == valid_subnode_data)
        else:
            assert data_node.inner.spam.value == valid_subnode_data
            assert data_node.inner.eggs.value == valid_subnode_data

    def test_replace_list_in_compilation(self, backend, schema_subnode,
                                         valid_subnode_data):
        schema_node = schema.Compilation({'spam': schema.List(schema_subnode)})
        data_node = backend.module.Compilation(schema_node, parent=None,
                                               new_params=backend.new_params)
        data_node.replace({'spam': [valid_subnode_data, valid_subnode_data]})
        if isinstance(valid_subnode_data, np.ndarray):
            assert np.all(data_node.spam[0].value == valid_subnode_data)
            assert np.all(data_node.spam[1].value == valid_subnode_data)
        else:
            assert data_node.spam[0].value == valid_subnode_data
            assert data_node.spam[1].value == valid_subnode_data

    def test_validate(self, data_node, valid_subnode_data):
        data_node.spam.replace(valid_subnode_data)
        data_node.eggs.replace(valid_subnode_data)
        data_node.validate()


class TestDate(ItemNodeTestBase):
    class_name = 'Date'
    schema_node = schema.Date()
    valid_data = datetime.date.today()

    def test_default_value(self, data_node):
        assert data_node._storage is None

    def test_default_value_set_on_create(self, backend):
        data_node = backend.module.Date(schema.Date(set_on_create=True),
                                        parent=None,
                                        new_params=backend.new_params)
        assert data_node._storage is not None
        assert data_node.value == datetime.date.today()


class TestDateTime(ItemNodeTestBase):
    class_name = 'DateTime'
    schema_node = schema.DateTime()
    valid_data = datetime.datetime.now()

    def test_default_value(self, data_node):
        assert data_node._storage is None

    def test_default_value_set_on_create(self, backend):
        data_node = backend.module.DateTime(
            schema.DateTime(set_on_create=True), parent=None,
            new_params=backend.new_params
        )
        assert data_node._storage is not None
        assert (data_node.value - datetime.datetime.now()).total_seconds() < 1


@pytest.mark.parametrize('schema_subnode,valid_subnode_data', (
    (schema.Array(dtype='int32'), np.array([23, 42])),
    (schema.Bool(), True),
    (schema.Date(), datetime.date.today()),
    (schema.DateTime(), datetime.datetime.now()),
    (schema.Scalar(dtype='int32'), np.int32(42)),
    (schema.String(), 'spam'),
    (schema.Time(), datetime.time(13, 37, 42, 23)),
))
class TestList:
    @pytest.fixture()
    def data_node(self, backend, schema_subnode):
        data_node = backend.module.List(schema.List(schema_subnode),
                                        parent=None,
                                        new_params=backend.new_params)
        return data_node

    def test_append(self, data_node, valid_subnode_data, backend,
                    schema_subnode):
        data_node.append(valid_subnode_data)
        assert len(data_node._subnodes) == 1
        assert isinstance(data_node._subnodes[0],
                          getattr(backend.module,
                                  type(schema_subnode).__name__))
        if isinstance(valid_subnode_data, np.ndarray):
            assert np.all(data_node._subnodes[0].value == valid_subnode_data)
        else:
            assert data_node._subnodes[0].value == valid_subnode_data

    def test_clear(self, data_node, valid_subnode_data):
        data_node.append(valid_subnode_data)
        assert len(data_node._subnodes) == 1
        data_node.clear()
        assert len(data_node._subnodes) == 0

    def test_complete(self, data_node, valid_subnode_data):
        assert data_node.complete
        data_node.append()
        assert not data_node.complete
        data_node[0].replace(valid_subnode_data)
        assert data_node.complete

    def test_empty(self, data_node, valid_subnode_data):
        assert data_node.empty
        data_node.replace([valid_subnode_data, valid_subnode_data])
        assert not data_node.empty

    def test_getitem(self, data_node, valid_subnode_data, backend,
                     schema_subnode):
        data_node.append(valid_subnode_data)
        assert isinstance(data_node[0], getattr(backend.module,
                                                type(schema_subnode).__name__))
        if isinstance(valid_subnode_data, np.ndarray):
            assert np.all(data_node[0].value == valid_subnode_data)
        else:
            assert data_node[0].value == valid_subnode_data

    def test_init(self, data_node, valid_subnode_data, schema_subnode):
        assert data_node.schema_node.subnode == schema_subnode
        assert data_node._subnodes == []

    def test_len(self, data_node, valid_subnode_data):
        assert len(data_node) == 0
        data_node.append(valid_subnode_data)
        assert len(data_node) == 1

    def test_replace(self, data_node, valid_subnode_data):
        data_node.replace([valid_subnode_data, valid_subnode_data])
        if isinstance(valid_subnode_data, np.ndarray):
            assert np.all(data_node[0].value == valid_subnode_data)
            assert np.all(data_node[1].value == valid_subnode_data)
        else:
            assert data_node[0].value == valid_subnode_data
            assert data_node[1].value == valid_subnode_data

    def test_replace_compilation_in_list(self, backend, schema_subnode,
                                         valid_subnode_data):
        schema_compnode = schema.Compilation({'spam': schema_subnode})
        data_node = backend.module.List(schema.List(schema_compnode),
                                        parent=None,
                                        new_params=backend.new_params)
        data_node.replace([{'spam': valid_subnode_data},
                           {'spam': valid_subnode_data}])
        if isinstance(valid_subnode_data, np.ndarray):
            assert np.all(data_node[0].spam.value == valid_subnode_data)
            assert np.all(data_node[1].spam.value == valid_subnode_data)
        else:
            assert data_node[0].spam.value == valid_subnode_data
            assert data_node[1].spam.value == valid_subnode_data

    def test_replace_list_in_list(self, backend, schema_subnode,
                                  valid_subnode_data):
        schema_lstnode = schema.List(schema_subnode)
        data_node = backend.module.List(schema.List(schema_lstnode),
                                        parent=None,
                                        new_params=backend.new_params)
        data_node.replace([[valid_subnode_data, valid_subnode_data]])
        if isinstance(valid_subnode_data, np.ndarray):
            assert np.all(data_node[0][0].value == valid_subnode_data)
            assert np.all(data_node[0][1].value == valid_subnode_data)
        else:
            assert data_node[0][0].value == valid_subnode_data
            assert data_node[0][1].value == valid_subnode_data

    def test_validate(self, data_node, valid_subnode_data):
        data_node.append(valid_subnode_data)
        data_node.validate()

    def test_validate_fail(self, data_node, valid_subnode_data):
        data_node.schema_node.max_length = 3
        data_node.append(valid_subnode_data)
        data_node.append(valid_subnode_data)
        data_node.append(valid_subnode_data)
        data_node.append(valid_subnode_data)
        with pytest.raises(schema.ValidationError):
            data_node.validate()


class TestScalar(ItemNodeTestBase):
    class_name = 'Scalar'
    schema_node = schema.Scalar(dtype='int32')
    valid_data = np.int32(42)


class TestString(ItemNodeTestBase):
    class_name = 'String'
    schema_node = schema.String()
    valid_data = 'spam'


class TestTime(ItemNodeTestBase):
    class_name = 'Time'
    schema_node = schema.Time()
    valid_data = datetime.time(13, 37, 42, 23)

    def test_default_value(self, data_node):
        assert data_node._storage is None

    def test_default_value_set_on_create(self, backend):
        data_node = backend.module.Time(schema.Time(set_on_create=True),
                                        parent=None,
                                        new_params=backend.new_params)
        assert data_node._storage is not None
        dt = datetime.datetime.now().time()
        assert ((data_node.value.hour, data_node.value.minute) ==
                (dt.hour, dt.minute))


def test_validation_error_chain(backend):
    schema_node = schema.Compilation({
        'spam': schema.List(schema.Compilation({
            'eggs': schema.List(schema.String(max_length=3))}))})
    data_node = backend.module.Compilation(schema_node, parent=None,
                                           new_params=backend.new_params)
    data_node.spam.append({'eggs': ['abc', 'def']})
    data_node.spam.append({'eggs': ['abc', 'def', 'ghij']})
    with pytest.raises(data.SubnodeValidationError) as err:
        data_node.validate()
    assert err.value.node_path() == 'spam[1].eggs[2]'
    assert err.value.__cause__.node_path() == '[1].eggs[2]'
    assert err.value.__cause__.__cause__.node_path() == 'eggs[2]'
    assert err.value.__cause__.__cause__.__cause__.node_path() == '[2]'

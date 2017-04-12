from collections import namedtuple
import datetime
import h5py
import numpy as np
import pytest
from dsch import helpers, schema


backend_data = namedtuple('backend_data', ('module', 'new_params'))


@pytest.fixture(params=('hdf5', 'mat', 'npz'))
def backend(request, tmpdir):
    if request.param == 'hdf5':
        hdf5file = h5py.File(str(tmpdir.join('hdf5test.h5')))
        new_params = {'name': 'test_data', 'parent': hdf5file['/']}
    elif request.param in ('mat', 'npz'):
        new_params = None
    return backend_data(module=helpers.backend_module(request.param),
                        new_params=new_params)


class TestArray:
    def test_clear(self, backend):
        data_node = backend.module.Array(schema.Array(dtype='int'),
                                         parent=None,
                                         new_params=backend.new_params)
        data_node.replace(np.array([23, 42]))
        data_node.clear()
        assert data_node._storage is None

    def test_complete(self, backend):
        data_node = backend.module.Array(schema.Array(dtype='int'),
                                         parent=None,
                                         new_params=backend.new_params)
        assert not data_node.complete
        data_node.replace(np.array([23, 42]))
        assert data_node.complete

    def test_empty(self, backend):
        data_node = backend.module.Array(schema.Array(dtype='int'),
                                         parent=None,
                                         new_params=backend.new_params)
        assert data_node.empty
        data_node.replace(np.array([23, 42]))
        assert not data_node.empty

    def test_getitem(self, backend):
        data_node = backend.module.Array(schema.Array(dtype='int'),
                                         parent=None,
                                         new_params=backend.new_params)
        data_node.replace(np.array([23, 42]))
        assert data_node[0] == 23
        assert data_node[1] == 42
        assert np.all(data_node[()] == np.array([23, 42]))

    def test_init_new(self, backend):
        schema_node = schema.Array(dtype='int')
        data_node = backend.module.Array(schema_node, parent=None,
                                         new_params=backend.new_params)
        assert data_node.schema_node == schema_node
        assert data_node._storage is None

    def test_replace(self, backend):
        data_node = backend.module.Array(schema.Array(dtype='int'),
                                         parent=None,
                                         new_params=backend.new_params)
        data_node.replace(np.array([23, 42]))
        assert np.all(data_node.value == np.array([23, 42]))

    def test_resize(self, backend):
        data_node = backend.module.Array(schema.Array(dtype='int'),
                                         parent=None,
                                         new_params=backend.new_params)
        data_node.replace(np.array([42]))
        assert data_node.value.shape == (1,)
        data_node.resize((5,))
        assert data_node.value.ndim == 1
        assert data_node.value.shape == (5,)

    def test_setitem(self, backend):
        data_node = backend.module.Array(schema.Array(dtype='int'),
                                         parent=None,
                                         new_params=backend.new_params)
        data_node.replace(np.array([5, 23, 42]))
        data_node[0] = 1
        assert np.all(data_node.value == np.array([1, 23, 42]))
        data_node[1:] = np.array([2, 3])
        assert np.all(data_node.value == np.array([1, 2, 3]))
        data_node[()] = np.array([42, 23, 5])
        assert np.all(data_node.value == np.array([42, 23, 5]))

    def test_validate(self, backend):
        data_node = backend.module.Array(schema.Array(dtype='int'),
                                         parent=None,
                                         new_params=backend.new_params)
        data_node.replace(np.array([23, 42]))
        data_node.validate()

    def test_validate_depends_on(self, backend):
        comp = backend.module.Compilation(schema.Compilation({
            'time': schema.Array(dtype='float'),
            'voltage': schema.Array(dtype='float', depends_on='time'),
        }), parent=None, new_params=backend.new_params)
        comp.time.replace(np.array([0, 1, 2], dtype='float'))
        comp.voltage.replace(np.array([2, 3, 5], dtype='float'))
        comp.voltage.validate()

    def test_value(self, backend):
        data_node = backend.module.Array(schema.Array(dtype='int'),
                                         parent=None,
                                         new_params=backend.new_params)
        data_node.replace(np.array([23, 42]))
        assert isinstance(data_node.value, np.ndarray)


class TestBool:
    def test_clear(self, backend):
        data_node = backend.module.Bool(schema.Bool(), parent=None,
                                        new_params=backend.new_params)
        data_node.replace(False)
        data_node.clear()
        assert data_node._storage is None

    def test_complete(self, backend):
        data_node = backend.module.Bool(schema.Bool(), parent=None,
                                        new_params=backend.new_params)
        assert not data_node.complete
        data_node.replace(False)
        assert data_node.complete

    def test_empty(self, backend):
        data_node = backend.module.Bool(schema.Bool(), parent=None,
                                        new_params=backend.new_params)
        assert data_node.empty
        data_node.replace(False)
        assert not data_node.empty

    def test_init_new(self, backend):
        schema_node = schema.Bool()
        data_node = backend.module.Bool(schema_node, parent=None,
                                        new_params=backend.new_params)
        assert data_node.schema_node == schema_node
        assert data_node._storage is None

    def test_replace(self, backend):
        data_node = backend.module.Bool(schema.Bool(), parent=None,
                                        new_params=backend.new_params)
        data_node.replace(True)
        assert data_node.value is True
        data_node.replace(False)
        assert data_node.value is False

    def test_validate(self, backend):
        data_node = backend.module.Bool(schema.Bool(), parent=None,
                                        new_params=backend.new_params)
        data_node.replace(True)
        data_node.validate()

    def test_value(self, backend):
        data_node = backend.module.Bool(schema.Bool(), parent=None,
                                        new_params=backend.new_params)
        data_node.replace(True)
        assert isinstance(data_node.value, bool)


class TestCompilation:
    def test_clear(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        comp.spam.replace(True)
        comp.eggs.replace(False)
        comp.clear()
        assert comp.spam._storage is None
        assert comp.eggs._storage is None

    def test_complete(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        assert not comp.complete
        comp.spam.replace(True)
        assert not comp.complete
        comp.eggs.replace(False)
        assert comp.complete

    def test_complete_optionals(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()},
                                         optionals=['eggs'])
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        assert not comp.complete
        comp.spam.replace(True)
        assert comp.complete

    def test_empty(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        assert comp.empty
        comp.spam.replace(True)
        assert not comp.empty

    def test_dir(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        assert 'spam' in dir(comp)
        assert 'eggs' in dir(comp)

    def test_getattr(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        assert comp.spam == comp._subnodes['spam']
        assert comp.eggs == comp._subnodes['eggs']

    def test_init(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        assert comp.schema_node == schema_node
        assert hasattr(comp, 'spam')
        assert hasattr(comp, 'eggs')
        assert 'spam' in comp._subnodes
        assert 'eggs' in comp._subnodes

    def test_replace(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        comp.replace({'spam': True, 'eggs': False})
        assert comp.spam.value is True
        assert comp.eggs.value is False

    def test_replace_compilation_in_compilation(self, backend):
        schema_node = schema.Compilation({
            'inner': schema.Compilation({'spam': schema.Bool(),
                                         'eggs': schema.Bool()})
        })
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        comp.replace({'inner': {'spam': True, 'eggs': False}})
        assert comp.inner.spam.value is True
        assert comp.inner.eggs.value is False

    def test_replace_list_in_compilation(self, backend):
        schema_node = schema.Compilation({'spam': schema.List(schema.Bool())})
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        comp.replace({'spam': [True, False]})
        assert comp.spam[0].value is True
        assert comp.spam[1].value is False

    def test_validate(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node, parent=None,
                                          new_params=backend.new_params)
        comp.spam.replace(True)
        comp.eggs.replace(False)
        comp.validate()


class TestDate:
    def test_clear(self, backend):
        data_node = backend.module.Date(schema.Date(), parent=None,
                                        new_params=backend.new_params)
        data_node.replace(datetime.date.today())
        data_node.clear()
        assert data_node._storage is None

    def test_complete(self, backend):
        data_node = backend.module.Date(schema.Date(), parent=None,
                                        new_params=backend.new_params)
        assert not data_node.complete
        data_node.replace(datetime.date.today())
        assert data_node.complete

    def test_empty(self, backend):
        data_node = backend.module.Date(schema.Date(), parent=None,
                                        new_params=backend.new_params)
        assert data_node.empty
        data_node.replace(datetime.date.today())
        assert not data_node.empty

    def test_init(self, backend):
        schema_node = schema.Date()
        data_node = backend.module.Date(schema_node, parent=None,
                                        new_params=backend.new_params)
        assert data_node.schema_node == schema_node

    def test_default_value(self, backend):
        data_node = backend.module.Date(schema.Date(set_on_create=False),
                                        parent=None,
                                        new_params=backend.new_params)
        assert data_node._storage is None

    def test_default_value_set_on_create(self, backend):
        data_node = backend.module.Date(schema.Date(set_on_create=True),
                                        parent=None,
                                        new_params=backend.new_params)
        assert data_node._storage is not None
        assert data_node.value == datetime.date.today()

    def test_replace(self, backend):
        data_node = backend.module.Date(schema.Date(), parent=None,
                                        new_params=backend.new_params)
        data_node.replace(datetime.date.today())
        assert data_node.value == datetime.date.today()

    def test_validate(self, backend):
        data_node = backend.module.Date(schema.Date(), parent=None,
                                        new_params=backend.new_params)
        data_node.replace(datetime.date.today())
        data_node.validate()

    def test_value(self, backend):
        data_node = backend.module.Date(schema.Date(), parent=None,
                                        new_params=backend.new_params)
        data_node.replace(datetime.date.today())
        assert isinstance(data_node.value, datetime.date)


class TestDateTime:
    def test_clear(self, backend):
        data_node = backend.module.DateTime(schema.DateTime(), parent=None,
                                            new_params=backend.new_params)
        dt = datetime.datetime.now()
        data_node.replace(dt)
        data_node.clear()
        assert data_node._storage is None

    def test_complete(self, backend):
        data_node = backend.module.DateTime(schema.DateTime(), parent=None,
                                            new_params=backend.new_params)
        assert not data_node.complete
        dt = datetime.datetime.now()
        data_node.replace(dt)
        assert data_node.complete

    def test_empty(self, backend):
        data_node = backend.module.DateTime(schema.DateTime(), parent=None,
                                            new_params=backend.new_params)
        assert data_node.empty
        dt = datetime.datetime.now()
        data_node.replace(dt)
        assert not data_node.empty

    def test_init(self, backend):
        schema_node = schema.DateTime()
        data_node = backend.module.DateTime(schema_node, parent=None,
                                            new_params=backend.new_params)
        assert data_node.schema_node == schema_node

    def test_default_value(self, backend):
        data_node = backend.module.DateTime(
            schema.DateTime(set_on_create=False), parent=None,
            new_params=backend.new_params
        )
        assert data_node._storage is None

    def test_default_value_set_on_create(self, backend):
        data_node = backend.module.DateTime(
            schema.DateTime(set_on_create=True), parent=None,
            new_params=backend.new_params
        )
        assert data_node._storage is not None
        assert (data_node.value - datetime.datetime.now()).total_seconds() < 1

    def test_replace(self, backend):
        data_node = backend.module.DateTime(schema.DateTime(), parent=None,
                                            new_params=backend.new_params)
        dt = datetime.datetime.now()
        data_node.replace(dt)
        assert data_node.value == dt

    def test_validate(self, backend):
        data_node = backend.module.DateTime(schema.DateTime(), parent=None,
                                            new_params=backend.new_params)
        data_node.replace(datetime.datetime.now())
        data_node.validate()

    def test_value(self, backend):
        data_node = backend.module.DateTime(schema.DateTime(), parent=None,
                                            new_params=backend.new_params)
        data_node.replace(datetime.datetime.now())
        assert isinstance(data_node.value, datetime.datetime)


class TestList:
    def test_append(self, backend):
        data_node = backend.module.List(schema.List(schema.Bool()),
                                        parent=None,
                                        new_params=backend.new_params)
        data_node.append(False)
        assert len(data_node._subnodes) == 1
        assert isinstance(data_node._subnodes[0], backend.module.Bool)
        assert data_node._subnodes[0].value is False

    def test_clear(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.module.List(schema.List(schema_subnode),
                                        parent=None,
                                        new_params=backend.new_params)
        data_node.append(True)
        assert len(data_node._subnodes) == 1
        data_node.clear()
        assert len(data_node._subnodes) == 0

    def test_complete(self, backend):
        data_node = backend.module.List(schema.List(schema.Bool()),
                                        parent=None,
                                        new_params=backend.new_params)
        assert data_node.complete
        data_node.append()
        assert not data_node.complete
        data_node[0].replace(True)
        assert data_node.complete

    def test_empty(self, backend):
        data_node = backend.module.List(schema.List(schema.Bool()),
                                        parent=None,
                                        new_params=backend.new_params)
        assert data_node.empty
        data_node.replace([True, False])
        assert not data_node.empty

    def test_getitem(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.module.List(schema.List(schema_subnode),
                                        parent=None,
                                        new_params=backend.new_params)
        data_node.append(False)
        assert isinstance(data_node[0], backend.module.Bool)
        assert data_node[0].value is False

    def test_init(self, backend):
        schema_node = schema.List(schema.Bool())
        data_node = backend.module.List(schema_node, parent=None,
                                        new_params=backend.new_params)
        assert data_node.schema_node == schema_node
        assert data_node._subnodes == []

    def test_len(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.module.List(schema.List(schema_subnode),
                                        parent=None,
                                        new_params=backend.new_params)
        assert len(data_node) == 0
        data_node.append(True)
        assert len(data_node) == 1

    def test_replace(self, backend):
        data_node = backend.module.List(schema.List(schema.Bool()),
                                        parent=None,
                                        new_params=backend.new_params)
        data_node.replace([True, False])
        assert data_node[0].value is True
        assert data_node[1].value is False

    def test_replace_compilation_in_list(self, backend):
        schema_subnode = schema.Compilation({'spam': schema.Bool()})
        data_node = backend.module.List(schema.List(schema_subnode),
                                        parent=None,
                                        new_params=backend.new_params)
        data_node.replace([{'spam': True}, {'spam': False}])
        assert data_node[0].spam.value is True
        assert data_node[1].spam.value is False

    def test_replace_list_in_list(self, backend):
        schema_subnode = schema.List(schema.Bool())
        data_node = backend.module.List(schema.List(schema_subnode),
                                        parent=None,
                                        new_params=backend.new_params)
        data_node.replace([[True, False]])
        assert data_node[0][0].value is True
        assert data_node[0][1].value is False

    def test_validate(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.module.List(schema.List(schema_subnode),
                                        parent=None,
                                        new_params=backend.new_params)
        data_node.append(True)
        data_node.validate()

    def test_validate_fail(self, backend):
        schema_subnode = schema.Bool()
        subnode = schema.List(schema_subnode, max_length=3)
        data_node = backend.module.List(subnode, parent=None,
                                        new_params=backend.new_params)
        data_node.append(True)
        data_node.append(False)
        data_node.append(True)
        data_node.append(False)
        with pytest.raises(schema.ValidationError):
            data_node.validate()


class TestScalar:
    def test_clear(self, backend):
        data_node = backend.module.Scalar(schema.Scalar(dtype='int32'),
                                          parent=None,
                                          new_params=backend.new_params)
        data_node.replace(42)
        data_node.clear()
        assert data_node._storage is None

    def test_complete(self, backend):
        data_node = backend.module.Scalar(schema.Scalar(dtype='int32'),
                                          parent=None,
                                          new_params=backend.new_params)
        assert not data_node.complete
        data_node.replace(42)
        assert data_node.complete

    def test_empty(self, backend):
        data_node = backend.module.Scalar(schema.Scalar(dtype='int32'),
                                          parent=None,
                                          new_params=backend.new_params)
        assert data_node.empty
        data_node.replace(42)
        assert not data_node.empty

    def test_init_new(self, backend):
        schema_node = schema.Scalar(dtype='int32')
        data_node = backend.module.Scalar(schema_node, parent=None,
                                          new_params=backend.new_params)
        assert data_node.schema_node == schema_node
        assert data_node._storage is None

    def test_replace(self, backend):
        data_node = backend.module.Scalar(schema.Scalar(dtype='int32'),
                                          parent=None,
                                          new_params=backend.new_params)
        data_node.replace(42)
        assert data_node.value == 42

    def test_validate(self, backend):
        data_node = backend.module.Scalar(schema.Scalar(dtype='int32'),
                                          parent=None,
                                          new_params=backend.new_params)
        data_node.replace(42)
        data_node.validate()

    def test_value(self, backend):
        data_node = backend.module.Scalar(schema.Scalar(dtype='int32'),
                                          parent=None,
                                          new_params=backend.new_params)
        data_node.replace(42)
        assert isinstance(data_node.value, np.int32)


class TestString:
    def test_clear(self, backend):
        data_node = backend.module.String(schema.String(), parent=None,
                                          new_params=backend.new_params)
        data_node.replace('spam')
        data_node.clear()
        assert data_node._storage is None

    def test_complete(self, backend):
        data_node = backend.module.String(schema.String(), parent=None,
                                          new_params=backend.new_params)
        assert not data_node.complete
        data_node.replace('spam')
        assert data_node.complete

    def test_empty(self, backend):
        data_node = backend.module.String(schema.String(), parent=None,
                                          new_params=backend.new_params)
        assert data_node.empty
        data_node.replace('spam')
        assert not data_node.empty

    def test_init_new(self, backend):
        schema_node = schema.String()
        data_node = backend.module.String(schema_node, parent=None,
                                          new_params=backend.new_params)
        assert data_node.schema_node == schema_node
        assert data_node._storage is None

    def test_replace(self, backend):
        data_node = backend.module.String(schema.String(), parent=None,
                                          new_params=backend.new_params)
        data_node.replace('spam')
        assert data_node.value == 'spam'

    def test_validate(self, backend):
        data_node = backend.module.String(schema.String(), parent=None,
                                          new_params=backend.new_params)
        data_node.replace('spam')
        data_node.validate()

    def test_value(self, backend):
        data_node = backend.module.String(schema.String(), parent=None,
                                          new_params=backend.new_params)
        data_node.replace('spam')
        assert isinstance(data_node.value, str)


class TestTime:
    def test_clear(self, backend):
        data_node = backend.module.Time(schema.Time(), parent=None,
                                        new_params=backend.new_params)
        dt = datetime.time(13, 37, 42, 23)
        data_node.replace(dt)
        data_node.clear()
        assert data_node._storage is None

    def test_complete(self, backend):
        data_node = backend.module.Time(schema.Time(), parent=None,
                                        new_params=backend.new_params)
        assert not data_node.complete
        dt = datetime.time(13, 37, 42, 23)
        data_node.replace(dt)
        assert data_node.complete

    def test_empty(self, backend):
        data_node = backend.module.Time(schema.Time(), parent=None,
                                        new_params=backend.new_params)
        assert data_node.empty
        dt = datetime.time(13, 37, 42, 23)
        data_node.replace(dt)
        assert not data_node.empty

    def test_init(self, backend):
        schema_node = schema.Time()
        data_node = backend.module.Time(schema_node, parent=None,
                                        new_params=backend.new_params)
        assert data_node.schema_node == schema_node

    def test_default_value(self, backend):
        data_node = backend.module.Time(schema.Time(set_on_create=False),
                                        parent=None,
                                        new_params=backend.new_params)
        assert data_node._storage is None

    def test_default_value_set_on_create(self, backend):
        data_node = backend.module.Time(schema.Time(set_on_create=True),
                                        parent=None,
                                        new_params=backend.new_params)
        assert data_node._storage is not None
        dt = datetime.datetime.now().time()
        assert ((data_node.value.hour, data_node.value.minute) ==
                (dt.hour, dt.minute))

    def test_replace(self, backend):
        data_node = backend.module.Time(schema.Time(), parent=None,
                                        new_params=backend.new_params)
        dt = datetime.time(13, 37, 42, 23)
        data_node.replace(dt)
        assert data_node.value == dt

    def test_validate(self, backend):
        data_node = backend.module.Time(schema.Time(), parent=None,
                                        new_params=backend.new_params)
        data_node.replace(datetime.time(13, 37, 42, 23))
        data_node.validate()

    def test_value(self, backend):
        data_node = backend.module.Time(schema.Time(), parent=None,
                                        new_params=backend.new_params)
        data_node.replace(datetime.time(13, 37, 42, 23))
        assert isinstance(data_node.value, datetime.time)

import importlib
import pytest
from dsch import schema


@pytest.fixture(params=('npz',))
def backend(request):
    backend_module = importlib.import_module('dsch.backends.' + request.param)
    return backend_module


class TestBool:
    def test_init(self, backend):
        schema_node = schema.Bool()
        data_node = backend.Bool(schema_node)
        assert data_node.schema_node == schema_node
        assert data_node.storage is None

    def test_replace(self, backend):
        data_node = backend.Bool(schema.Bool())
        data_node.replace(True)
        assert data_node.value is True
        data_node.replace(False)
        assert data_node.value is False

    def test_validate(self, backend):
        data_node = backend.Bool(schema.Bool())
        data_node.replace(True)
        data_node.validate()

    def test_value(self, backend):
        data_node = backend.Bool(schema.Bool())
        assert isinstance(data_node.value, bool)


class TestCompilation:
    def test_dir(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.Compilation(schema_node)
        assert 'spam' in dir(comp)
        assert 'eggs' in dir(comp)

    def test_getattr(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.Compilation(schema_node)
        assert comp.spam == comp._subnodes['spam']
        assert comp.eggs == comp._subnodes['eggs']

    def test_init(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.Compilation(schema_node)
        assert comp.schema_node == schema_node
        assert hasattr(comp, 'spam')
        assert hasattr(comp, 'eggs')
        assert 'spam' in comp._subnodes
        assert 'eggs' in comp._subnodes

    def test_replace(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.Compilation(schema_node)
        comp.replace({'spam': True, 'eggs': False})
        assert comp.spam.value is True
        assert comp.eggs.value is False

    def test_replace_compilation_in_compilation(self, backend):
        schema_node = schema.Compilation({
            'inner': schema.Compilation({'spam': schema.Bool(),
                                         'eggs': schema.Bool()})
        })
        comp = backend.Compilation(schema_node)
        comp.replace({'inner': {'spam': True, 'eggs': False}})
        assert comp.inner.spam.value is True
        assert comp.inner.eggs.value is False

    def test_replace_list_in_compilation(self, backend):
        schema_node = schema.Compilation({'spam': schema.List(schema.Bool())})
        comp = backend.Compilation(schema_node)
        comp.replace({'spam': [True, False]})
        assert comp.spam[0].value is True
        assert comp.spam[1].value is False

    def test_validate(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.Compilation(schema_node)
        comp.spam.replace(True)
        comp.eggs.replace(False)
        comp.validate()


class TestList:
    def test_append(self, backend):
        schema_node = schema.List(schema.Bool())
        data_node = backend.List(schema_node)
        data_node.append(False)
        assert len(data_node._subnodes) == 1
        assert isinstance(data_node._subnodes[0], backend.Bool)
        assert data_node._subnodes[0].value is False

    def test_clear(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.List(schema.List(schema_subnode))
        data_subnode = backend.Bool(schema_subnode)
        data_node._subnodes.append(data_subnode)
        assert len(data_node._subnodes) == 1
        data_node.clear()
        assert len(data_node._subnodes) == 0

    def test_getitem(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.List(schema.List(schema_subnode))
        data_subnode = backend.Bool(schema_subnode)
        data_subnode.replace(False)
        data_node._subnodes.append(data_subnode)
        assert isinstance(data_node[0], backend.Bool)
        assert data_node[0].value is False

    def test_init(self, backend):
        schema_node = schema.List(schema.Bool())
        data_node = backend.List(schema_node)
        assert data_node.schema_node == schema_node
        assert data_node._subnodes == []

    def test_len(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.List(schema.List(schema_subnode))
        data_subnode = backend.Bool(schema_subnode)
        assert len(data_node) == 0
        data_node._subnodes.append(data_subnode)
        assert len(data_node) == 1

    def test_replace(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.List(schema.List(schema_subnode))
        data_node.replace([True, False])
        assert data_node[0].value is True
        assert data_node[1].value is False

    def test_replace_compilation_in_list(self, backend):
        schema_subnode = schema.Compilation({'spam': schema.Bool()})
        data_node = backend.List(schema.List(schema_subnode))
        data_node.replace([{'spam': True}, {'spam': False}])
        assert data_node[0].spam.value is True
        assert data_node[1].spam.value is False

    def test_replace_list_in_list(self, backend):
        schema_subnode = schema.List(schema.Bool())
        data_node = backend.List(schema.List(schema_subnode))
        data_node.replace([[True, False]])
        assert data_node[0][0].value is True
        assert data_node[0][1].value is False

    def test_validate(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.List(schema.List(schema_subnode))
        data_subnode = backend.Bool(schema_subnode)
        data_subnode.replace(False)
        data_node._subnodes.append(data_subnode)
        data_node.validate()


class TestString:
    def test_init(self, backend):
        schema_node = schema.String()
        data_node = backend.String(schema_node)
        assert data_node.schema_node == schema_node
        assert data_node.storage is None

    def test_replace(self, backend):
        data_node = backend.String(schema.String())
        data_node.replace('spam')
        assert data_node.value == 'spam'

    def test_validate(self, backend):
        data_node = backend.String(schema.String())
        data_node.replace('spam')
        data_node.validate()

    def test_value(self, backend):
        data_node = backend.String(schema.String())
        data_node.replace('spam')
        assert isinstance(data_node.value, str)

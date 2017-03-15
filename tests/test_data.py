from collections import namedtuple
import pytest
from dsch import helpers, schema


backend_data = namedtuple('backend_data', ('module', 'new_params'))


@pytest.fixture(params=('npz',))
def backend(request, tmpdir):
    if request.param == 'npz':
        new_params = None
    return backend_data(module=helpers.backend_module(request.param),
                        new_params=new_params)


class TestBool:
    def test_init(self, backend):
        schema_node = schema.Bool()
        data_node = backend.module.Bool(schema_node,
                                        new_params=backend.new_params)
        assert data_node.schema_node == schema_node

    def test_replace(self, backend):
        data_node = backend.module.Bool(schema.Bool(),
                                        new_params=backend.new_params)
        data_node.replace(True)
        assert data_node.value is True
        data_node.replace(False)
        assert data_node.value is False

    def test_validate(self, backend):
        data_node = backend.module.Bool(schema.Bool(),
                                        new_params=backend.new_params)
        data_node.replace(True)
        data_node.validate()

    def test_value(self, backend):
        data_node = backend.module.Bool(schema.Bool(),
                                        new_params=backend.new_params)
        assert isinstance(data_node.value, bool)


class TestCompilation:
    def test_dir(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node,
                                          new_params=backend.new_params)
        assert 'spam' in dir(comp)
        assert 'eggs' in dir(comp)

    def test_getattr(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node,
                                          new_params=backend.new_params)
        assert comp.spam == comp._subnodes['spam']
        assert comp.eggs == comp._subnodes['eggs']

    def test_init(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node,
                                          new_params=backend.new_params)
        assert comp.schema_node == schema_node
        assert hasattr(comp, 'spam')
        assert hasattr(comp, 'eggs')
        assert 'spam' in comp._subnodes
        assert 'eggs' in comp._subnodes

    def test_replace(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node,
                                          new_params=backend.new_params)
        comp.replace({'spam': True, 'eggs': False})
        assert comp.spam.value is True
        assert comp.eggs.value is False

    def test_replace_compilation_in_compilation(self, backend):
        schema_node = schema.Compilation({
            'inner': schema.Compilation({'spam': schema.Bool(),
                                         'eggs': schema.Bool()})
        })
        comp = backend.module.Compilation(schema_node,
                                          new_params=backend.new_params)
        comp.replace({'inner': {'spam': True, 'eggs': False}})
        assert comp.inner.spam.value is True
        assert comp.inner.eggs.value is False

    def test_replace_list_in_compilation(self, backend):
        schema_node = schema.Compilation({'spam': schema.List(schema.Bool())})
        comp = backend.module.Compilation(schema_node,
                                          new_params=backend.new_params)
        comp.replace({'spam': [True, False]})
        assert comp.spam[0].value is True
        assert comp.spam[1].value is False

    def test_validate(self, backend):
        schema_node = schema.Compilation({'spam': schema.Bool(),
                                          'eggs': schema.Bool()})
        comp = backend.module.Compilation(schema_node,
                                          new_params=backend.new_params)
        comp.spam.replace(True)
        comp.eggs.replace(False)
        comp.validate()


class TestList:
    def test_append(self, backend):
        data_node = backend.module.List(schema.List(schema.Bool()),
                                        new_params=backend.new_params)
        data_node.append(False)
        assert len(data_node._subnodes) == 1
        assert isinstance(data_node._subnodes[0], backend.module.Bool)
        assert data_node._subnodes[0].value is False

    def test_clear(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.module.List(schema.List(schema_subnode),
                                        new_params=backend.new_params)
        data_node.append(True)
        assert len(data_node._subnodes) == 1
        data_node.clear()
        assert len(data_node._subnodes) == 0

    def test_getitem(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.module.List(schema.List(schema_subnode),
                                        new_params=backend.new_params)
        data_node.append(False)
        assert isinstance(data_node[0], backend.module.Bool)
        assert data_node[0].value is False

    def test_init(self, backend):
        schema_node = schema.List(schema.Bool())
        data_node = backend.module.List(schema_node,
                                        new_params=backend.new_params)
        assert data_node.schema_node == schema_node
        assert data_node._subnodes == []

    def test_len(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.module.List(schema.List(schema_subnode),
                                        new_params=backend.new_params)
        assert len(data_node) == 0
        data_node.append(True)
        assert len(data_node) == 1

    def test_replace(self, backend):
        data_node = backend.module.List(schema.List(schema.Bool()),
                                        new_params=backend.new_params)
        data_node.replace([True, False])
        assert data_node[0].value is True
        assert data_node[1].value is False

    def test_replace_compilation_in_list(self, backend):
        schema_subnode = schema.Compilation({'spam': schema.Bool()})
        data_node = backend.module.List(schema.List(schema_subnode),
                                        new_params=backend.new_params)
        data_node.replace([{'spam': True}, {'spam': False}])
        assert data_node[0].spam.value is True
        assert data_node[1].spam.value is False

    def test_replace_list_in_list(self, backend):
        schema_subnode = schema.List(schema.Bool())
        data_node = backend.module.List(schema.List(schema_subnode),
                                        new_params=backend.new_params)
        data_node.replace([[True, False]])
        assert data_node[0][0].value is True
        assert data_node[0][1].value is False

    def test_validate(self, backend):
        schema_subnode = schema.Bool()
        data_node = backend.module.List(schema.List(schema_subnode),
                                        new_params=backend.new_params)
        data_node.append(True)
        data_node.validate()


class TestString:
    def test_init(self, backend):
        schema_node = schema.String()
        data_node = backend.module.String(schema_node,
                                          new_params=backend.new_params)
        assert data_node.schema_node == schema_node

    def test_replace(self, backend):
        data_node = backend.module.String(schema.String(),
                                          new_params=backend.new_params)
        data_node.replace('spam')
        assert data_node.value == 'spam'

    def test_validate(self, backend):
        data_node = backend.module.String(schema.String(),
                                          new_params=backend.new_params)
        data_node.replace('spam')
        data_node.validate()

    def test_value(self, backend):
        data_node = backend.module.String(schema.String(),
                                          new_params=backend.new_params)
        data_node.replace('spam')
        assert isinstance(data_node.value, str)

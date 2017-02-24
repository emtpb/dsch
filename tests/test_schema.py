import numpy as np
import pytest
from dsch import schema


class TestBool:
    def test_from_dict(self):
        node = schema.Bool.from_dict({'node_type': 'Bool', 'config': {}})
        assert isinstance(node, schema.Bool)

    def test_to_dict(self):
        node = schema.Bool()
        node_dict = node.to_dict()
        assert 'node_type' in node_dict
        assert node_dict['node_type'] == 'Bool'
        assert 'config' in node_dict
        assert node_dict['config'] == {}

    @pytest.mark.parametrize('test_data', (True, False))
    def test_validate(self, test_data):
        node = schema.Bool()
        node.validate(test_data)

    @pytest.mark.parametrize('test_data', (0, 1, [23, 42], 'spam',
                                           np.array([True])))
    def test_validate_fail(self, test_data):
        node = schema.Bool()
        with pytest.raises(schema.ValidationError):
            node.validate(test_data)


class TestCompilation:
    class ExampleData:
        pass

    def test_from_dict(self):
        node_dict = {'node_type': 'Compilation', 'config': {
            'subnodes': {
                'spam': {'node_type': 'Bool', 'config': {}},
                'eggs': {'node_type': 'Bool', 'config': {}},
            }}}
        node = schema.Compilation.from_dict(node_dict)
        assert len(node.subnodes) == 2
        assert 'spam' in node.subnodes
        assert 'eggs' in node.subnodes
        assert isinstance(node.subnodes['spam'], schema.Bool)
        assert isinstance(node.subnodes['eggs'], schema.Bool)

    def test_from_dict_compilation_in_compilation(self):
        node_dict = {'node_type': 'Compilation', 'config': {
            'subnodes': {
                'bacon': {
                    'node_type': 'Compilation',
                    'config': {
                        'subnodes': {
                            'spam': {'node_type': 'Bool', 'config': {}},
                            'eggs': {'node_type': 'Bool', 'config': {}},
                        }}}}}}
        node = schema.Compilation.from_dict(node_dict)
        assert len(node.subnodes) == 1
        assert 'bacon' in node.subnodes
        assert isinstance(node.subnodes['bacon'], schema.Compilation)
        assert len(node.subnodes['bacon'].subnodes) == 2
        assert 'spam' in node.subnodes['bacon'].subnodes
        assert 'eggs' in node.subnodes['bacon'].subnodes
        assert isinstance(node.subnodes['bacon'].subnodes['spam'], schema.Bool)
        assert isinstance(node.subnodes['bacon'].subnodes['eggs'], schema.Bool)

    def test_init(self):
        node = schema.Compilation({'spam': schema.Bool(),
                                   'eggs': schema.Bool()})
        assert len(node.subnodes) == 2
        assert 'spam' in node.subnodes
        assert 'eggs' in node.subnodes

    def test_to_dict(self):
        node = schema.Compilation({'spam': schema.Bool(),
                                   'eggs': schema.Bool()})
        node_dict = node.to_dict()
        assert 'node_type' in node_dict
        assert node_dict['node_type'] == 'Compilation'
        assert 'config' in node_dict
        assert 'subnodes' in node_dict['config']
        assert 'spam' in node_dict['config']['subnodes']
        assert 'eggs' in node_dict['config']['subnodes']
        for subnode_dict in node_dict['config']['subnodes'].values():
            assert 'node_type' in subnode_dict
            assert 'config' in subnode_dict
            assert subnode_dict['node_type'] == 'Bool'
            assert subnode_dict['config'] == {}

    def test_validate(self):
        node = schema.Compilation({'spam': schema.Bool(),
                                   'eggs': schema.Bool()})
        test_data = self.ExampleData()
        test_data.spam = True
        test_data.eggs = False
        node.validate(test_data)

    def test_validate_fail_invalid(self):
        node = schema.Compilation({'spam': schema.Bool(),
                                   'eggs': schema.Bool()})
        test_data = self.ExampleData()
        test_data.spam = True
        test_data.eggs = 42
        with pytest.raises(schema.ValidationError) as err:
            node.validate(test_data)
        assert err.value.message == 'Invalid type/value.'

    def test_validate_fail_missing(self):
        node = schema.Compilation({'spam': schema.Bool(),
                                   'eggs': schema.Bool()})
        test_data = self.ExampleData()
        test_data.spam = True
        with pytest.raises(schema.ValidationError) as err:
            node.validate(test_data)
        assert err.value.message == 'Missing data attribute.'
        assert err.value.expected == 'eggs'


def test_validation_error():
    ve = schema.ValidationError('Error message.', 'foo', 'baz')
    assert ve.message == 'Error message.'
    assert ve.expected == 'foo'
    assert ve.got == 'baz'

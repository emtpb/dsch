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

    def test_from_dict_list_in_compilation(self):
        node_dict = {'node_type': 'Compilation', 'config': {
            'subnodes': {
                'bacon': {
                    'node_type': 'List',
                    'config': {
                        'subnode': {'node_type': 'Bool', 'config': {}}
                    }}}}}
        node = schema.Compilation.from_dict(node_dict)
        assert len(node.subnodes) == 1
        assert 'bacon' in node.subnodes
        assert isinstance(node.subnodes['bacon'], schema.List)
        assert isinstance(node.subnodes['bacon'].subnode, schema.Bool)

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


class TestList:
    def test_from_dict(self):
        node_dict = {'node_type': 'List', 'config': {
            'subnode': {'node_type': 'Bool', 'config': {}}
        }}
        node = schema.List.from_dict(node_dict)
        assert isinstance(node.subnode, schema.Bool)

    def test_from_dict_compilation_in_list(self):
        node_dict = {'node_type': 'List', 'config': {
            'subnode': {
                'node_type': 'Compilation',
                'config': {
                    'subnodes': {
                        'spam': {'node_type': 'Bool', 'config': {}},
                        'eggs': {'node_type': 'Bool', 'config': {}},
                    }}}}}
        node = schema.List.from_dict(node_dict)
        assert isinstance(node.subnode, schema.Compilation)
        assert len(node.subnode.subnodes) == 2
        assert 'spam' in node.subnode.subnodes
        assert 'eggs' in node.subnode.subnodes
        assert isinstance(node.subnode.subnodes['spam'], schema.Bool)
        assert isinstance(node.subnode.subnodes['eggs'], schema.Bool)

    def test_from_dict_list_in_list(self):
        node_dict = {'node_type': 'List', 'config': {
            'subnode': {
                'node_type': 'List',
                'config': {
                    'subnode': {'node_type': 'Bool', 'config': {}}
                }}}}
        node = schema.List.from_dict(node_dict)
        assert isinstance(node.subnode, schema.List)
        assert isinstance(node.subnode.subnode, schema.Bool)

    def test_init(self):
        node = schema.List(schema.Bool())
        assert isinstance(node.subnode, schema.Bool)

    def test_to_dict(self):
        subnode = schema.Bool()
        node = schema.List(subnode)
        node_dict = node.to_dict()
        assert 'node_type' in node_dict
        assert node_dict['node_type'] == 'List'
        assert 'config' in node_dict
        assert 'subnode' in node_dict['config']
        assert node_dict['config']['subnode'] == subnode.to_dict()


class TestString:
    @pytest.mark.parametrize('config,expected', (
        ({}, {'min_length': None, 'max_length': None}),
        ({'min_length': 3}, {'min_length': 3, 'max_length': None}),
        ({'max_length': 5}, {'min_length': None, 'max_length': 5}),
        ({'min_length': 3, 'max_length': 5},
         {'min_length': 3, 'max_length': 5}),
    ))
    def test_from_dict(self, config, expected):
        node_dict = {'node_type': 'String', 'config': config}
        node = schema.String.from_dict(node_dict)
        for attr, value in expected.items():
            assert getattr(node, attr) == value

    def test_to_dict(self):
        node = schema.String(min_length=3, max_length=5)
        node_dict = node.to_dict()
        assert 'node_type' in node_dict
        assert node_dict['node_type'] == 'String'
        assert 'config' in node_dict
        assert 'min_length' in node_dict['config']
        assert 'max_length' in node_dict['config']
        assert node_dict['config']['min_length'] == 3
        assert node_dict['config']['max_length'] == 5

    @pytest.mark.parametrize('config', (
        {}, {'min_length': 3}, {'max_length': 5},
        {'min_length': 3, 'max_length': 5}
    ))
    def test_validate(self, config):
        node_dict = {'node_type': 'String', 'config': config}
        node = schema.String.from_dict(node_dict)
        node.validate('spam')

    @pytest.mark.parametrize('test_data', (0, [23, 42], True, b'spam'))
    def test_validate_fail_type(self, test_data):
        node = schema.String()
        with pytest.raises(schema.ValidationError) as err:
            node.validate(test_data)
        assert err.value.message == 'Invalid type/value.'

    def test_validate_fail_max_length(self):
        node = schema.String(max_length=3)
        with pytest.raises(schema.ValidationError) as err:
            node.validate('abcd')
        assert err.value.message == 'Maximum string length exceeded.'
        assert err.value.expected == 3
        assert err.value.got == 4

    def test_validate_fail_min_length(self):
        node = schema.String(min_length=3)
        with pytest.raises(schema.ValidationError) as err:
            node.validate('ab')
        assert err.value.message == 'Minimum string length undercut.'
        assert err.value.expected == 3
        assert err.value.got == 2


def test_validation_error():
    ve = schema.ValidationError('Error message.', 'foo', 'baz')
    assert ve.message == 'Error message.'
    assert ve.expected == 'foo'
    assert ve.got == 'baz'

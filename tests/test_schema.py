import datetime
import numpy as np
import pytest
from dsch import schema


class TestArray:
    def test_from_dict(self):
        node = schema.Array.from_dict({'node_type': 'Array', 'config': {
            'dtype': 'float',
            'unit': 'V',
            'max_shape': (3, 2),
            'min_shape': (1, 1),
            'ndim': 2,
            'max_value': 42,
            'min_value': 23,
        }})
        assert isinstance(node, schema.Array)
        assert node.dtype == 'float'
        assert node.unit == 'V'
        assert node.max_shape == (3, 2)
        assert node.min_shape == (1, 1)
        assert node.ndim == 2
        assert node.max_value == 42
        assert node.min_value == 23

    def test_from_dict_fail(self):
        with pytest.raises(ValueError) as err:
            schema.Array.from_dict({'node_type': 'SPAM', 'config': {}})
        assert err.value.args[0] == 'Invalid node type in dict.'

    def test_init_defaults(self):
        node = schema.Array(dtype='int')
        assert node.dtype == 'int'
        assert node.unit == ''
        assert node.ndim == 1
        assert node.max_shape is None
        assert node.min_shape is None
        assert node.max_value is None
        assert node.min_value is None

    def test_init_defaults_ndim(self):
        node = schema.Array(dtype='int', ndim=2)
        assert node.dtype == 'int'
        assert node.unit == ''
        assert node.ndim == 2
        assert node.max_shape is None
        assert node.min_shape is None
        assert node.max_value is None
        assert node.min_value is None

    def test_init_defaults_max_shape(self):
        node = schema.Array(dtype='int', max_shape=(5, 1))
        assert node.dtype == 'int'
        assert node.unit == ''
        assert node.ndim == 2
        assert node.max_shape == (5, 1)
        assert node.min_shape is None
        assert node.max_value is None
        assert node.min_value is None

    def test_init_defaults_min_shape(self):
        node = schema.Array(dtype='int', min_shape=(2, 1))
        assert node.dtype == 'int'
        assert node.unit == ''
        assert node.ndim == 2
        assert node.max_shape is None
        assert node.min_shape == (2, 1)
        assert node.max_value is None
        assert node.min_value is None

    def test_init_fail_shapes(self):
        with pytest.raises(ValueError) as err:
            schema.Array(dtype='int', max_shape=(3, 2), min_shape=(2,))
        assert err.value.args[0] == ('Shape constraints must have the same '
                                     'length.')

    def test_to_dict(self):
        node = schema.Array(dtype='int', unit='V', max_shape=(3, 2),
                            min_shape=(1, 1), max_value=42, min_value=23)
        node_dict = node.to_dict()
        assert 'node_type' in node_dict
        assert node_dict['node_type'] == 'Array'
        assert 'config' in node_dict
        assert node_dict['config'] == {
            'dtype': 'int',
            'unit': 'V',
            'max_shape': (3, 2),
            'min_shape': (1, 1),
            'ndim': 2,
            'max_value': 42,
            'min_value': 23,
        }

    def test_validate(self):
        node = schema.Array(dtype='int', max_shape=(3,), min_shape=(2,),
                            max_value=42, min_value=23)
        node.validate(np.array([23, 42], dtype='int'))

    def test_validate_fail_dtype(self):
        node = schema.Array(dtype='int8')
        with pytest.raises(schema.ValidationError) as err:
            node.validate(np.array([23., 42]))
        assert err.value.message == 'Invalid dtype.'
        assert err.value.expected == 'int8'
        assert err.value.got == 'float'

    def test_validate_fail_max_shape(self):
        node = schema.Array(dtype='int', max_shape=(3,))
        with pytest.raises(schema.ValidationError) as err:
            node.validate(np.array([1, 2, 3, 4]))
        assert err.value.message == 'Maximum array shape exceeded.'
        assert err.value.expected == (3,)
        assert err.value.got == (4,)

    def test_validate_fail_min_shape(self):
        node = schema.Array(dtype='int', min_shape=(3, 1))
        with pytest.raises(schema.ValidationError) as err:
            node.validate(np.array([[1, 2], [3, 4]]))
        assert err.value.message == 'Minimum array shape undercut.'
        assert err.value.expected == (3, 1)
        assert err.value.got == (2, 2)

    def test_validate_fail_ndim(self):
        node = schema.Array(dtype='int', ndim=1)
        with pytest.raises(schema.ValidationError) as err:
            node.validate(np.array([[1, 2], [3, 4]]))
        assert err.value.message == 'Invalid number of array dimensions.'
        assert err.value.expected == 1
        assert err.value.got == 2

    def test_validate_fail_max_value(self):
        node = schema.Array(dtype='int', max_value=42)
        with pytest.raises(schema.ValidationError) as err:
            node.validate(np.array([23, 43]))
        assert err.value.message == 'Maximum array element value exceeded.'
        assert err.value.expected == 42
        assert err.value.got == np.array([43])

    def test_validate_fail_min_value(self):
        node = schema.Array(dtype='int', min_value=23)
        with pytest.raises(schema.ValidationError) as err:
            node.validate(np.array([22, 42]))
        assert err.value.message == 'Minimum array element value undercut.'
        assert err.value.expected == 23
        assert err.value.got == np.array([22])

    @pytest.mark.parametrize('test_data', (0, 1, [23, 42], 'spam'))
    def test_validate_fail_type(self, test_data):
        node = schema.Array(dtype='int')
        with pytest.raises(schema.ValidationError) as err:
            node.validate(test_data)
        assert err.value.message == 'Invalid type/value.'
        assert err.value.expected == 'numpy.ndarray'


class TestBool:
    def test_from_dict(self):
        node = schema.Bool.from_dict({'node_type': 'Bool', 'config': {}})
        assert isinstance(node, schema.Bool)

    def test_from_dict_fail(self):
        with pytest.raises(ValueError) as err:
            schema.Bool.from_dict({'node_type': 'SPAM', 'config': {}})
        assert err.value.args[0] == 'Invalid node type in dict.'

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

    def test_from_dict_fail(self):
        with pytest.raises(ValueError) as err:
            schema.Compilation.from_dict({'node_type': 'SPAM', 'config': {}})
        assert err.value.args[0] == 'Invalid node type in dict.'

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


class TestDate:
    def test_from_dict(self):
        node = schema.Date.from_dict({'node_type': 'Date', 'config':
                                      {'set_on_create': True}})
        assert isinstance(node, schema.Date)
        assert node.set_on_create

    def test_from_dict_fail(self):
        with pytest.raises(ValueError) as err:
            schema.Date.from_dict({'node_type': 'SPAM', 'config': {}})
        assert err.value.args[0] == 'Invalid node type in dict.'

    def test_to_dict(self):
        node = schema.Date()
        node_dict = node.to_dict()
        assert 'node_type' in node_dict
        assert node_dict['node_type'] == 'Date'
        assert 'config' in node_dict
        assert node_dict['config'] == {'set_on_create': False}

    def test_validate(self):
        node = schema.Date()
        node.validate(datetime.date.today())

    @pytest.mark.parametrize('test_data', (0, 1, [23, 42], 'spam',
                                           np.array([True])))
    def test_validate_fail(self, test_data):
        node = schema.Date()
        with pytest.raises(schema.ValidationError):
            node.validate(test_data)


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

    def test_from_dict_fail(self):
        with pytest.raises(ValueError) as err:
            schema.List.from_dict({'node_type': 'SPAM', 'config': {}})
        assert err.value.args[0] == 'Invalid node type in dict.'

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

    def test_from_dict_fail(self):
        with pytest.raises(ValueError) as err:
            schema.String.from_dict({'node_type': 'SPAM', 'config': {}})
        assert err.value.args[0] == 'Invalid node type in dict.'

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


class TestTime:
    def test_from_dict(self):
        node = schema.Time.from_dict({'node_type': 'Time', 'config':
                                      {'set_on_create': True}})
        assert isinstance(node, schema.Time)
        assert node.set_on_create

    def test_from_dict_fail(self):
        with pytest.raises(ValueError) as err:
            schema.Time.from_dict({'node_type': 'SPAM', 'config': {}})
        assert err.value.args[0] == 'Invalid node type in dict.'

    def test_to_dict(self):
        node = schema.Time()
        node_dict = node.to_dict()
        assert 'node_type' in node_dict
        assert node_dict['node_type'] == 'Time'
        assert 'config' in node_dict
        assert node_dict['config'] == {'set_on_create': False}

    def test_validate(self):
        node = schema.Time()
        node.validate(datetime.time(13, 37, 42))

    @pytest.mark.parametrize('test_data', (0, 1, [23, 42], 'spam',
                                           np.array([True])))
    def test_validate_fail(self, test_data):
        node = schema.Time()
        with pytest.raises(schema.ValidationError):
            node.validate(test_data)


def test_validation_error():
    ve = schema.ValidationError('Error message.', 'foo', 'baz')
    assert ve.message == 'Error message.'
    assert ve.expected == 'foo'
    assert ve.got == 'baz'

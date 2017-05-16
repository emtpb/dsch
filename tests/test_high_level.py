import datetime
import numpy as np
import pytest
import dsch


# Ensure dsch.schema is automatically imported alongside the dsch package.
# Normally, we would get the "schema" shorthand via "from dsch import schema".
schema = dsch.schema


@pytest.fixture(params=('hdf5', 'mat', 'npz'))
def storage_path(request, tmpdir):
    if request.param == 'hdf5':
        storage_path = str(tmpdir.join('test_file.h5'))
    elif request.param == 'mat':
        storage_path = str(tmpdir.join('test_file.mat'))
    elif request.param == 'npz':
        storage_path = str(tmpdir.join('test_file.npz'))
    return storage_path


example_values1 = {
    schema.Array: np.array([23, 42]),
    schema.Bool: True,
    schema.Date: datetime.date(1970, 1, 1),
    schema.DateTime: datetime.datetime(1970, 1, 1, 13, 37, 42, 23),
    schema.Scalar: np.int32(42),
    schema.String: 'spam',
    schema.Time: datetime.time(13, 37, 42, 23),
}
example_values2 = {
    schema.Array: np.array([1, 2, 3]),
    schema.Bool: False,
    schema.Date: datetime.date(1984, 5, 23),
    schema.DateTime: datetime.datetime(1984, 5, 23, 1, 2, 3, 4),
    schema.Scalar: np.int32(23),
    schema.String: 'eggs',
    schema.Time: datetime.time(1, 2, 3, 4),
}


@pytest.mark.parametrize('schema_node', (
    schema.Array(dtype='int'),
    schema.Bool(),
    schema.Date(),
    schema.DateTime(),
    schema.Scalar(dtype='int32'),
    schema.String(),
    schema.Time(),
))
def test_item_node(storage_path, schema_node):
    storage = dsch.create(storage_path=storage_path,
                          schema_node=schema_node)
    storage.data.value = example_values1[type(schema_node)]
    storage.data.validate()
    storage.save()

    new_storage = dsch.load(storage_path)
    assert np.all(new_storage.data.value ==
                  example_values1[type(schema_node)])

    storage.data.value = example_values2[type(schema_node)]
    new_storage.save()


def apply_example_values(data_node, example_values):
    if isinstance(data_node.schema_node, schema.Compilation):
        for subnode_name in data_node.schema_node.subnodes:
            apply_example_values(getattr(data_node, subnode_name),
                                 example_values)
    elif isinstance(data_node.schema_node, schema.List):
        data_node.append()
        data_node.append()
        for item in data_node:
            apply_example_values(item, example_values)
    else:
        data_node.value = example_values[type(data_node.schema_node)]


def assert_example_values(data_node, example_values):
    if isinstance(data_node.schema_node, schema.Compilation):
        for subnode_name in data_node.schema_node.subnodes:
            assert_example_values(getattr(data_node, subnode_name),
                                  example_values)
    elif isinstance(data_node.schema_node, schema.List):
        assert len(data_node) == 2
        for item in data_node:
            assert_example_values(item, example_values)
    else:
        assert np.all(data_node.value ==
                      example_values[type(data_node.schema_node)])


def test_compilation(storage_path):
    schema_node = schema.Compilation({
        'test_array': schema.Array(dtype='int'),
        'test_bool': schema.Bool(),
        'test_date': schema.Date(),
        'test_datetime': schema.DateTime(),
        'test_scalar': schema.Scalar(dtype='int32'),
        'test_string': schema.String(),
        'test_time': schema.Time(),
        'test_comp': schema.Compilation({
            'comp_array': schema.Array(dtype='int'),
            'comp_bool': schema.Bool(),
            'comp_date': schema.Date(),
            'comp_datetime': schema.DateTime(),
            'comp_scalar': schema.Scalar(dtype='int32'),
            'comp_string': schema.String(),
            'comp_time': schema.Time(),
        }),
        'test_list_array': schema.List(schema.Array(dtype='int')),
        'test_list_bool': schema.List(schema.Bool()),
        'test_list_date': schema.List(schema.Date()),
        'test_list_datetime': schema.List(schema.DateTime()),
        'test_list_scalar': schema.List(schema.Scalar(dtype='int32')),
        'test_list_string': schema.List(schema.String()),
        'test_list_time': schema.List(schema.Time()),
        'test_complist': schema.List(schema.Compilation({
            'complist_array': schema.Array(dtype='int'),
            'complist_bool': schema.Bool(),
            'complist_date': schema.Date(),
            'complist_datetime': schema.DateTime(),
            'complist_scalar': schema.Scalar(dtype='int32'),
            'complist_string': schema.String(),
            'complist_time': schema.Time(),
        })),
        'test_listlist': schema.List(schema.List(schema.Bool())),
    })

    storage = dsch.create(storage_path=storage_path,
                          schema_node=schema_node)

    apply_example_values(storage.data, example_values1)
    storage.data.validate()
    storage.save()

    new_storage = dsch.load(storage_path)
    assert_example_values(new_storage.data, example_values1)

    apply_example_values(new_storage.data, example_values2)
    new_storage.save()


def test_list(storage_path):
    schema_node = schema.List(
        schema.Compilation({
            'test_array': schema.Array(dtype='int'),
            'test_bool': schema.Bool(),
            'test_date': schema.Date(),
            'test_datetime': schema.DateTime(),
            'test_scalar': schema.Scalar(dtype='int32'),
            'test_string': schema.String(),
            'test_time': schema.Time(),
            'test_comp': schema.Compilation({
                'comp_array': schema.Array(dtype='int'),
                'comp_bool': schema.Bool(),
                'comp_date': schema.Date(),
                'comp_datetime': schema.DateTime(),
                'comp_scalar': schema.Scalar(dtype='int32'),
                'comp_string': schema.String(),
                'comp_time': schema.Time(),
            }),
            'test_list_array': schema.List(schema.Array(dtype='int')),
            'test_list_bool': schema.List(schema.Bool()),
            'test_list_date': schema.List(schema.Date()),
            'test_list_datetime': schema.List(schema.DateTime()),
            'test_list_scalar': schema.List(schema.Scalar(dtype='int32')),
            'test_list_string': schema.List(schema.String()),
            'test_list_time': schema.List(schema.Time()),
            'test_listlist': schema.List(schema.List(schema.Bool())),
        }))

    storage = dsch.create(storage_path=storage_path,
                          schema_node=schema_node)

    apply_example_values(storage.data, example_values1)
    storage.data.validate()
    storage.save()

    new_storage = dsch.load(storage_path)
    assert_example_values(new_storage.data, example_values1)

    apply_example_values(new_storage.data, example_values2)
    new_storage.save()

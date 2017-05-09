import datetime
import numpy as np
import pytest
from dsch import frontend, schema


@pytest.fixture(params=('hdf5', 'mat', 'npz'))
def storage_path(request, tmpdir):
    if request.param == 'hdf5':
        storage_path = str(tmpdir.join('test_file.h5'))
    elif request.param == 'mat':
        storage_path = str(tmpdir.join('test_file.mat'))
    elif request.param == 'npz':
        storage_path = str(tmpdir.join('test_file.npz'))
    return storage_path


@pytest.mark.parametrize('schema_node,valid_data,valid_data2', (
    (schema.Array(dtype='int'), np.array([23, 42]), np.array([1, 2, 3])),
    (schema.Bool(), True, False),
    (schema.Date(), datetime.date.today(), datetime.date(1970, 1, 1)),
    (schema.DateTime(), datetime.datetime.now(),
     datetime.datetime(1970, 1, 1, 0, 0, 0, 0)),
    (schema.Scalar(dtype='int32'), np.int32(42), np.int32(23)),
    (schema.String(), 'spam', 'eggs'),
    (schema.Time(), datetime.datetime.now().time(),
     datetime.time(13, 37, 42, 23)),
))
def test_item_node(storage_path, schema_node, valid_data, valid_data2):
    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)
    storage.data.replace(valid_data)
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    assert np.all(new_storage.data.value == valid_data)

    new_storage.data.replace(valid_data2)
    new_storage.save()


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

    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)

    storage.data.test_array.replace(np.array([23, 42]))
    storage.data.test_bool.replace(True)
    storage.data.test_date.replace(datetime.date(1985, 5, 23))
    storage.data.test_datetime.replace(datetime.datetime(1985, 5, 23, 5, 23,
                                                         42))
    storage.data.test_scalar.replace(42)
    storage.data.test_string.replace('spam')
    storage.data.test_time.replace(datetime.time(5, 23, 42))
    storage.data.test_comp.comp_array.replace(np.array([23, 42]))
    storage.data.test_comp.comp_bool.replace(True)
    storage.data.test_comp.comp_date.replace(datetime.date(1985, 5, 23))
    storage.data.test_comp.comp_datetime.replace(datetime.datetime(
        1985, 5, 23, 5, 23, 42))
    storage.data.test_comp.comp_scalar.replace(42)
    storage.data.test_comp.comp_string.replace('eggs')
    storage.data.test_comp.comp_time.replace(datetime.time(5, 23, 42))
    storage.data.test_list_array.replace([np.array([23, 42]),
                                          np.array([1, 0])])
    storage.data.test_list_bool.replace([True, False])
    storage.data.test_list_date.replace([datetime.date(1985, 5, 23),
                                         datetime.date(1970, 1, 1)])
    storage.data.test_list_datetime.replace([
        datetime.datetime(1985, 5, 23, 5, 23, 42),
        datetime.datetime(1970, 1, 1, 0, 0, 0)
    ])
    storage.data.test_list_scalar.replace([23, 42])
    storage.data.test_list_string.replace(['spam', 'eggs'])
    storage.data.test_list_time.replace([datetime.time(5, 23, 42),
                                         datetime.time(0, 0, 0)])
    storage.data.test_complist.replace([
        {
            'complist_array': np.array([23, 42]),
            'complist_bool': True,
            'complist_date': datetime.date(1985, 5, 23),
            'complist_datetime': datetime.datetime(1985, 5, 23, 5, 23, 42),
            'complist_scalar': 23,
            'complist_string': 'spam',
            'complist_time': datetime.time(5, 23, 42),
        },
        {
            'complist_array': np.array([1, 0]),
            'complist_bool': False,
            'complist_date': datetime.date(1970, 1, 1),
            'complist_datetime': datetime.datetime(1970, 1, 1, 0, 0, 0),
            'complist_scalar': 42,
            'complist_string': 'eggs',
            'complist_time': datetime.time(0, 0, 0),
        },
    ])
    storage.data.test_listlist.replace([[True, False], [False, True]])
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    data = new_storage.data
    assert np.all(data.test_array.value == np.array([23, 42]))
    assert data.test_bool.value is True
    assert data.test_date.value == datetime.date(1985, 5, 23)
    assert data.test_datetime.value == datetime.datetime(1985, 5, 23, 5,
                                                         23, 42)
    assert data.test_scalar.value == 42
    assert data.test_string.value == 'spam'
    assert data.test_time.value == datetime.time(5, 23, 42)
    assert np.all(data.test_comp.comp_array.value == np.array([23, 42]))
    assert data.test_comp.comp_bool.value is True
    assert data.test_comp.comp_date.value == datetime.date(1985, 5, 23)
    assert data.test_comp.comp_datetime.value == datetime.datetime(
        1985, 5, 23, 5, 23, 42)
    assert data.test_comp.comp_scalar.value == 42
    assert data.test_comp.comp_string.value == 'eggs'
    assert data.test_comp.comp_time.value == datetime.time(5, 23, 42)
    assert np.all(data.test_list_array[0].value == np.array([23, 42]))
    assert np.all(data.test_list_array[1].value == np.array([1, 0]))
    assert data.test_list_bool[0].value is True
    assert data.test_list_bool[1].value is False
    assert data.test_list_date[0].value == datetime.date(1985, 5, 23)
    assert data.test_list_date[1].value == datetime.date(1970, 1, 1)
    assert data.test_list_datetime[0].value == datetime.datetime(
        1985, 5, 23, 5, 23, 42)
    assert data.test_list_datetime[1].value == datetime.datetime(
        1970, 1, 1, 0, 0, 0)
    assert data.test_list_scalar[0].value == 23
    assert data.test_list_scalar[1].value == 42
    assert data.test_list_string[0].value == 'spam'
    assert data.test_list_string[1].value == 'eggs'
    assert data.test_list_time[0].value == datetime.time(5, 23, 42)
    assert data.test_list_time[1].value == datetime.time(0, 0, 0)
    assert np.all(data.test_complist[0].complist_array.value == np.array([23,
                                                                          42]))
    assert data.test_complist[0].complist_bool.value is True
    assert data.test_complist[0].complist_date.value == datetime.date(
        1985, 5, 23)
    assert data.test_complist[0].complist_datetime.value == datetime.datetime(
        1985, 5, 23, 5, 23, 42)
    assert data.test_complist[0].complist_scalar.value == 23
    assert data.test_complist[0].complist_string.value == 'spam'
    assert data.test_complist[0].complist_time.value == datetime.time(
        5, 23, 42)
    assert np.all(data.test_complist[1].complist_array.value == np.array([1,
                                                                          0]))
    assert data.test_complist[1].complist_bool.value is False
    assert data.test_complist[1].complist_date.value == datetime.date(
        1970, 1, 1)
    assert data.test_complist[1].complist_datetime.value == datetime.datetime(
        1970, 1, 1, 0, 0, 0)
    assert data.test_complist[1].complist_scalar.value == 42
    assert data.test_complist[1].complist_string.value == 'eggs'
    assert data.test_complist[1].complist_time.value == datetime.time(0, 0, 0)
    assert data.test_listlist[0][0].value is True
    assert data.test_listlist[0][1].value is False
    assert data.test_listlist[1][0].value is False
    assert data.test_listlist[1][1].value is True

    new_storage.data.test_array.replace(np.array([1, 2, 3]))
    new_storage.data.test_bool.replace(False)
    new_storage.data.test_date.replace(datetime.date(1970, 1, 1))
    new_storage.data.test_datetime.replace(datetime.datetime(1970, 1, 1, 0, 0, 0))
    new_storage.data.test_scalar.replace(23)
    new_storage.data.test_string.replace('eggs')
    new_storage.data.test_time.replace(datetime.time(0, 1, 2))
    new_storage.data.test_comp.comp_array.replace(np.array([1, 2, 3]))
    new_storage.data.test_comp.comp_bool.replace(False)
    new_storage.data.test_comp.comp_date.replace(datetime.date(1970, 1, 1))
    new_storage.data.test_comp.comp_datetime.replace(datetime.datetime(
        1970, 1, 1, 0, 0, 0))
    new_storage.data.test_comp.comp_scalar.replace(23)
    new_storage.data.test_comp.comp_string.replace('spam')
    new_storage.data.test_comp.comp_time.replace(datetime.time(0, 1, 2))
    new_storage.data.test_list_array.replace([np.array([1, 2]),
                                          np.array([3, 4])])
    new_storage.data.test_list_bool.replace([False, True])
    new_storage.data.test_list_date.replace([datetime.date(1970, 1, 1),
                                         datetime.date(1985, 5, 23)])
    new_storage.data.test_list_datetime.replace([
        datetime.datetime(1970, 1, 1, 0, 0, 0),
        datetime.datetime(1985, 5, 23, 5, 23, 42)
    ])
    new_storage.data.test_list_scalar.replace([1, 2])
    new_storage.data.test_list_string.replace(['ham', 'spam'])
    new_storage.data.test_list_time.replace([datetime.time(0, 0, 0),
                                         datetime.time(5, 23, 42)])
    new_storage.data.test_complist.replace([
        {
            'complist_array': np.array([1, 0]),
            'complist_bool': False,
            'complist_date': datetime.date(1970, 1, 1),
            'complist_datetime': datetime.datetime(1970, 1, 1, 0, 0, 0),
            'complist_scalar': 42,
            'complist_string': 'eggs',
            'complist_time': datetime.time(0, 0, 0),
        },
        {
            'complist_array': np.array([23, 42]),
            'complist_bool': True,
            'complist_date': datetime.date(1985, 5, 23),
            'complist_datetime': datetime.datetime(1985, 5, 23, 5, 23, 42),
            'complist_scalar': 23,
            'complist_string': 'spam',
            'complist_time': datetime.time(5, 23, 42),
        },
    ])
    new_storage.data.test_listlist.replace([[False, False], [True, True]])
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

    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)

    storage.data.append()
    storage.data[0].test_array.replace(np.array([23, 42]))
    storage.data[0].test_bool.replace(True)
    storage.data[0].test_date.replace(datetime.date(1985, 5, 23))
    storage.data[0].test_datetime.replace(datetime.datetime(1985, 5, 23, 5, 23,
                                                            42))
    storage.data[0].test_scalar.replace(23)
    storage.data[0].test_string.replace('spam')
    storage.data[0].test_time.replace(datetime.time(5, 23, 42))
    storage.data[0].test_comp.comp_array.replace(np.array([23, 42]))
    storage.data[0].test_comp.comp_bool.replace(True)
    storage.data[0].test_comp.comp_date.replace(datetime.date(1985, 5, 23))
    storage.data[0].test_comp.comp_datetime.replace(datetime.datetime(
        1985, 5, 23, 5, 23, 42))
    storage.data[0].test_comp.comp_scalar.replace(23)
    storage.data[0].test_comp.comp_string.replace('eggs')
    storage.data[0].test_comp.comp_time.replace(datetime.time(5, 23, 42))
    storage.data[0].test_list_array.replace([np.array([23, 42]),
                                             np.array([1, 0])])
    storage.data[0].test_list_bool.replace([True, False])
    storage.data[0].test_list_date.replace([datetime.date(1985, 5, 23),
                                            datetime.date(1970, 1, 1)])
    storage.data[0].test_list_datetime.replace([
        datetime.datetime(1985, 5, 23, 5, 23, 42),
        datetime.datetime(1970, 1, 1, 0, 0, 0)
    ])
    storage.data[0].test_list_scalar.replace([23, 42])
    storage.data[0].test_list_string.replace(['spam', 'eggs'])
    storage.data[0].test_list_time.replace([datetime.time(5, 23, 42),
                                            datetime.time(0, 0, 0)])
    storage.data[0].test_listlist.replace([[True, False], [False, True]])
    storage.data.append()
    storage.data[1].test_array.replace(np.array([23, 42]))
    storage.data[1].test_bool.replace(True)
    storage.data[1].test_date.replace(datetime.date(1985, 5, 23))
    storage.data[1].test_datetime.replace(datetime.datetime(1985, 5, 23, 5, 23,
                                                            42))
    storage.data[1].test_scalar.replace(23)
    storage.data[1].test_string.replace('spam')
    storage.data[1].test_time.replace(datetime.time(5, 23, 42))
    storage.data[1].test_comp.comp_array.replace(np.array([23, 42]))
    storage.data[1].test_comp.comp_bool.replace(True)
    storage.data[1].test_comp.comp_date.replace(datetime.date(1985, 5, 23))
    storage.data[1].test_comp.comp_datetime.replace(datetime.datetime(
        1985, 5, 23, 5, 23, 42))
    storage.data[1].test_comp.comp_scalar.replace(23)
    storage.data[1].test_comp.comp_string.replace('eggs')
    storage.data[1].test_comp.comp_time.replace(datetime.time(5, 23, 42))
    storage.data[1].test_list_array.replace([np.array([23, 42]),
                                             np.array([1, 0])])
    storage.data[1].test_list_bool.replace([True, False])
    storage.data[1].test_list_date.replace([datetime.date(1985, 5, 23),
                                            datetime.date(1970, 1, 1)])
    storage.data[1].test_list_datetime.replace([
        datetime.datetime(1985, 5, 23, 5, 23, 42),
        datetime.datetime(1970, 1, 1, 0, 0, 0)
    ])
    storage.data[1].test_list_scalar.replace([23, 42])
    storage.data[1].test_list_string.replace(['spam', 'eggs'])
    storage.data[1].test_list_time.replace([datetime.time(5, 23, 42),
                                            datetime.time(0, 0, 0)])
    storage.data[1].test_listlist.replace([[True, False], [False, True]])
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    for data in new_storage.data:
        assert np.all(data.test_array.value == np.array([23, 42]))
        assert data.test_bool.value is True
        assert data.test_date.value == datetime.date(1985, 5, 23)
        assert data.test_datetime.value == datetime.datetime(1985, 5, 23, 5,
                                                             23, 42)
        assert data.test_scalar.value == 23
        assert data.test_string.value == 'spam'
        assert data.test_time.value == datetime.time(5, 23, 42)
        assert np.all(data.test_comp.comp_array.value == np.array([23, 42]))
        assert data.test_comp.comp_bool.value is True
        assert data.test_comp.comp_date.value == datetime.date(1985, 5, 23)
        assert data.test_comp.comp_datetime.value == datetime.datetime(
            1985, 5, 23, 5, 23, 42)
        assert data.test_comp.comp_scalar.value == 23
        assert data.test_comp.comp_string.value == 'eggs'
        assert data.test_comp.comp_time.value == datetime.time(5, 23, 42)
        assert np.all(data.test_list_array[0].value == np.array([23, 42]))
        assert np.all(data.test_list_array[1].value == np.array([1, 0]))
        assert data.test_list_bool[0].value is True
        assert data.test_list_bool[1].value is False
        assert data.test_list_date[0].value == datetime.date(1985, 5, 23)
        assert data.test_list_date[1].value == datetime.date(1970, 1, 1)
        assert data.test_list_datetime[0].value == datetime.datetime(
            1985, 5, 23, 5, 23, 42)
        assert data.test_list_datetime[1].value == datetime.datetime(
            1970, 1, 1, 0, 0, 0)
        assert data.test_list_scalar[0].value == 23
        assert data.test_list_scalar[1].value == 42
        assert data.test_list_string[0].value == 'spam'
        assert data.test_list_string[1].value == 'eggs'
        assert data.test_list_time[0].value == datetime.time(5, 23, 42)
        assert data.test_list_time[1].value == datetime.time(0, 0, 0)
        assert data.test_listlist[0][0].value is True
        assert data.test_listlist[0][1].value is False
        assert data.test_listlist[1][0].value is False
        assert data.test_listlist[1][1].value is True

    new_storage.data[0].test_array.replace(np.array([1, 2]))
    new_storage.data[0].test_bool.replace(False)
    new_storage.data[0].test_date.replace(datetime.date(1970, 1, 1))
    new_storage.data[0].test_datetime.replace(datetime.datetime(1970, 1, 1, 0,
                                                                0, 0))
    new_storage.data[0].test_scalar.replace(42)
    new_storage.data[0].test_string.replace('eggs')
    new_storage.data[0].test_time.replace(datetime.time(0, 1, 2))
    new_storage.data[0].test_comp.comp_array.replace(np.array([1, 2]))
    new_storage.data[0].test_comp.comp_bool.replace(False)
    new_storage.data[0].test_comp.comp_date.replace(datetime.date(1970, 1, 1))
    new_storage.data[0].test_comp.comp_datetime.replace(datetime.datetime(
        1970, 1, 1, 0, 0, 0))
    new_storage.data[0].test_comp.comp_scalar.replace(42)
    new_storage.data[0].test_comp.comp_string.replace('ham')
    new_storage.data[0].test_comp.comp_time.replace(datetime.time(0, 1, 2))
    new_storage.data[0].test_list_array.replace([np.array([1, 2]),
                                                 np.array([3, 4])])
    new_storage.data[0].test_list_bool.replace([False, True])
    new_storage.data[0].test_list_date.replace([datetime.date(1970, 1, 1),
                                                datetime.date(1985, 5, 23)])
    new_storage.data[0].test_list_datetime.replace([
        datetime.datetime(1970, 1, 1, 0, 0, 0),
        datetime.datetime(1985, 5, 23, 5, 23, 42)
    ])
    new_storage.data[0].test_list_scalar.replace([1, 2])
    new_storage.data[0].test_list_string.replace(['ham', 'spam'])
    new_storage.data[0].test_list_time.replace([datetime.time(0, 0, 0),
                                                datetime.time(5, 23, 42)])
    new_storage.data[0].test_listlist.replace([[False, False], [True, True]])

    new_storage.data.append()
    new_storage.data[2].test_array.replace(np.array([23, 42]))
    new_storage.data[2].test_bool.replace(True)
    new_storage.data[2].test_date.replace(datetime.date(1985, 5, 23))
    new_storage.data[2].test_datetime.replace(datetime.datetime(1985, 5, 23, 5, 23,
                                                                42))
    new_storage.data[2].test_scalar.replace(23)
    new_storage.data[2].test_string.replace('spam')
    new_storage.data[2].test_time.replace(datetime.time(5, 23, 42))
    new_storage.data[2].test_comp.comp_array.replace(np.array([23, 42]))
    new_storage.data[2].test_comp.comp_bool.replace(True)
    new_storage.data[2].test_comp.comp_date.replace(datetime.date(1985, 5, 23))
    new_storage.data[2].test_comp.comp_datetime.replace(datetime.datetime(
        1985, 5, 23, 5, 23, 42))
    new_storage.data[2].test_comp.comp_scalar.replace(23)
    new_storage.data[2].test_comp.comp_string.replace('eggs')
    new_storage.data[2].test_comp.comp_time.replace(datetime.time(5, 23, 42))
    new_storage.data[2].test_list_array.replace([np.array([23, 42]),
                                                 np.array([1, 0])])
    new_storage.data[2].test_list_bool.replace([True, False])
    new_storage.data[2].test_list_date.replace([datetime.date(1985, 5, 23),
                                                datetime.date(1970, 1, 1)])
    new_storage.data[2].test_list_datetime.replace([
        datetime.datetime(1985, 5, 23, 5, 23, 42),
        datetime.datetime(1970, 1, 1, 0, 0, 0)
    ])
    new_storage.data[2].test_list_scalar.replace([23, 42])
    new_storage.data[2].test_list_string.replace(['spam', 'eggs'])
    new_storage.data[2].test_list_time.replace([datetime.time(5, 23, 42),
                                                datetime.time(0, 0, 0)])
    new_storage.data[2].test_listlist.replace([[True, False], [False, True]])
    new_storage.save()

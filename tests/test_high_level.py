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


def test_array(storage_path, tmpdir):
    schema_node = schema.Array(dtype='int')
    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)
    storage.data.replace(np.array([23, 42]))
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    data = new_storage.data
    assert np.all(data.value == np.array([23, 42]))


def test_bool(storage_path, tmpdir):
    schema_node = schema.Bool()
    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)
    storage.data.replace(True)
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    data = new_storage.data
    assert data.value is True


def test_compilation(storage_path, tmpdir):
    schema_node = schema.Compilation({
        'test_array': schema.Array(dtype='int'),
        'test_bool': schema.Bool(),
        'test_date': schema.Date(),
        'test_datetime': schema.DateTime(),
        'test_string': schema.String(),
        'test_time': schema.Time(),
        'test_comp': schema.Compilation({
            'comp_array': schema.Array(dtype='int'),
            'comp_bool': schema.Bool(),
            'comp_date': schema.Date(),
            'comp_datetime': schema.DateTime(),
            'comp_string': schema.String(),
            'comp_time': schema.Time(),
        }),
        'test_list_array': schema.List(schema.Array(dtype='int')),
        'test_list_bool': schema.List(schema.Bool()),
        'test_list_date': schema.List(schema.Date()),
        'test_list_datetime': schema.List(schema.DateTime()),
        'test_list_string': schema.List(schema.String()),
        'test_list_time': schema.List(schema.Time()),
        'test_complist': schema.List(schema.Compilation({
            'complist_array': schema.Array(dtype='int'),
            'complist_bool': schema.Bool(),
            'complist_date': schema.Date(),
            'complist_datetime': schema.DateTime(),
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
    storage.data.test_string.replace('spam')
    storage.data.test_time.replace(datetime.time(5, 23, 42))
    storage.data.test_comp.comp_array.replace(np.array([23, 42]))
    storage.data.test_comp.comp_bool.replace(True)
    storage.data.test_comp.comp_date.replace(datetime.date(1985, 5, 23))
    storage.data.test_comp.comp_datetime.replace(datetime.datetime(
        1985, 5, 23, 5, 23, 42))
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
    storage.data.test_list_string.replace(['spam', 'eggs'])
    storage.data.test_list_time.replace([datetime.time(5, 23, 42),
                                         datetime.time(0, 0, 0)])
    storage.data.test_complist.replace([
        {
            'complist_array': np.array([23, 42]),
            'complist_bool': True,
            'complist_date': datetime.date(1985, 5, 23),
            'complist_datetime': datetime.datetime(1985, 5, 23, 5, 23, 42),
            'complist_string': 'spam',
            'complist_time': datetime.time(5, 23, 42),
        },
        {
            'complist_array': np.array([1, 0]),
            'complist_bool': False,
            'complist_date': datetime.date(1970, 1, 1),
            'complist_datetime': datetime.datetime(1970, 1, 1, 0, 0, 0),
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
    assert data.test_string.value == 'spam'
    assert data.test_time.value == datetime.time(5, 23, 42)
    assert np.all(data.test_comp.comp_array.value == np.array([23, 42]))
    assert data.test_comp.comp_bool.value is True
    assert data.test_comp.comp_date.value == datetime.date(1985, 5, 23)
    assert data.test_comp.comp_datetime.value == datetime.datetime(
        1985, 5, 23, 5, 23, 42)
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
    assert data.test_complist[1].complist_string.value == 'eggs'
    assert data.test_complist[1].complist_time.value == datetime.time(0, 0, 0)
    assert data.test_listlist[0][0].value is True
    assert data.test_listlist[0][1].value is False
    assert data.test_listlist[1][0].value is False
    assert data.test_listlist[1][1].value is True


def test_date(storage_path, tmpdir):
    schema_node = schema.Date()
    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)
    dt = datetime.date.today()
    storage.data.replace(dt)
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    assert new_storage.data.value == dt


def test_datetime(storage_path, tmpdir):
    schema_node = schema.DateTime()
    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)
    dt = datetime.datetime.now()
    storage.data.replace(dt)
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    assert new_storage.data.value == dt


def test_list(storage_path, tmpdir):
    schema_node = schema.List(
        schema.Compilation({
            'test_array': schema.Array(dtype='int'),
            'test_bool': schema.Bool(),
            'test_date': schema.Date(),
            'test_datetime': schema.DateTime(),
            'test_string': schema.String(),
            'test_time': schema.Time(),
            'test_comp': schema.Compilation({
                'comp_array': schema.Array(dtype='int'),
                'comp_bool': schema.Bool(),
                'comp_date': schema.Date(),
                'comp_datetime': schema.DateTime(),
                'comp_string': schema.String(),
                'comp_time': schema.Time(),
            }),
            'test_list_array': schema.List(schema.Array(dtype='int')),
            'test_list_bool': schema.List(schema.Bool()),
            'test_list_date': schema.List(schema.Date()),
            'test_list_datetime': schema.List(schema.DateTime()),
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
    storage.data[0].test_string.replace('spam')
    storage.data[0].test_time.replace(datetime.time(5, 23, 42))
    storage.data[0].test_comp.comp_array.replace(np.array([23, 42]))
    storage.data[0].test_comp.comp_bool.replace(True)
    storage.data[0].test_comp.comp_date.replace(datetime.date(1985, 5, 23))
    storage.data[0].test_comp.comp_datetime.replace(datetime.datetime(
        1985, 5, 23, 5, 23, 42))
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
    storage.data[1].test_string.replace('spam')
    storage.data[1].test_time.replace(datetime.time(5, 23, 42))
    storage.data[1].test_comp.comp_array.replace(np.array([23, 42]))
    storage.data[1].test_comp.comp_bool.replace(True)
    storage.data[1].test_comp.comp_date.replace(datetime.date(1985, 5, 23))
    storage.data[1].test_comp.comp_datetime.replace(datetime.datetime(
        1985, 5, 23, 5, 23, 42))
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
        assert data.test_string.value == 'spam'
        assert data.test_time.value == datetime.time(5, 23, 42)
        assert np.all(data.test_comp.comp_array.value == np.array([23, 42]))
        assert data.test_comp.comp_bool.value is True
        assert data.test_comp.comp_date.value == datetime.date(1985, 5, 23)
        assert data.test_comp.comp_datetime.value == datetime.datetime(
            1985, 5, 23, 5, 23, 42)
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
        assert data.test_list_string[0].value == 'spam'
        assert data.test_list_string[1].value == 'eggs'
        assert data.test_list_time[0].value == datetime.time(5, 23, 42)
        assert data.test_list_time[1].value == datetime.time(0, 0, 0)
        assert data.test_listlist[0][0].value is True
        assert data.test_listlist[0][1].value is False
        assert data.test_listlist[1][0].value is False
        assert data.test_listlist[1][1].value is True


def test_string(storage_path, tmpdir):
    schema_node = schema.String()
    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)
    storage.data.replace('spam')
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    data = new_storage.data
    assert data.value == 'spam'


def test_time(storage_path, tmpdir):
    schema_node = schema.Time()
    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)
    dt = datetime.datetime.now().time()
    storage.data.replace(dt)
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    assert new_storage.data.value == dt

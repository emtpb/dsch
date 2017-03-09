import pytest
from dsch import frontend, schema


@pytest.fixture(params=('npz',))
def storage_path(request, tmpdir):
    if request.param == 'npz':
        storage_path = str(tmpdir.join('test_file.npz'))
    return storage_path


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
        'test_bool': schema.Bool(),
        'test_string': schema.String(),
        'test_comp': schema.Compilation({
            'comp_bool': schema.Bool(),
            'comp_string': schema.String()
        }),
        'test_list_bool': schema.List(schema.Bool()),
        'test_list_string': schema.List(schema.String()),
        'test_complist': schema.List(schema.Compilation({
            'complist_bool': schema.Bool(),
            'complist_string': schema.String()
        })),
        'test_listlist': schema.List(schema.List(schema.Bool())),
    })

    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)

    storage.data.test_bool.replace(True)
    storage.data.test_string.replace('spam')
    storage.data.test_comp.comp_bool.replace(True)
    storage.data.test_comp.comp_string.replace('eggs')
    storage.data.test_list_bool.replace([True, False])
    storage.data.test_list_string.replace(['spam', 'eggs'])
    storage.data.test_complist.replace([
        {'complist_bool': True, 'complist_string': 'spam'},
        {'complist_bool': False, 'complist_string': 'eggs'},
    ])
    storage.data.test_listlist.replace([[True, False], [False, True]])
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    data = new_storage.data
    assert data.test_bool.value is True
    assert data.test_string.value == 'spam'
    assert data.test_comp.comp_bool.value is True
    assert data.test_comp.comp_string.value == 'eggs'
    assert data.test_list_bool[0].value is True
    assert data.test_list_bool[1].value is False
    assert data.test_list_string[0].value == 'spam'
    assert data.test_list_string[1].value == 'eggs'
    assert data.test_complist[0].complist_bool.value is True
    assert data.test_complist[0].complist_string.value == 'spam'
    assert data.test_complist[1].complist_bool.value is False
    assert data.test_complist[1].complist_string.value == 'eggs'
    assert data.test_listlist[0][0].value is True
    assert data.test_listlist[0][1].value is False
    assert data.test_listlist[1][0].value is False
    assert data.test_listlist[1][1].value is True


def test_list(storage_path, tmpdir):
    schema_node = schema.List(
        schema.Compilation({
            'test_bool': schema.Bool(),
            'test_string': schema.String(),
            'test_comp': schema.Compilation({
                'comp_bool': schema.Bool(),
                'comp_string': schema.String()
            }),
            'test_list_bool': schema.List(schema.Bool()),
            'test_list_string': schema.List(schema.String()),
            'test_listlist': schema.List(schema.List(schema.Bool())),
        }))

    storage = frontend.create(storage_path=storage_path,
                              schema_node=schema_node)

    storage.data.append()
    storage.data[0].test_bool.replace(True)
    storage.data[0].test_string.replace('spam')
    storage.data[0].test_comp.comp_bool.replace(True)
    storage.data[0].test_comp.comp_string.replace('eggs')
    storage.data[0].test_list_bool.replace([True, False])
    storage.data[0].test_list_string.replace(['spam', 'eggs'])
    storage.data[0].test_listlist.replace([[True, False], [False, True]])
    storage.data.append()
    storage.data[1].test_bool.replace(True)
    storage.data[1].test_string.replace('spam')
    storage.data[1].test_comp.comp_bool.replace(True)
    storage.data[1].test_comp.comp_string.replace('eggs')
    storage.data[1].test_list_bool.replace([True, False])
    storage.data[1].test_list_string.replace(['spam', 'eggs'])
    storage.data[1].test_listlist.replace([[True, False], [False, True]])
    storage.data.validate()
    storage.save()

    new_storage = frontend.load(storage_path)
    for data in new_storage.data:
        assert data.test_bool.value is True
        assert data.test_string.value == 'spam'
        assert data.test_comp.comp_bool.value is True
        assert data.test_comp.comp_string.value == 'eggs'
        assert data.test_list_bool[0].value is True
        assert data.test_list_bool[1].value is False
        assert data.test_list_string[0].value == 'spam'
        assert data.test_list_string[1].value == 'eggs'
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

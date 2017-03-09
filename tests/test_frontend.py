from dsch import frontend, schema
from dsch.backends import npz


def test_create(tmpdir):
    schema_node = schema.Bool()
    storage_path = str(tmpdir.join('test_create.npz'))
    storage = frontend.create(storage_path, schema_node)
    assert isinstance(storage, npz.Storage)
    assert storage.schema_node == schema_node


def test_load(tmpdir):
    schema_node = schema.Bool()
    storage_path = str(tmpdir.join('test_load.npz'))
    storage = frontend.create(storage_path, schema_node)
    storage.data.replace(True)
    storage.save()

    new_storage = frontend.load(storage_path)
    assert isinstance(new_storage, npz.Storage)
    assert new_storage.data.value is True

from collections import namedtuple
import importlib
import itertools
import pytest
from dsch import exceptions, frontend, schema
from dsch.backends import inmem


backend_data = namedtuple('backend_data', ('module', 'storage_path'))


@pytest.fixture(params=('hdf5', 'mat', 'npz'))
def backend(request, tmpdir):
    backend = backend_data(
        module=importlib.import_module('dsch.backends.' + request.param),
        storage_path=str(tmpdir.join('test_frontend.' + request.param))
    )
    return backend


@pytest.fixture(params=('hdf5', 'inmem', 'mat', 'npz'))
def foreign_backend(request, tmpdir):
    if request.param == 'inmem':
        storage_path = '::inmem::'
    else:
        storage_path = str(tmpdir.join('test_frontend_foreign.' +
                                       request.param))
    backend = backend_data(
        module=importlib.import_module('dsch.backends.' + request.param),
        storage_path=storage_path
    )
    return backend


def test_create(backend):
    schema_node = schema.Bool()
    storage = frontend.create(backend.storage_path, schema_node)
    assert isinstance(storage, backend.module.Storage)
    assert storage.schema_node == schema_node


def test_create_inmem():
    schema_node = schema.Bool()
    storage = frontend.create('::inmem::', schema_node)
    assert isinstance(storage, inmem.Storage)
    assert storage.schema_node == schema_node


def test_create_from(backend, foreign_backend):
    schema_node = schema.Bool()
    source_storage = frontend.create(foreign_backend.storage_path, schema_node)
    source_storage.data.value = True

    dest_storage = frontend.create_from(backend.storage_path, source_storage)
    assert dest_storage.schema_node.hash() == source_storage.schema_node.hash()
    assert dest_storage.data.value is True


def test_load(backend):
    schema_node = schema.Bool()
    storage = frontend.create(backend.storage_path, schema_node)
    storage.data.value = True
    storage.save()

    new_storage = frontend.load(backend.storage_path)
    assert isinstance(new_storage, backend.module.Storage)
    assert new_storage.data.value is True


def test_load_require_schema(backend):
    schema_node = schema.Bool()
    storage = frontend.create(backend.storage_path, schema_node)
    storage.data.value = True
    storage.save()

    frontend.load(backend.storage_path, required_schema=schema_node)


def test_load_required_schema_hash(backend):
    schema_node = schema.Bool()
    storage = frontend.create(backend.storage_path, schema_node)
    storage.data.value = True
    storage.save()

    schema_hash = storage.schema_hash()
    frontend.load(backend.storage_path, required_schema_hash=schema_hash)


def test_load_validation_fail(backend):
    schema_node = schema.String(max_length=3)
    storage = frontend.create(backend.storage_path, schema_node)
    storage.data.value = 'spam'
    storage.save(force=True)

    with pytest.raises(exceptions.ValidationError):
        frontend.load(backend.storage_path)
    # With force=True, no exception must be raised.
    frontend.load(backend.storage_path, force=True)


class TestPseudoStorageNode:
    """Tests for the PseudoStorage class when given a data node."""

    @pytest.fixture
    def storage(self, foreign_backend):
        schema_node = schema.Compilation({
            'spam': schema.Bool(),
            'eggs': schema.Bytes(),
        })
        return frontend.create(foreign_backend.storage_path, schema_node)

    def test_context(self, storage):
        pseudo = frontend.PseudoStorage(storage.data.spam, schema.Bool(), True)
        with pseudo as p:
            assert p.data is not None
            assert p.storage is None
            assert pseudo.schema_node == storage.data.spam.schema_node
        assert pseudo.data is None
        assert pseudo.storage is None
        assert pseudo.schema_node is None

    def test_init(self, storage):
        pseudo = frontend.PseudoStorage(storage.data.spam, schema.Bool(), False)
        assert pseudo.data is not None
        assert pseudo.storage is None
        assert pseudo.schema_node is not None

    def test_init_deferred(self, storage):
        pseudo = frontend.PseudoStorage(storage.data.spam, schema.Bool(), True)
        assert pseudo.data is None
        assert pseudo.storage is None
        assert pseudo.schema_node is None

    def test_open_close(self, storage):
        pseudo = frontend.PseudoStorage(storage.data.spam, schema.Bool(), True)
        pseudo.open()
        assert pseudo.data is not None
        assert pseudo.storage is None
        assert pseudo.schema_node == storage.data.spam.schema_node
        pseudo.close()
        assert pseudo.data is None
        assert pseudo.storage is None
        assert pseudo.schema_node is None

    def test_open_fail(self, storage):
        pseudo = frontend.PseudoStorage(storage.data.spam, schema.Bytes(),
                                        True)
        with pytest.raises(exceptions.InvalidSchemaError):
            pseudo.open()

    def test_schema_alternative(self, storage):
        pseudo = frontend.PseudoStorage(storage.data.spam, schema.Bytes(),
                                        True, (schema.Bool(),))
        pseudo.open()
        assert pseudo.schema_node == storage.data.spam.schema_node

    def test_schema_alternative_fail(self, storage):
        pseudo = frontend.PseudoStorage(storage.data.spam, schema.Bytes(),
                                        True, (schema.String(),))
        with pytest.raises(exceptions.InvalidSchemaError):
            pseudo.open()


class TestPseudoStorageStr:
    """Tests for the PseudoStorage class when given a string."""

    @pytest.fixture(params=tuple(
        (b, e) for b, e in itertools.product(
            ('hdf5', '::inmem::', 'npz', 'mat'), (False, True)
        ) if not (b == '::inmem::' and e)
    ))
    def storage_path(self, request, tmpdir):
        """Prepare storage path for all valid variants.

        Variants are tuples ``(backend, existing)``, where ``existing``
        indicates whether a storage should be created before the test.
        """
        backend, existing = request.param
        if backend in ('hdf5', 'npz', 'mat'):
            # File backends
            storage_path = str(tmpdir.join('test_pseudo.' + backend))
        elif backend == '::inmem::':
            storage_path = '::inmem::'
        if existing:
            storage = frontend.create(storage_path, schema.Bool())
            storage.data.value = True
            if hasattr(storage, 'save') and callable(storage.save):
                storage.save()
            del storage
        return storage_path

    @pytest.fixture(params=('hdf5', 'npz', 'mat'))
    def storage_path_existing(self, request, tmpdir):
        storage_path = str(tmpdir.join('test_pseudo.' + request.param))
        storage = frontend.create(storage_path, schema.Bool())
        storage.data.value = True
        storage.save()
        del storage
        return storage_path

    def test_context(self, storage_path):
        pseudo = frontend.PseudoStorage(storage_path, schema.Bool(), True)
        with pseudo as p:
            assert p.data is not None
            assert p.storage is not None
            assert p.schema_node == p.storage.data.schema_node
        assert pseudo.data is None
        assert pseudo.storage is None
        assert pseudo.schema_node is None

    def test_init(self, storage_path):
        pseudo = frontend.PseudoStorage(storage_path, schema.Bool(), False)
        assert pseudo.data is not None
        assert pseudo.storage is not None
        assert pseudo.storage.storage_path == storage_path
        assert pseudo.schema_node is not None

    def test_init_deferred(self, storage_path):
        pseudo = frontend.PseudoStorage(storage_path, schema.Bool(), True)
        assert pseudo.data is None
        assert pseudo.storage is None
        assert pseudo.schema_node is None

    def test_open_close(self, storage_path):
        pseudo = frontend.PseudoStorage(storage_path, schema.Bool(), True)
        pseudo.open()
        assert pseudo.data is not None
        assert pseudo.storage is not None
        assert pseudo.schema_node == pseudo.storage.data.schema_node
        pseudo.close()
        assert pseudo.data is None
        assert pseudo.storage is None
        assert pseudo.schema_node is None

    def test_open_fail(self, storage_path_existing):
        pseudo = frontend.PseudoStorage(storage_path_existing, schema.Bytes(), True)
        with pytest.raises(exceptions.InvalidSchemaError):
            pseudo.open()

    def test_schema_alternative(self, storage_path_existing):
        pseudo = frontend.PseudoStorage(storage_path_existing, schema.Bytes(),
                                        True, (schema.Bool(),))
        pseudo.open()
        assert pseudo.schema_node == pseudo.storage.data.schema_node

    def test_schema_alternative_fail(self, storage_path_existing):
        pseudo = frontend.PseudoStorage(storage_path_existing, schema.Bytes(),
                                        True, (schema.String(),))
        with pytest.raises(exceptions.InvalidSchemaError):
            pseudo.open()

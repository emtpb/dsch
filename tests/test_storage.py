from dsch import schema, storage


def test_schema_hash():
    schema_node = schema.Bool()
    storage_obj = storage.Storage('', schema_node)
    assert storage_obj.schema_hash() == ('45d0233242870dd39f632cb5dd78704b'
                                         '901db11b9483de5bcc6489b1d3b76235')


def test_schema_from_json():
    storage_obj = storage.Storage('')
    storage_obj._schema_from_json('{"config": {}, "node_type": "Bool"}')
    assert isinstance(storage_obj.schema_node, schema.Bool)


def test_schema_to_json():
    storage_obj = storage.Storage('', schema.Bool())
    json_data = storage_obj._schema_to_json()
    assert json_data == '{"config": {}, "node_type": "Bool"}'

from dsch import schema, storage


def test_schema_from_json():
    storage_obj = storage.Storage('')
    storage_obj._schema_from_json('{"config": {}, "node_type": "Bool"}')
    assert isinstance(storage_obj.schema_node, schema.Bool)


def test_schema_to_json():
    storage_obj = storage.Storage('', schema.Bool())
    json_data = storage_obj._schema_to_json()
    assert json_data == '{"config": {}, "node_type": "Bool"}'

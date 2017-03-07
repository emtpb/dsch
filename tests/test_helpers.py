from dsch import helpers


def test_inflate_dotted():
    input = {
        'foo.foo.foo': 23,
        'foo.bar.foo': 42,
        'foo.bar.baz': 1337,
        'baz.bar.foo': 9000
    }
    output = helpers.inflate_dotted(input)
    assert output == {
        'foo': {'foo': {'foo': 23}, 'bar': {'foo': 42, 'baz': 1337}},
        'baz': {'bar': {'foo': 9000}}
    }


def test_flatten_dotted():
    input = {
        'foo': {'foo': {'foo': 23}, 'bar': {'foo': 42, 'baz': 1337}},
        'baz': {'bar': {'foo': 9000}}
    }
    output = helpers.flatten_dotted(input)
    assert output == {
        'foo.foo.foo': 23,
        'foo.bar.foo': 42,
        'foo.bar.baz': 1337,
        'baz.bar.foo': 9000
    }

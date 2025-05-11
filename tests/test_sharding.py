import pytest

from wdbx.sharding import ShardManager


def test_shard_manager_consistency():
    nodes = ['nodeA', 'nodeB', 'nodeC']
    manager = ShardManager(nodes, replicas=10)
    # Same key should map to the same node consistently
    key = 'my_key'
    node1 = manager.get_node(key)
    node2 = manager.get_node(key)
    assert node1 == node2
    # Key should map to one of the provided nodes
    assert node1 in nodes


def test_shard_manager_empty_ring():
    manager = ShardManager([], replicas=5)
    with pytest.raises(ValueError):
        manager.get_node('any')


def test_shard_distribution_uniformity():
    # Test that keys distribute among nodes (not strictly uniform at low count)
    nodes = ['X', 'Y', 'Z']
    manager = ShardManager(nodes, replicas=5)
    mapping = {node: 0 for node in nodes}
    for i in range(100):
        key = f'key_{i}'
        node = manager.get_node(key)
        mapping[node] += 1
    # Ensure each node got at least one key
    for count in mapping.values():
        assert count > 0 
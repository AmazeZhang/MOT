import json


def test_action_difference_mask_is_symmetric_difference(tmp_path):
    # Contract-level example used by the exporter/model input.
    union = [(0, 0), (0, 1), (1, 0)]
    m_edges = {(0, 0), (1, 1)}
    action_edges = {(0, 1), (1, 0)}
    values = [int((edge in action_edges) != (edge in m_edges)) for edge in union]
    assert values == [1, 1, 1]
    assert json.loads(json.dumps(values)) == values

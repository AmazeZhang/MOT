import torch

from mot_research.models.component_action_selector import (
    ComponentActionSelector, FeatureNormalization, parameter_count,
)


def inputs():
    return (torch.zeros(3, 8), torch.zeros(3, 2), torch.tensor([0, 0, 1]),
            torch.tensor([0, 1, 0]), torch.tensor([True, False, True]),
            torch.tensor([2.0, 2.0, 2.0]))


def test_local_and_context_are_parameter_matched_and_bounded():
    norm = FeatureNormalization((0.0,) * 8, (1.0,) * 8)
    local = ComponentActionSelector("Local", norm)
    context = ComponentActionSelector("Context", norm)
    assert parameter_count(local) == parameter_count(context)
    for model in (local, context):
        value = model(*inputs())
        assert value.shape == (2,)
        assert torch.all((value >= 0) & (value <= 1))


def test_zero_edge_action_is_supported():
    norm = FeatureNormalization((0.0,) * 8, (1.0,) * 8)
    args = list(inputs())
    args[4] = torch.tensor([False, False, False])
    assert ComponentActionSelector("Local", norm)(*args).shape == (2,)

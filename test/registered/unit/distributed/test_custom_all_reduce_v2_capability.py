from unittest.mock import Mock

import pytest

from sglang.srt.distributed.device_communicators import custom_all_reduce_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _patch_group(monkeypatch, *, world_size, same_node):
    group = object()
    device = object()
    monkeypatch.setattr(
        custom_all_reduce_v2.dist,
        "get_world_size",
        lambda group: world_size,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "get_supported_world_sizes",
        lambda: (world_size,),
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "in_the_same_node_as",
        lambda group, source_rank: [same_node] * world_size,
    )
    return group, device


@pytest.mark.parametrize(
    ("same_node", "has_fabric_clique", "uses_vmm", "expected"),
    [
        (False, True, True, True),
        (False, False, True, False),
        (False, True, False, False),
        (True, None, None, True),
    ],
)
def test_topology_capability(
    monkeypatch, same_node, has_fabric_clique, uses_vmm, expected
):
    monkeypatch.setattr(custom_all_reduce_v2, "_IS_HIP", False)
    world_size = 8 if same_node else 16
    group, device = _patch_group(
        monkeypatch,
        world_size=world_size,
        same_node=same_node,
    )

    def is_one_clique(group, device):
        if same_node:
            pytest.fail("intra-node groups do not need a fabric clique")
        return has_fabric_clique

    def is_vmm_backed(device):
        if same_node:
            pytest.fail("intra-node groups do not need VMM")
        return uses_vmm

    intra_node_capability = Mock(return_value=True)

    monkeypatch.setattr(
        custom_all_reduce_v2,
        "is_one_nvlink_clique",
        is_one_clique,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "_is_vmm_backed_allocator",
        is_vmm_backed,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "can_use_custom_all_reduce_with_nvlink",
        intra_node_capability,
    )

    assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is expected
    if same_node:
        intra_node_capability.assert_called_once_with(
            group=group,
            device=device,
            supported_world_size=[world_size],
            cls_name="CustomAllReduceV2",
        )
    else:
        intra_node_capability.assert_not_called()


@pytest.mark.parametrize(
    ("same_node", "is_gfx1250", "fabric_supported", "expected"),
    [
        (True, True, None, True),
        (True, False, None, False),
        (False, True, True, True),
        (False, True, False, False),
    ],
)
def test_rocm_topology_capability(
    monkeypatch, same_node, is_gfx1250, fabric_supported, expected
):
    world_size = 4
    group, device = _patch_group(
        monkeypatch,
        world_size=world_size,
        same_node=same_node,
    )
    intra_node_capability = Mock(return_value=True)
    monkeypatch.setattr(custom_all_reduce_v2, "_IS_HIP", True)
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "_is_gfx1250",
        lambda device: is_gfx1250,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "can_use_custom_all_reduce_with_nvlink",
        intra_node_capability,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "is_one_nvlink_clique",
        lambda group, device: pytest.fail(
            "ROCm must not enter the CUDA multi-node clique path"
        ),
    )
    fabric_capability = Mock()
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "probe_local_fabric_capability",
        lambda device, rank: fabric_capability,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "validate_fabric_clique",
        lambda capabilities: (
            fabric_supported,
            () if fabric_supported else ("fabric unavailable",),
        ),
    )
    monkeypatch.setattr(
        custom_all_reduce_v2.dist,
        "get_rank",
        lambda group: 0,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2.dist,
        "all_gather_object",
        lambda output, value, group: output.__setitem__(
            slice(None), [value] * len(output)
        ),
    )
    if not same_node:
        device = Mock(index=0)

    assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is expected
    intra_node_capability.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

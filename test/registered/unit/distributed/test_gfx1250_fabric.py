from pathlib import Path

from sglang.srt.distributed.device_communicators.gfx1250_fabric import (
    FabricCapability,
    _find_native_fabric_header,
    _find_ualink_device,
    _torch_pci_bus_id,
    validate_fabric_clique,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _capability(
    rank: int,
    *,
    ppod_id: str = "ppod",
    vpod_id: str = "3",
    reasons: tuple[str, ...] = (),
) -> FabricCapability:
    return FabricCapability(
        rank=rank,
        hostname=f"host-{rank // 4}",
        device=rank % 4,
        gcn_arch="gfx1250",
        pci_bus_id=f"0000:{rank:02x}:00.0",
        drm_card=f"card{rank}",
        link_type="UALoE",
        accel_state="active",
        ppod_id=ppod_id,
        ppod_size=72,
        vpod_id=vpod_id,
        vpod_size=8,
        header="/rocm/include/hip/hip_runtime_api.h",
        header_tokens=(
            "hipMemFabricHandle_t",
            "hipMemHandleTypeFabric",
            "hipDeviceAttributeHandleTypeFabricSupported",
            "hipMemImportFromShareableHandle",
        ),
        runtime_library="libamdhip64.so",
        runtime_symbols=(
            "hipMemCreate",
            "hipMemExportToShareableHandle",
            "hipMemImportFromShareableHandle",
            "hipMemAddressReserve",
            "hipMemMap",
            "hipMemSetAccess",
        ),
        device_attribute_supported=True,
        reasons=reasons,
    )


def test_find_native_fabric_header(tmp_path: Path):
    header = tmp_path / "hip_runtime_api.h"
    header.write_text(
        """
        typedef struct hipMemFabricHandle_t hipMemFabricHandle_t;
        hipError_t hipMemImportFromShareableHandle();
        hipMemHandleTypeFabric = 8,
        hipDeviceAttributeHandleTypeFabricSupported = 128,
        """,
        encoding="utf-8",
    )

    path, tokens, attribute = _find_native_fabric_header([header])

    assert path == header
    assert len(tokens) == 4
    assert attribute == 128


def test_find_ualink_device_by_pci_bus_id(tmp_path: Path):
    pci_device = tmp_path / "devices" / "0000:41:00.0"
    ualink = pci_device / "ualink"
    ualink.mkdir(parents=True)
    values = {
        "link_type": "UALoE",
        "accel_state": "active",
        "ppod_id": "uuid",
        "ppod_size": "72",
        "vpod_id": "3",
        "vpod_size": "8",
    }
    for name, value in values.items():
        (ualink / name).write_text(value, encoding="utf-8")
    card = tmp_path / "card0"
    card.mkdir()
    (card / "device").symlink_to(pci_device, target_is_directory=True)

    card_name, actual = _find_ualink_device("0000:41:00.0", tmp_path)

    assert card_name == "card0"
    assert actual == values


def test_torch_pci_bus_id():
    properties = type(
        "Properties",
        (),
        {"pci_domain_id": 1, "pci_bus_id": 4, "pci_device_id": 0},
    )()
    assert _torch_pci_bus_id(properties) == "0001:04:00.0"


def test_validate_fabric_clique():
    supported, reasons = validate_fabric_clique([_capability(0), _capability(1)])
    assert supported
    assert not reasons

    supported, reasons = validate_fabric_clique(
        [_capability(0), _capability(1, vpod_id="4")]
    )
    assert not supported
    assert any("VPOD" in reason for reason in reasons)


def test_validate_fabric_clique_propagates_rank_failure():
    supported, reasons = validate_fabric_clique(
        [_capability(0), _capability(1, reasons=("fabric API missing",))]
    )
    assert not supported
    assert "rank 1: fabric API missing" in reasons

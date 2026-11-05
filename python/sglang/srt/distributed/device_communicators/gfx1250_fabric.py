"""Capability detection for gfx1250 UALink/UALoE fabric VMM.

These checks only read sysfs, headers, and runtime symbols. They never
allocate, export or import handles, or map peer memory; ``HIPFabricPlane``
does that once every rank has passed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

_FABRIC_HEADER_TOKENS = (
    "hipMemFabricHandle_t",
    "hipMemHandleTypeFabric",
    "hipDeviceAttributeHandleTypeFabricSupported",
    "hipMemImportFromShareableHandle",
)
_FABRIC_RUNTIME_SYMBOLS = (
    "hipMemCreate",
    "hipMemExportToShareableHandle",
    "hipMemImportFromShareableHandle",
    "hipMemAddressReserve",
    "hipMemMap",
    "hipMemSetAccess",
)
_SUPPORTED_LINK_TYPES = {"ualink", "ualoe"}
_READY_STATES = {"active", "ready"}


@dataclass(frozen=True)
class FabricCapability:
    rank: int
    hostname: str
    device: int
    gcn_arch: str
    pci_bus_id: Optional[str]
    drm_card: Optional[str]
    link_type: Optional[str]
    accel_state: Optional[str]
    ppod_id: Optional[str]
    ppod_size: Optional[int]
    vpod_id: Optional[str]
    vpod_size: Optional[int]
    header: Optional[str]
    header_tokens: tuple[str, ...]
    runtime_library: Optional[str]
    runtime_symbols: tuple[str, ...]
    device_attribute_supported: Optional[bool]
    reasons: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.reasons


def _read_sysfs_value(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value, 0)
    except ValueError:
        return None


def _candidate_hip_headers() -> list[Path]:
    roots = []
    for name in ("ROCM_HOME", "ROCM_PATH"):
        if value := os.getenv(name):
            roots.append(Path(value))
    roots.extend(
        (
            Path("/opt/rocm"),
            Path(sys.prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages/_rocm_sdk_devel",
        )
    )
    headers = []
    for root in roots:
        candidate = root / "include/hip/hip_runtime_api.h"
        if candidate.is_file() and candidate not in headers:
            headers.append(candidate)
    return headers


def _find_native_fabric_header(
    headers: Optional[Iterable[Path]] = None,
) -> tuple[Optional[Path], tuple[str, ...], Optional[int]]:
    best_path = None
    best_tokens: tuple[str, ...] = ()
    attribute_value = None
    for header in headers if headers is not None else _candidate_hip_headers():
        try:
            text = header.read_text(encoding="utf-8")
        except OSError:
            continue
        tokens = tuple(token for token in _FABRIC_HEADER_TOKENS if token in text)
        if len(tokens) > len(best_tokens):
            best_path = header
            best_tokens = tokens
        match = re.search(
            r"hipDeviceAttributeHandleTypeFabricSupported\s*=\s*(\d+)", text
        )
        if match:
            attribute_value = int(match.group(1))
    return best_path, best_tokens, attribute_value


def _load_hip_runtime() -> tuple[Optional[ctypes.CDLL], Optional[str]]:
    candidates = []
    if library := ctypes.util.find_library("amdhip64"):
        candidates.append(library)
    sdk_root = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages/_rocm_sdk_devel"
    )
    for root in (
        os.getenv("ROCM_HOME"),
        os.getenv("ROCM_PATH"),
        "/opt/rocm",
        str(sdk_root),
    ):
        if root:
            candidates.append(str(Path(root) / "lib/libamdhip64.so"))
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError:
            continue
    return None, None


def _torch_pci_bus_id(properties) -> Optional[str]:
    required = ("pci_domain_id", "pci_bus_id", "pci_device_id")
    if not all(hasattr(properties, name) for name in required):
        return None
    return (
        f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
        f"{properties.pci_device_id:02x}.0"
    )


def _fabric_device_attribute(
    runtime: Optional[ctypes.CDLL], device: int, attribute: Optional[int]
) -> Optional[bool]:
    if (
        runtime is None
        or attribute is None
        or not hasattr(runtime, "hipDeviceGetAttribute")
    ):
        return None
    function = runtime.hipDeviceGetAttribute
    function.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int)
    function.restype = ctypes.c_int
    result = ctypes.c_int()
    if function(ctypes.byref(result), attribute, device) != 0:
        return None
    return bool(result.value)


def _find_ualink_device(
    pci_bus_id: Optional[str], sysfs_root: Path
) -> tuple[Optional[str], dict[str, Optional[str]]]:
    empty = {
        "link_type": None,
        "accel_state": None,
        "ppod_id": None,
        "ppod_size": None,
        "vpod_id": None,
        "vpod_size": None,
    }
    if pci_bus_id is None:
        return None, empty
    expected = pci_bus_id.removeprefix("0000:")
    for card in sorted(sysfs_root.glob("card[0-9]*")):
        device_path = card / "device"
        try:
            actual = device_path.resolve().name.removeprefix("0000:")
        except OSError:
            continue
        if actual.lower() != expected:
            continue
        ualink = device_path / "ualink"
        values = {
            name: _read_sysfs_value(ualink / name)
            for name in (
                "link_type",
                "accel_state",
                "ppod_id",
                "ppod_size",
                "vpod_id",
                "vpod_size",
            )
        }
        return card.name, values
    return None, empty


def probe_local_fabric_capability(
    device: Optional[int] = None,
    *,
    rank: int = 0,
    sysfs_root: Path = Path("/sys/class/drm"),
    headers: Optional[Iterable[Path]] = None,
) -> FabricCapability:
    device = torch.cuda.current_device() if device is None else device
    properties = torch.cuda.get_device_properties(device)
    gcn_arch = getattr(properties, "gcnArchName", "")
    runtime, runtime_library = _load_hip_runtime()
    pci_bus_id = _torch_pci_bus_id(properties)
    card, fabric = _find_ualink_device(pci_bus_id, sysfs_root)
    header, header_tokens, attribute = _find_native_fabric_header(headers)
    runtime_symbols = tuple(
        symbol
        for symbol in _FABRIC_RUNTIME_SYMBOLS
        if runtime is not None and hasattr(runtime, symbol)
    )
    device_attribute = _fabric_device_attribute(runtime, device, attribute)

    reasons = []
    if "gfx1250" not in gcn_arch:
        reasons.append(f"device architecture is {gcn_arch or 'unknown'}, not gfx1250")
    if card is None:
        reasons.append("no UALink sysfs record matches the device PCI bus ID")
    link_type = fabric["link_type"]
    if link_type is None or link_type.lower() not in _SUPPORTED_LINK_TYPES:
        reasons.append(f"unsupported fabric link type: {link_type!r}")
    accel_state = fabric["accel_state"]
    if accel_state is None or accel_state.lower() not in _READY_STATES:
        reasons.append(f"fabric state is not ACTIVE or READY: {accel_state!r}")
    if not fabric["ppod_id"]:
        reasons.append("fabric PPOD UUID is missing")
    if not fabric["vpod_id"]:
        reasons.append("fabric VPOD/clique ID is missing")
    missing_tokens = sorted(set(_FABRIC_HEADER_TOKENS) - set(header_tokens))
    if missing_tokens:
        reasons.append(f"HIP headers lack native fabric API: {missing_tokens}")
    missing_symbols = sorted(set(_FABRIC_RUNTIME_SYMBOLS) - set(runtime_symbols))
    if missing_symbols:
        reasons.append(f"HIP runtime lacks VMM symbols: {missing_symbols}")
    if device_attribute is False:
        reasons.append("hipDeviceAttributeHandleTypeFabricSupported is false")

    return FabricCapability(
        rank=rank,
        hostname=os.uname().nodename,
        device=device,
        gcn_arch=gcn_arch,
        pci_bus_id=pci_bus_id,
        drm_card=card,
        link_type=link_type,
        accel_state=accel_state,
        ppod_id=fabric["ppod_id"],
        ppod_size=_parse_int(fabric["ppod_size"]),
        vpod_id=fabric["vpod_id"],
        vpod_size=_parse_int(fabric["vpod_size"]),
        header=str(header) if header is not None else None,
        header_tokens=header_tokens,
        runtime_library=runtime_library,
        runtime_symbols=runtime_symbols,
        device_attribute_supported=device_attribute,
        reasons=tuple(reasons),
    )


def validate_fabric_clique(
    capabilities: Iterable[FabricCapability],
) -> tuple[bool, tuple[str, ...]]:
    capabilities = tuple(capabilities)
    reasons = []
    if not capabilities:
        return False, ("no fabric capability records were provided",)
    for capability in capabilities:
        reasons.extend(
            f"rank {capability.rank}: {reason}" for reason in capability.reasons
        )
    ppod_ids = {capability.ppod_id for capability in capabilities}
    vpod_ids = {capability.vpod_id for capability in capabilities}
    if len(ppod_ids) != 1:
        reasons.append(f"ranks span PPOD UUIDs: {sorted(ppod_ids, key=str)}")
    if len(vpod_ids) != 1:
        reasons.append(f"ranks span VPOD/clique IDs: {sorted(vpod_ids, key=str)}")
    expected_size = capabilities[0].vpod_size
    if expected_size is not None and len(capabilities) > expected_size:
        reasons.append(
            f"group has {len(capabilities)} ranks but VPOD size is {expected_size}"
        )
    return not reasons, tuple(reasons)

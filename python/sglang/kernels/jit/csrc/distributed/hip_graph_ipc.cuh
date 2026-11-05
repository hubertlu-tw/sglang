#include <sgl_kernel/ffi.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime_api.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace sglang {

namespace host::distributed {

struct HIPAllocationRange {
  uintptr_t base;
  size_t size;
  size_t offset;
};

inline HIPAllocationRange get_hip_allocation_range(uintptr_t ptr) {
  void* base = nullptr;
  size_t size = 0;
  RuntimeDeviceCheck(hipMemGetAddressRange(&base, &size, reinterpret_cast<void*>(ptr)));
  const auto base_address = reinterpret_cast<uintptr_t>(base);
  return {.base = base_address, .size = size, .offset = ptr - base_address};
}

/**
 * Batched HIP-IPC handle exchange for same-host graph input pointers.
 *
 * Handles always describe the base allocation. The returned offset preserves
 * tensor views into a caching-allocator block. Cross-host graph inputs use a
 * separate future fabric-VMM registration path.
 */
struct HIPGraphIPCManager : public tvm::ffi::Object {
 public:
  using IPCHandle = std::array<char, sizeof(hipIpcMemHandle_t)>;
  using FFIHandle = tvm::ffi::Array<char>;
  using BatchResult = tvm::ffi::Array<tvm::ffi::Tuple<FFIHandle, size_t>>;
  using Pointers = tvm::ffi::Array<uintptr_t>;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.HIPGraphIPCManager", HIPGraphIPCManager, tvm::ffi::Object);
  static constexpr bool _type_mutable = true;

  HIPGraphIPCManager() = default;
  ~HIPGraphIPCManager() {
    destroy();
  }

  BatchResult batch_get_handles(const Pointers& pointers) {
    RuntimeCheck(m_ptr_cache.empty(), "Internal error: stale HIP pointer cache");
    BatchResult result;
    result.reserve(pointers.size());
    using Tuple = tvm::ffi::Tuple<FFIHandle, size_t>;
    for (const auto pointer : pointers) {
      const auto [handle, offset] = get_handle(pointer);
      result.emplace_back(Tuple{to_ffi(handle), offset});
    }
    m_ptr_cache.clear();
    return result;
  }

  Pointers batch_open_handles(const BatchResult& handles) {
    Pointers result;
    result.reserve(handles.size());
    for (const auto& pair : handles) {
      const auto handle = from_ffi(get<0>(pair));
      result.push_back(open_handle(handle) + get<1>(pair));
    }
    return result;
  }

  void destroy() {
    for (const auto& [handle, pointer] : m_handle_cache) {
      if (m_local_handles.count(handle)) continue;
      RuntimeDeviceCheck(hipIpcCloseMemHandle(reinterpret_cast<void*>(pointer)));
    }
    m_handle_cache.clear();
    m_ptr_cache.clear();
    m_local_handles.clear();
  }

 private:
  static IPCHandle from_ffi(const FFIHandle& input) {
    RuntimeCheck(input.size() == sizeof(hipIpcMemHandle_t), "Invalid HIP IPC handle size");
    IPCHandle result{};
    for (size_t i = 0; i < result.size(); ++i)
      result[i] = static_cast<char>(input[i]);
    return result;
  }

  static FFIHandle to_ffi(const IPCHandle& input) {
    FFIHandle result;
    result.reserve(input.size());
    for (const auto byte : input)
      result.push_back(static_cast<uint8_t>(byte));
    return result;
  }

  std::pair<IPCHandle, size_t> get_handle(uintptr_t pointer) {
    auto iterator = m_ptr_cache.upper_bound(pointer);
    if (iterator != m_ptr_cache.begin()) {
      --iterator;
      const auto offset = pointer - iterator->first;
      if (offset < iterator->second.second) return {iterator->second.first, offset};
    }

    const auto range = get_hip_allocation_range(pointer);
    hipIpcMemHandle_t native{};
    RuntimeDeviceCheck(hipIpcGetMemHandle(&native, reinterpret_cast<void*>(range.base)));
    IPCHandle handle{};
    std::memcpy(handle.data(), &native, sizeof(native));
    const auto [_, inserted] = m_ptr_cache.try_emplace(range.base, handle, range.size);
    RuntimeCheck(inserted, "Internal error: duplicate HIP allocation base");
    m_handle_cache.try_emplace(handle, range.base);
    m_local_handles.insert(handle);
    return {handle, range.offset};
  }

  uintptr_t open_handle(const IPCHandle& handle) {
    const auto iterator = m_handle_cache.find(handle);
    if (iterator != m_handle_cache.end()) return iterator->second;

    hipIpcMemHandle_t native{};
    std::memcpy(&native, handle.data(), sizeof(native));
    void* pointer = nullptr;
    RuntimeDeviceCheck(hipIpcOpenMemHandle(&pointer, native, hipIpcMemLazyEnablePeerAccess));
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    const auto [_, inserted] = m_handle_cache.try_emplace(handle, address);
    RuntimeCheck(inserted, "Internal error: duplicate HIP IPC handle");
    return address;
  }

  struct HandleEqual {
    bool operator()(const IPCHandle& left, const IPCHandle& right) const {
      return std::memcmp(left.data(), right.data(), left.size()) == 0;
    }
  };

  struct HandleHash {
    size_t operator()(const IPCHandle& handle) const {
      return std::hash<std::string_view>{}(std::string_view{handle.data(), handle.size()});
    }
  };

  std::unordered_map<IPCHandle, uintptr_t, HandleHash, HandleEqual> m_handle_cache;
  std::unordered_set<IPCHandle, HandleHash, HandleEqual> m_local_handles;
  std::map<uintptr_t, std::pair<IPCHandle, size_t>> m_ptr_cache;
};

}  // namespace host::distributed

inline void register_hip_graph_ipc_manager() {
  namespace refl = tvm::ffi::reflection;
  using Class = host::distributed::HIPGraphIPCManager;
  refl::ObjectDef<Class>()
      .def(refl::init<>(), "__init__")
      .def("batch_get_handles", &Class::batch_get_handles)
      .def("batch_open_handles", &Class::batch_open_handles)
      .def("destroy", &Class::destroy);
}

}  // namespace sglang

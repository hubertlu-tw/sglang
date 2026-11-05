#include <sgl_kernel/ffi.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace sglang {

namespace host::distributed {

/**
 * Owns one static, uncached HIP allocation and maps the matching allocation
 * from every peer. It lives outside PyTorch symmetric memory because the
 * gfx1250 system-scope release/acquire flags were validated on HIP IPC
 * mappings.
 */
struct HIPIPCPlane : public tvm::ffi::Object {
 public:
  using IPCHandle = tvm::ffi::Array<char>;
  using IPCHandles = tvm::ffi::Array<IPCHandle>;
  using Pointers = tvm::ffi::Array<uintptr_t>;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.HIPIPCPlane", HIPIPCPlane, tvm::ffi::Object);
  static constexpr bool _type_mutable = true;

  HIPIPCPlane(size_t nbytes, uint32_t rank, uint32_t world_size)
      : m_nbytes(nbytes), m_rank(rank), m_world_size(world_size), m_ptrs(world_size, nullptr) {
    RuntimeCheck(nbytes > 0, "HIP IPC plane must be non-empty");
    RuntimeCheck(rank < world_size, "Invalid rank");
    RuntimeDeviceCheck(hipExtMallocWithFlags(&m_local, nbytes, hipDeviceMallocUncached));
    RuntimeDeviceCheck(hipMemset(m_local, 0, nbytes));
    RuntimeDeviceCheck(hipIpcGetMemHandle(&m_handle, m_local));
    m_ptrs[rank] = m_local;
  }

  ~HIPIPCPlane() {
    destroy();
  }

  IPCHandle get_handle() const {
    IPCHandle result;
    result.reserve(sizeof(hipIpcMemHandle_t));
    const auto* bytes = reinterpret_cast<const uint8_t*>(&m_handle);
    for (size_t i = 0; i < sizeof(hipIpcMemHandle_t); ++i)
      result.push_back(bytes[i]);
    return result;
  }

  void open_handles(const IPCHandles& handles) {
    RuntimeCheck(handles.size() == m_world_size, "Bad HIP IPC handle count");
    for (uint32_t peer = 0; peer < m_world_size; ++peer) {
      RuntimeCheck(handles[peer].size() == sizeof(hipIpcMemHandle_t), "Bad HIP IPC handle size");
      if (peer == m_rank) continue;
      hipIpcMemHandle_t handle;
      auto* bytes = reinterpret_cast<uint8_t*>(&handle);
      for (size_t i = 0; i < sizeof(handle); ++i)
        bytes[i] = static_cast<uint8_t>(handles[peer][i]);
      RuntimeDeviceCheck(hipIpcOpenMemHandle(&m_ptrs[peer], handle, hipIpcMemLazyEnablePeerAccess));
    }
  }

  Pointers get_pointers() const {
    Pointers result;
    result.reserve(m_world_size);
    for (const auto ptr : m_ptrs) {
      RuntimeCheck(ptr != nullptr, "HIP IPC handles have not all been opened");
      result.push_back(reinterpret_cast<uintptr_t>(ptr));
    }
    return result;
  }

  void destroy() {
    for (uint32_t peer = 0; peer < m_ptrs.size(); ++peer) {
      if (peer != m_rank && m_ptrs[peer] != nullptr) {
        RuntimeDeviceCheck(hipIpcCloseMemHandle(m_ptrs[peer]));
      }
      m_ptrs[peer] = nullptr;
    }
    if (m_local != nullptr) {
      RuntimeDeviceCheck(hipFree(m_local));
      m_local = nullptr;
    }
  }

 private:
  size_t m_nbytes;
  uint32_t m_rank;
  uint32_t m_world_size;
  void* m_local = nullptr;
  hipIpcMemHandle_t m_handle{};
  std::vector<void*> m_ptrs;
};

}  // namespace host::distributed

inline void register_hip_ipc_plane() {
  namespace refl = tvm::ffi::reflection;
  using Class = host::distributed::HIPIPCPlane;
  refl::ObjectDef<Class>()
      .def(refl::init<size_t, uint32_t, uint32_t>(), "__init__")
      .def("get_handle", &Class::get_handle)
      .def("open_handles", &Class::open_handles)
      .def("get_pointers", &Class::get_pointers)
      .def("destroy", &Class::destroy);
}

}  // namespace sglang

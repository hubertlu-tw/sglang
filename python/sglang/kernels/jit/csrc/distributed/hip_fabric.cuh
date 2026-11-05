#include <sgl_kernel/ffi.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime_api.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sglang {

namespace host::distributed {

/**
 * Owns one uncached, fabric-exportable HIP VMM allocation and maps the
 * matching allocation from every rank in one UALink/UALoE clique.
 *
 * Python exchanges the handles. Construction only creates and exports the
 * local allocation; open_handles() imports peer descriptors once every rank
 * has finished construction.
 */
struct HIPFabricPlane : public tvm::ffi::Object {
 public:
  using FabricHandle = tvm::ffi::Array<char>;
  using FabricHandles = tvm::ffi::Array<FabricHandle>;
  using Pointers = tvm::ffi::Array<uintptr_t>;
  using Sizes = tvm::ffi::Array<size_t>;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.HIPFabricPlane", HIPFabricPlane, tvm::ffi::Object);
  static constexpr bool _type_mutable = true;

  HIPFabricPlane(size_t requested_nbytes, uint32_t rank, uint32_t world_size)
      : m_requested_nbytes(requested_nbytes),
        m_rank(rank),
        m_world_size(world_size),
        m_ptrs(world_size, nullptr),
        m_mapped_sizes(world_size, 0),
        m_imported(world_size, false) {
    RuntimeCheck(requested_nbytes > 0, "HIP fabric plane must be non-empty");
    RuntimeCheck(rank < world_size, "Invalid rank");

    int device = 0;
    RuntimeDeviceCheck(hipGetDevice(&device));
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypeUncached;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    prop.requestedHandleTypes = hipMemHandleTypeFabric;

    size_t granularity = 0;
    RuntimeDeviceCheck(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    RuntimeCheck(granularity > 0, "HIP fabric allocation granularity is zero");
    m_nbytes = (requested_nbytes + granularity - 1) / granularity * granularity;

    RuntimeDeviceCheck(hipMemCreate(&m_local_handle, m_nbytes, &prop, 0));
    RuntimeDeviceCheck(hipMemAddressReserve(&m_local, m_nbytes, granularity, nullptr, 0));
    RuntimeDeviceCheck(hipMemMap(m_local, m_nbytes, 0, m_local_handle, 0));
    set_access(m_local, m_nbytes, device);
    RuntimeDeviceCheck(hipMemset(m_local, 0, m_nbytes));
    RuntimeDeviceCheck(hipMemExportToShareableHandle(&m_fabric_handle, m_local_handle, hipMemHandleTypeFabric, 0));
    m_ptrs[rank] = m_local;
    m_mapped_sizes[rank] = m_nbytes;
  }

  ~HIPFabricPlane() {
    destroy();
  }

  FabricHandle get_handle() const {
    FabricHandle result;
    result.reserve(sizeof(hipMemFabricHandle_t));
    const auto* bytes = reinterpret_cast<const uint8_t*>(&m_fabric_handle);
    for (size_t i = 0; i < sizeof(hipMemFabricHandle_t); ++i)
      result.push_back(bytes[i]);
    return result;
  }

  size_t get_nbytes() const {
    return m_nbytes;
  }

  void open_handles(const FabricHandles& handles, const Sizes& sizes) {
    RuntimeCheck(handles.size() == m_world_size, "Bad HIP fabric handle count");
    RuntimeCheck(sizes.size() == m_world_size, "Bad HIP fabric allocation-size count");
    int device = 0;
    RuntimeDeviceCheck(hipGetDevice(&device));
    for (uint32_t peer = 0; peer < m_world_size; ++peer) {
      RuntimeCheck(handles[peer].size() == sizeof(hipMemFabricHandle_t), "Bad HIP fabric handle size");
      if (peer == m_rank) continue;

      hipMemFabricHandle_t descriptor{};
      auto* bytes = reinterpret_cast<uint8_t*>(&descriptor);
      for (size_t i = 0; i < sizeof(descriptor); ++i)
        bytes[i] = static_cast<uint8_t>(handles[peer][i]);

      hipMemGenericAllocationHandle_t imported{};
      RuntimeDeviceCheck(hipMemImportFromShareableHandle(&imported, &descriptor, hipMemHandleTypeFabric));

      hipMemAllocationProp prop{};
      RuntimeDeviceCheck(hipMemGetAllocationPropertiesFromHandle(&prop, imported));
      size_t granularity = 0;
      RuntimeDeviceCheck(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
      const auto peer_nbytes = sizes[peer];
      RuntimeCheck(peer_nbytes >= m_requested_nbytes, "Peer HIP fabric allocation is smaller than requested");
      RuntimeCheck(peer_nbytes % granularity == 0, "Peer HIP fabric allocation is not granularity-aligned");
      void* peer_ptr = nullptr;
      RuntimeDeviceCheck(hipMemAddressReserve(&peer_ptr, peer_nbytes, granularity, nullptr, 0));
      RuntimeDeviceCheck(hipMemMap(peer_ptr, peer_nbytes, 0, imported, 0));
      set_access(peer_ptr, peer_nbytes, device);
      RuntimeDeviceCheck(hipMemRelease(imported));

      m_ptrs[peer] = peer_ptr;
      m_mapped_sizes[peer] = peer_nbytes;
      m_imported[peer] = true;
    }
  }

  Pointers get_pointers() const {
    Pointers result;
    result.reserve(m_world_size);
    for (const auto ptr : m_ptrs) {
      RuntimeCheck(ptr != nullptr, "HIP fabric handles have not all been opened");
      result.push_back(reinterpret_cast<uintptr_t>(ptr));
    }
    return result;
  }

  void destroy() {
    for (uint32_t peer = 0; peer < m_world_size; ++peer) {
      if (!m_imported[peer] || m_ptrs[peer] == nullptr) continue;
      RuntimeDeviceCheck(hipMemUnmap(m_ptrs[peer], m_mapped_sizes[peer]));
      RuntimeDeviceCheck(hipMemAddressFree(m_ptrs[peer], m_mapped_sizes[peer]));
      m_ptrs[peer] = nullptr;
      m_mapped_sizes[peer] = 0;
      m_imported[peer] = false;
    }
    if (m_local != nullptr) {
      RuntimeDeviceCheck(hipMemUnmap(m_local, m_nbytes));
      RuntimeDeviceCheck(hipMemAddressFree(m_local, m_nbytes));
      m_local = nullptr;
    }
    if (m_local_handle != 0) {
      RuntimeDeviceCheck(hipMemRelease(m_local_handle));
      m_local_handle = 0;
    }
  }

 private:
  static void set_access(void* ptr, size_t nbytes, int device) {
    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = device;
    access.flags = hipMemAccessFlagsProtReadWrite;
    RuntimeDeviceCheck(hipMemSetAccess(ptr, nbytes, &access, 1));
  }

  size_t m_requested_nbytes;
  size_t m_nbytes = 0;
  uint32_t m_rank;
  uint32_t m_world_size;
  void* m_local = nullptr;
  hipMemGenericAllocationHandle_t m_local_handle{};
  hipMemFabricHandle_t m_fabric_handle{};
  std::vector<void*> m_ptrs;
  std::vector<size_t> m_mapped_sizes;
  std::vector<bool> m_imported;
};

}  // namespace host::distributed

inline void register_hip_fabric_plane() {
  namespace refl = tvm::ffi::reflection;
  using Class = host::distributed::HIPFabricPlane;
  refl::ObjectDef<Class>()
      .def(refl::init<size_t, uint32_t, uint32_t>(), "__init__")
      .def("get_handle", &Class::get_handle)
      .def("get_nbytes", &Class::get_nbytes)
      .def("open_handles", &Class::open_handles)
      .def("get_pointers", &Class::get_pointers)
      .def("destroy", &Class::destroy);
}

}  // namespace sglang

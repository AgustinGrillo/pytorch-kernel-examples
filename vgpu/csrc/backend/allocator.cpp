#include <ATen/ATen.h>
#include <c10/core/impl/alloc_cpu.h>
#include <iostream>

// ====================================
// ========= Custom Allocator =========
// ====================================

// A custom (cpu-based) allocator for the new backend
struct VgpuAllocator final : at::Allocator {
  VgpuAllocator() = default;
  at::DataPtr allocate(size_t nbytes) override {
    // std::cout << "Custom allocator's allocate() called!" << std::endl;
    void *data = c10::alloc_cpu(nbytes);
    // void *data = malloc(nbytes);
    return {data, data, &ReportAndDelete,
            at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void ReportAndDelete(void *ptr) {
    if (!ptr) {
      return;
    }
    // std::cout << "Custom allocator's delete() called!" << std::endl;
    c10::free_cpu(ptr);
    // free(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override { return &ReportAndDelete; }

  void copy_data(void *dest, const void *src,
                 std::size_t count) const override {
    // std::cout << "Custom allocator's copy_data() called!" << std::endl;
    default_copy_data(dest, src, count);
  }
};

// Register allocator
static VgpuAllocator g_allocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

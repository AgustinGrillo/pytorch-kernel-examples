#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/alloc_cpu.h>
#include <iostream>
#include <torch/torch.h>

namespace vgpu {

// ====================================
// ========= Custom Operators =========
// ====================================

// Custom matrix multiplication operation
torch::Tensor mm(const torch::Tensor &a, const torch::Tensor &b) {
  std::cout << "Custom torch.mm called!" << std::endl;

  torch::Tensor output = torch::empty({a.size(0), b.size(1)});

  auto a_accessor = a.accessor<float, 2>();
  auto b_accessor = b.accessor<float, 2>();
  auto output_accessor = output.accessor<float, 2>();

  for (int i = 0; i < a.size(0); i++) {
    for (int j = 0; j < b.size(1); j++) {
      float sum = 0;
      for (int k = 0; k < a.size(1); k++) {
        sum += a_accessor[i][k] * b_accessor[k][j];
      }
      output_accessor[i][j] = sum;
    }
  }
  return output.clone();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) { m.impl("mm", &mm); }

// // No Autograd support
// TORCH_LIBRARY_IMPL(aten, Autograd, m) {
//   m.impl("mm", torch::autograd::autogradNotImplementedFallback());
// }

void custom_cpu_fallback(const c10::OperatorHandle &op,
                         torch::jit::Stack *stack) {
  TORCH_WARN("Operator ", op.schema().operator_name(),
             " not supported in custom backend (vGPU). Falling back to CPU.");
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

// ====================================
// ========= Custom Allocator =========
// ====================================

// A custom (cpu-based) allocator for the new backend
struct CustomAllocator final : at::Allocator {
  CustomAllocator() = default;
  at::DataPtr allocate(size_t nbytes) override {
    std::cout << "Custom allocator's allocate() called!" << std::endl;
    void *data = c10::alloc_cpu(nbytes);
    // void *data = malloc(nbytes);
    return {data, data, &ReportAndDelete,
            at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void ReportAndDelete(void *ptr) {
    if (!ptr) {
      return;
    }
    std::cout << "Custom allocator's delete() called!" << std::endl;
    c10::free_cpu(ptr);
    // free(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override { return &ReportAndDelete; }

  void copy_data(void *dest, const void *src,
                 std::size_t count) const override {
    std::cout << "Custom allocator's copy_data() called!" << std::endl;
    default_copy_data(dest, src, count);
  }
};

// Register allocator
static CustomAllocator allocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &allocator);

// // To create a TensorImpl on PrivateUse1 backend, pass in the following ks to
// // TensorImpl creation.
// c10::DispatchKeySet ks = c10::DispatchKeySet{
//     c10::DispatchKey::PrivateUse1, c10::DispatchKey::AutogradPrivateUse1};
//
// /* Example TensorImpl constructor */
// c10::TensorImpl(c10::Storage &&storage, c10::DispatchKeySet ks,
//                 const caffe2::TypeMeta data_type);

// ===========================
// ========= Kernels =========
// ===========================

// Custom empty implementation
at::Tensor empty_memory_format(
    c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {

  std::cout << "Custom aten::empty.memory_format() called!" << std::endl;

  // at::ScalarType dtype = dtype_opt ? *dtype_opt : at::kFloat;
  at::ScalarType dtype = c10::dtype_or_default(dtype_opt);

  c10::DispatchKeySet ks = c10::DispatchKeySet{
      c10::DispatchKey::PrivateUse1, c10::DispatchKey::AutogradPrivateUse1};

  // c10::Storage storage = c10::Storage(
  //     c10::Storage::use_byte_size_t(), options.dtype().itemsize() *
  //     size.size(), c10::DataPtr(nullptr, c10::Device(options.device())),
  //     /*allocator=*/nullptr,
  //     /*resizable=*/false);
  //
  //
  // c10::intrusive_ptr<c10::TensorImpl> impl =
  //     c10::make_intrusive<c10::TensorImpl>(std::move(storage), ks,
  //                                          options.dtype());
  //
  // // torch::Tensor tensor = torch::Tensor(std::move(impl));
  // torch::Tensor tensor = torch::Tensor::wrap_tensor_impl(impl);

  // c10::Allocator *allocator = c10::GetAllocator(c10::DeviceType::CPU);

  torch::Tensor tensor =
      at::detail::empty_generic(size, &allocator, ks, dtype, memory_format_opt);

  return tensor;
}

torch::Tensor _copy_from(const torch::Tensor &self, const torch::Tensor &dst,
                         bool non_blocking) {
  std::cout << "Custom _copy_from called!" << std::endl;
  std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(),
              self.storage().nbytes());
  return dst;
}

torch::Tensor _copy_from_and_resize(const torch::Tensor &self,
                                    const torch::Tensor &dst) {
  return vgpu::_copy_from(self, dst, false);
}

torch::Tensor view(const torch::Tensor &self, c10::IntArrayRef size) {
  std::cout << "View running inside custom backend!" << std::endl;
  return self.clone();
}

torch::Tensor &fill_scalar(torch::Tensor &self, const at::Scalar &value) {
  std::cout << "Fill scalar running inside custom backend!" << std::endl;
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  // TODO: Should fill the tensor's data with "value".
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_memory_format);
  m.impl("_copy_from", _copy_from);
  m.impl("_copy_from_and_resize", _copy_from_and_resize);
  m.impl("fill_.Scalar", fill_scalar);
  m.impl("view", view);
}

/* Register generator for the new backend */

// struct CustomGeneratorImpl : public c10::GeneratorImpl {
//   // Implementation of generator in new backend
// };
//
// at::Generator make_custom_generator(c10::DeviceIndex device_index) {
//   return at::make_generator<CustomGeneratorImpl>(device_index);
// }
//
// REGISTER_GENERATOR_PRIVATEUSE1(make_cumstom_generator);

} // namespace vgpu

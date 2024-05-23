#include <ATen/ATen.h>
#include <ATen/InferSize.h>
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

  torch::Tensor output = torch::empty({a.size(0), b.size(1)}, a.options());

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
  return output;
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
static CustomAllocator g_allocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

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
  caffe2::TypeMeta meta_dtype = scalarTypeToTypeMeta(dtype);
  size_t size_bytes =
      at::detail::computeStorageNbytesContiguous(size, meta_dtype.itemsize());

  c10::DispatchKeySet ks = c10::DispatchKeySet{c10::DispatchKey::PrivateUse1};

  c10::Allocator *allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);

  /*
   Storage:
   - c10::StorageImpl / c10::Storage
   - The actual physical data, physical size, dtype, device.
    - Fields: data_ptr, size, allocator..
   **/

  c10::Storage storage =
      c10::Storage(c10::Storage::use_byte_size_t(), size_bytes, allocator,
                   /*resizable=*/true);

  // auto storage = c10::make_intrusive<c10::StorageImpl>(
  //     c10::StorageImpl::use_byte_size_t(), size_bytes,
  //     // c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1)),
  //     allocator,
  //     /*resizable=*/true);

  /*
  TensorImpl:
  - c10::TensorImpl
  - The logical view (size, strides, storage_offset) of the tensor.
    - Fields: sizes, dtype, type_id, strides, storage_offset, storage, dim.
  **/

  // c10::intrusive_ptr<c10::TensorImpl> impl =
  //     c10::make_intrusive<c10::TensorImpl>(std::move(storage), ks,
  //     meta_dtype);
  // torch::Tensor tensor = torch::Tensor::wrap_tensor_impl(impl);

  torch::Tensor tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage), ks, meta_dtype);

  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);

  // torch::Tensor tensor =
  //     at::detail::empty_generic(size, allocator, ks, dtype,
  //     memory_format_opt);

  return tensor;
}

torch::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                            c10::optional<at::ScalarType> dtype,
                            c10::optional<at::Layout> layout,
                            c10::optional<at::Device> device,
                            c10::optional<bool> pin_memory) {

  std::cout << "Custom aten::empty_strided() called!" << std::endl;

  // TODO:
  // caffe2::TypeMeta meta_dtype =
  //     scalarTypeToTypeMeta(c10::dtype_or_default(dtype));
  // size_t size_bytes =
  //     at::detail::computeStorageNbytes(size, stride, meta_dtype.itemsize());
  //
  // c10::DispatchKeySet ks =
  // c10::DispatchKeySet{c10::DispatchKey::PrivateUse1};
  //
  // c10::Allocator *allocator =
  // c10::GetAllocator(c10::DeviceType::PrivateUse1);
  //
  // c10::Storage storage =
  //     c10::Storage(c10::Storage::use_byte_size_t(), size_bytes, allocator,
  //                  /*resizable=*/true);
  //
  // auto tensor = at::detail::make_tensor<c10::TensorImpl>(std::move(storage),
  // ks,
  //                                                        meta_dtype);
  // tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

  torch::Tensor tensor = vgpu::empty_memory_format(size, dtype, layout, device,
                                                   pin_memory, c10::nullopt);
  return tensor;
}

torch::Tensor _copy_from(const torch::Tensor &self, const torch::Tensor &dst,
                         bool non_blocking) {
  std::cout << "Custom aten::_copy_from called!" << std::endl;
  std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(),
              self.storage().nbytes());
  return dst;
}

torch::Tensor _copy_from_and_resize(const torch::Tensor &self,
                                    const torch::Tensor &dst) {
  return vgpu::_copy_from(self, dst, false);
}

torch::Tensor view(const torch::Tensor &self, c10::IntArrayRef shape) {
  std::cout << "Custom aten::view called!" << std::endl;
  // at::DimVector inferred_size = at::infer_size_dv(shape, self.numel());
  //
  // torch::Tensor self_ = at::detail::make_tensor<at::TensorImpl>(
  //     self.storage(), self.storage_offset(), self.options());
  // return self_;
  return self.clone();

  // torch::Tensor data = at::alias(self);
  // TORCH_CHECK(data.is_contiguous(), "View imlemented on contiguous array");
  // data.getIntrusivePtr()->set_sizes_contiguous(inferred_size);
  // return data;
}

torch::Tensor &fill_scalar(torch::Tensor &self, const at::Scalar &value) {
  std::cout << "Custom aten::fill_ called!" << std::endl;
  // TODO: Should fill the tensor's data with "value".
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("aten::empty.memory_format", empty_memory_format);
  m.impl("aten::empty_strided", empty_strided);
  m.impl("aten::_copy_from", _copy_from);
  m.impl("aten::_copy_from_and_resize", _copy_from_and_resize);
  m.impl("aten::fill_.Scalar", fill_scalar);
  m.impl("aten::view", view);
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

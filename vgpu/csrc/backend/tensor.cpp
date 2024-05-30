#include <ATen/ATen.h>
#include <ATen/InferSize.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/alloc_cpu.h>
#include <iostream>
#include <torch/torch.h>

namespace vgpu {

// ===========================
// ========= Kernels =========
// ===========================

// Custom empty implementation
at::Tensor empty_memory_format(
    c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {

  // std::cout << "Custom aten::empty.memory_format() called!" << std::endl;

  // at::ScalarType dtype = dtype_opt ? *dtype_opt : at::kFloat;
  at::ScalarType dtype = c10::dtype_or_default(dtype_opt);
  caffe2::TypeMeta meta_dtype = scalarTypeToTypeMeta(dtype);
  size_t size_bytes =
      at::detail::computeStorageNbytesContiguous(size, meta_dtype.itemsize());

  c10::DispatchKeySet ks = c10::DispatchKeySet{c10::DispatchKey::PrivateUse1};

  c10::Allocator *allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);

  c10::Storage storage =
      c10::Storage(c10::Storage::use_byte_size_t(), size_bytes, allocator,
                   /*resizable=*/true);

  torch::Tensor tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage), ks, meta_dtype);

  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);

  return tensor;
}

torch::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                            c10::optional<at::ScalarType> dtype,
                            c10::optional<at::Layout> layout,
                            c10::optional<at::Device> device,
                            c10::optional<bool> pin_memory) {

  // std::cout << "Custom aten::empty_strided() called!" << std::endl;

  torch::Tensor tensor = vgpu::empty_memory_format(size, dtype, layout, device,
                                                   pin_memory, c10::nullopt);
  return tensor;
}

torch::Tensor _copy_from(const torch::Tensor &self, const torch::Tensor &dst,
                         bool non_blocking) {
  // std::cout << "Custom aten::_copy_from called!" << std::endl;

  if (self.is_same(dst)) {
    return self;
  }
  std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(),
              self.storage().nbytes());

  return dst;
}

torch::Tensor _copy_from_and_resize(const torch::Tensor &self,
                                    const torch::Tensor &dst) {
  // std::cout << "Custom aten::_copy_from_and_resize called!" << std::endl;
  return vgpu::_copy_from(self, dst, false);
}

torch::Tensor view(const torch::Tensor &self, c10::IntArrayRef shape) {
  // std::cout << "Custom aten::view called!" << std::endl;
  at::DimVector inferred_size = at::infer_size_dv(shape, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);

  torch::Tensor self_ = at::detail::make_tensor<c10::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());

  auto *self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(self.storage_offset());
  self_tmp_->set_sizes_and_strides(inferred_size, *stride);

  return self_;
}

torch::Tensor &fill_scalar(torch::Tensor &self, const at::Scalar &value) {
  // std::cout << "Custom aten::fill_ called!" << std::endl;
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

} // namespace vgpu

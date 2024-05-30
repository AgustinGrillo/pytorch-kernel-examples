#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>

// ============================
// ========= Fallback =========
// ============================
//
void custom_cpu_fallback(const c10::OperatorHandle &op,
                         torch::jit::Stack *stack) {
  // TORCH_WARN("Operator ", op.schema().operator_name(),
  //            " not supported in custom backend (vGPU). Falling back to
  //            CPU.");
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

// TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
//   m.fallback(torch::CppFunction::makeFallthrough());
// }

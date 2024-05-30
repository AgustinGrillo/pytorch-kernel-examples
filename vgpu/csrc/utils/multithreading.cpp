#include "multithreading.hpp"

// Global variable (for demonstration purposes)
int64_t num_threads = 1;

/**
 * Set the number of threads for multithreaded operations
 */

void vgpu::set_num_threads(int64_t threads) {
  num_threads = threads;
#ifdef DEBUG
  std::cout << "Setting " << num_threads << " threads" << std::endl;
#endif
}

int64_t vgpu::get_num_threads() { return num_threads; }

void vgpu::set_thread_affinity_for_core(int i) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(i, &cpuset);
  auto s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (s != 0) {
    std::cerr << "Error setting thread affinity for thread " << i << std::endl;
  }
}

/* Python bindings */
TORCH_LIBRARY_FRAGMENT(vgpu, m) {
  m.def("set_num_threads", vgpu::set_num_threads);
}

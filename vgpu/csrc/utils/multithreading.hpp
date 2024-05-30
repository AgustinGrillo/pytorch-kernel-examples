#include <cstdint>
#include <thread>
#include <torch/torch.h>

namespace vgpu {

/**
 * Set the number of threads for multithreaded operations
 */
void set_num_threads(int64_t threads);

/**
 * Get the number of threads for multithreaded operations
 */
int64_t get_num_threads();

void set_thread_affinity_for_core(int i);

} // namespace vgpu

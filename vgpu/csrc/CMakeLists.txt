cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(vgpu)

find_package(Torch REQUIRED)

# set MKL root directory
set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")

# Add debug preprocessor directive
# add_definitions(-DDEBUG)

# Define our library target 
add_library(vgpu SHARED ops/mmult.cpp utils/multithreading.cpp backend/tensor.cpp 
            backend/fallback.cpp backend/allocator.cpp) 
# define architecture 
target_compile_options(vgpu PRIVATE -march=native)
# define optimization level 
target_compile_options(vgpu PRIVATE -O3)
# Enable C++20
target_compile_features(vgpu PRIVATE cxx_std_20)

# Include MKL headers 
target_include_directories(vgpu PRIVATE "${MKL_ROOT}/include")
# Link against MKL 
target_link_libraries(vgpu "${MKL_ROOT}/lib/intel64/libmkl_intel_ilp64.a" "${MKL_ROOT}/lib/intel64/libmkl_sequential.a" "${MKL_ROOT}/lib/intel64/libmkl_core.a")
# Link against LibTorch
target_link_libraries(vgpu "${TORCH_LIBRARIES}")
# Link against math and dynamic linking libraries
target_link_libraries(vgpu m dl)


# Define executable target (benchmark)
add_executable(bechmark benchmark.cpp)
# Enable C++20
target_compile_features(bechmark PRIVATE cxx_std_20)
# define architecture
target_compile_options(bechmark PRIVATE -march=native)
# define optimization level
target_compile_options(bechmark PRIVATE -O3)

# link against custom backend
target_link_libraries(bechmark vgpu)
target_include_directories(bechmark PRIVATE "${MKL_ROOT}/include")

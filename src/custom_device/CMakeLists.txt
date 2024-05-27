cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_backend)

find_package(Torch REQUIRED)

# set MKL root directory
set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")

# Add debug preprocessor directive
add_definitions(-DDEBUG)

# Define our library target
add_library(custom_backend SHARED custom_backend.cpp)
# Enable C++20
target_compile_features(custom_backend PRIVATE cxx_std_20)
# define architecture 
target_compile_options(custom_backend PRIVATE -march=native)
# define optimization level 
target_compile_options(custom_backend PRIVATE -O3)
# Link against LibTorch
target_link_libraries(custom_backend "${TORCH_LIBRARIES}")

# Include MKL headers 
target_include_directories(custom_backend PRIVATE "${MKL_ROOT}/include")
# Link against MKL 
target_link_libraries(custom_backend "${MKL_ROOT}/lib/intel64/libmkl_intel_ilp64.a" "${MKL_ROOT}/lib/intel64/libmkl_sequential.a" "${MKL_ROOT}/lib/intel64/libmkl_core.a")
# Link against math and dynamic linking libraries
target_link_libraries(custom_backend m dl)



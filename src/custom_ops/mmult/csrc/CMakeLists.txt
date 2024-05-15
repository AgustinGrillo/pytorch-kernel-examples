cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_mmult)

find_package(Torch REQUIRED)

# set MKL root directory
set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")

# Add debug preprocessor directive
add_definitions(-DDEBUG)

# Define our library target
add_library(custom_mmult SHARED mmult.cpp)
# define architecture 
target_compile_options(custom_mmult PRIVATE -march=native)
# define optimization level 
target_compile_options(custom_mmult PRIVATE -O3)
# Enable C++20
target_compile_features(custom_mmult PRIVATE cxx_std_20)
# Link against LibTorch
target_link_libraries(custom_mmult "${TORCH_LIBRARIES}")


# Include MKL headers 
target_include_directories(custom_mmult PRIVATE "${MKL_ROOT}/include")
# Link against MKL 
target_link_libraries(custom_mmult "${MKL_ROOT}/lib/intel64/libmkl_intel_ilp64.a" "${MKL_ROOT}/lib/intel64/libmkl_sequential.a" "${MKL_ROOT}/lib/intel64/libmkl_core.a")
# Link against pthread, math and dynamic linking libraries
target_link_libraries(custom_mmult pthread m dl)



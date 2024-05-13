cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_mmult)

find_package(Torch REQUIRED)

# Define our library target
add_library(custom_mmult SHARED custom_mmult_op.cpp)
# Enable C++14
target_compile_features(custom_mmult PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(custom_mmult "${TORCH_LIBRARIES}")

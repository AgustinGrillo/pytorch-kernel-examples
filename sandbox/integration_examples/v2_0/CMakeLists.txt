cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_backend)

find_package(Torch REQUIRED)

# Define our library target
add_library(custom_backend SHARED custom_backend.cpp)
# Enable C++14
target_compile_features(custom_backend PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(custom_backend "${TORCH_LIBRARIES}")

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(transformer-learning)

find_package(Torch REQUIRED)

set(CMAKE_XCC_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(transformer-learning main.cpp)
target_link_libraries(transformer-learning "${TORCH_LIBRARIES}")
set_property(TARGET transformer-learning PROPERTY CXX_STANDARD 17)

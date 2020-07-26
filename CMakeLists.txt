cmake_minimum_required(VERSION 3.1)
project(QCSimulatorRoot)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++0x -O2 -g -Wall")
set(CUDA_NVCC_FLAGS "-std=c++11 -O2 -g -arch=compute_70 -code=sm_70 --ptxas-options=-v")

include_directories ("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("src")
add_executable(parser tasks/from_qasm.cpp)
target_link_libraries(parser QCSimulator)



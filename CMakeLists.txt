cmake_minimum_required(VERSION 3.1)
project(QCSimulatorRoot)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++0x -O2 -g -Wall")
set(CUDA_NVCC_FLAGS "-std=c++11 -O2 -g -arch=compute_70 -code=sm_70 --ptxas-options=-v")

option(USE_GROUP "Grouping the gates before execution" ON)
option(SHOW_SCHEDULE "Print the schedule" ON)
option(SHOW_SUMMARY "Show the running details" ON)
option(MEASURE_STAGE "Measure time of each stage" OFF)

if (USE_GROUP)
    add_definitions(-DUSE_GROUP)
endif(USE_GROUP)
if (SHOW_SCHEDULE)
    add_definitions(-DSHOW_SCHEDULE)
endif(SHOW_SCHEDULE)
if (SHOW_SUMMARY)
    add_definitions(-DSHOW_SUMMARY)
endif(SHOW_SUMMARY)
if (MEASURE_STAGE)
    add_definitions(-DMEASURE_STAGE)
endif(MEASURE_STAGE)


include_directories ("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("src")
add_executable(main main.cpp)
target_link_libraries(main QCSimulator)



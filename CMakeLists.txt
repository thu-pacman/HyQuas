cmake_minimum_required(VERSION 3.1)
project(QCSimulatorRoot CXX)
set(CMAKE_CXX_FLAGS "-std=c++0x -O3 -g -Wall")

include_directories ("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("src")

set(TASKS GHZ_QFT random)

foreach(TASK IN LISTS TASKS)
	message(STATUS "Compiling ${TASK}")
  add_executable(${TASK} tasks/${TASK}.cpp)
  target_link_libraries(${TASK} QCSimulator)
endforeach()
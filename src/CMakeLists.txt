project(QCSimulator)
aux_source_directory(. SRC_CXX)
set(SRC_CU ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu)

cuda_add_library(QCSimulator ${SRC_CXX} ${SRC_CU})

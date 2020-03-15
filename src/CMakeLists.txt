project(QCSimulator)
aux_source_directory(. SRC_CXX)

cuda_add_library(QCSimulator ${SRC_CXX})

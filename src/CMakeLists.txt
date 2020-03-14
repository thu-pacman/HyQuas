project(QCSimulator CXX)

aux_source_directory(. DIR_LIB_SRCS)
add_library (QCSimulator ${DIR_LIB_SRCS})

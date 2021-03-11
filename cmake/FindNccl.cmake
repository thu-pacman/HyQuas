if (NCCL_LIBRARY)
  if(NOT USE_NCCL_LIB_PATH)
    # Don't cache NCCL_LIBRARY to enable switching between static and shared.
    unset(NCCL_LIBRARY CACHE)
  endif(NOT USE_NCCL_LIB_PATH)
endif()

if (BUILD_WITH_SHARED_NCCL)
  # libnccl.so
  set(NCCL_LIB_NAME nccl)
else ()
  # libnccl_static.a
  set(NCCL_LIB_NAME nccl_static)
endif (BUILD_WITH_SHARED_NCCL)

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  PATHS $ENV{NCCL_ROOT}/include ${NCCL_ROOT}/include)

find_library(NCCL_LIBRARY
  NAMES ${NCCL_LIB_NAME}
  PATHS $ENV{NCCL_ROOT}/lib/ ${NCCL_ROOT}/lib)

message(STATUS "Using nccl library: ${NCCL_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nccl DEFAULT_MSG
                                  NCCL_INCLUDE_DIR NCCL_LIBRARY)

mark_as_advanced(
  NCCL_INCLUDE_DIR
  NCCL_LIBRARY
)
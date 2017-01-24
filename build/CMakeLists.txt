# CMakeLists.txt
cmake_minimum_required(VERSION 2.8)

project( neunet)

# set( CUDA_TOOLKIT_ROOT_DIR "/Developer/NVIDIA/CUDA-8.0/")
find_package(CUDA QUIET REQUIRED)

set( CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set( ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(SOURCE
   ${SRC_DIR}/layer.cpp
   ${SRC_DIR}/main.cpp
   ${SRC_DIR}/neuron.cpp
   ${SRC_DIR}/network.cpp
   ${SRC_DIR}/parallelnetwork.cpp
   ${SRC_DIR}/parallel.cpp
)

set(CUDA_SOURCE
   ${SOURCE}
   ${SRC_DIR}/gpunetwork.cu
   ${SRC_DIR}/device.cu
)

set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}/../include
   /usr/local/include
   /usr/include
   /opt/cuda/include
)

IF (WIN32)
ELSE()
  set( CMAKE_CXX_FLAGS "-g -W -Wall -pthread -std=c++11")
ENDIF()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2")

set( CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets --expt-relaxed-constexpr --resource-usage -arch=sm_50 --use_fast_math")

include_directories( ${INCLUDE_DIRS})
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_NAME "neunet")

#add_executable(${EXECUTABLE_NAME} ${SOURCE})
cuda_add_executable( ${EXECUTABLE_NAME} ${CUDA_SOURCE} )
set_target_properties( ${EXECUTABLE_NAME} PROPERTIES COMPILE_FLAGS "-DDEVICE_COMPILE")

# CMakeLists.txt
cmake_minimum_required(VERSION 2.8)
project( neunet)

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

set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}/../include
   /usr/include
)

IF (WIN32)
ELSE()
  set( CMAKE_CXX_FLAGS "-g -W -Wall -pthread -std=c++11")
ENDIF()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2")

include_directories( ${INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_NAME "neunet")

add_executable(${EXECUTABLE_NAME} ${SOURCE})

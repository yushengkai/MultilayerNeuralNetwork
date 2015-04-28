# Main entry of blas_nn project build system
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE(${CMAKE_SOURCE_DIR}/cmake_common.txt)
MESSAGE(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
MESSAGE(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")

PROJECT(blas_nn)
MESSAGE(STATUS "PROJECT_SOURCE_DIR= ${PROJECT_SOURCE_DIR}")
SET(SRC_ROOT ${PROJECT_SOURCE_DIR}/src/nn)


ADD_EXECUTABLE(${exec_name} ${SRC_ROOT}/test/main.cc)
TARGET_LINK_LIBRARIES(${exec_name} ${LIBS})
SET_TARGET_PROPERTIES(${exec_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


ADD_EXECUTABLE(softmax ${SRC_ROOT}/test/softmax.cc)
TARGET_LINK_LIBRARIES(softmax ${LIBS})
SET_TARGET_PROPERTIES(softmax PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

aux_source_directory(./src/nn/main TRAIN_SRCS)
aux_source_directory(./src/nn/net TRAIN_SRCS)
aux_source_directory(./src/nn/tool TRAIN_SRCS)
aux_source_directory(./src/nn/solver TRAIN_SRCS)
ADD_EXECUTABLE(train ${TRAIN_SRCS})
TARGET_LINK_LIBRARIES(train ${LIBS})
SET_TARGET_PROPERTIES(train PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

aux_source_directory(./src/nn/net TEST_SRCS)
aux_source_directory(./src/nn/tool TEST_SRCS)
aux_source_directory(./src/nn/solver TEST_SRCS)
ADD_EXECUTABLE(all_test ${TEST_SRCS} src/nn/test/all_test.cc)
TARGET_LINK_LIBRARIES(all_test ${LIBS})
SET_TARGET_PROPERTIES(all_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


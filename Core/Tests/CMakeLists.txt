cmake_minimum_required(VERSION 3.21)
project(Tests)

add_subdirectory(Lib)

add_executable(Tests operations/softmax_test.cpp network/neural_network_test.cpp)
target_link_libraries(Tests PRIVATE Catch2::Catch2WithMain Core)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Lib/Catch2/extras)
include(CTest)
include(Catch)
catch_discover_tests(Tests)

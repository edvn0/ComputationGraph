cmake_minimum_required(VERSION 3.21)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

set(CG_BUILD_TESTS OFF)
set(CG_IS_DEBUG OFF)

if(${CG_DEBUG})
    set(CG_BUILD_TESTS ON)
    set(CG_IS_DEBUG ON)
endif()

if(${CG_FORCE_TESTS})
    set(CG_BUILD_TESTS ON)
endif()

enable_testing()
add_subdirectory(Core)
add_subdirectory(App)

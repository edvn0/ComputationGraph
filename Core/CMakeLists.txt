cmake_minimum_required(VERSION 3.10)
project(Core)

add_subdirectory(Lib)

set(CORE_INCLUDES
    src/nodes/operations/ReduceSumOperation.cpp
    include/nodes/operations/ReduceSumOperation.hpp
    src/nodes/operations/MultiplyOperation.cpp
    include/nodes/operations/MultiplyOperation.hpp
    src/nodes/operations/MultiplyOperation.cpp
    include/nodes/operations/MultiplyOperation.hpp
    src/nodes/operations/LogOperation.cpp
    include/nodes/operations/LogOperation.hpp
    include/core/PointerMap.hpp
    include/core/Recursion.hpp
    include/network/NeuralNetwork.hpp
    include/network/Session.hpp
    include/network/LayerDefinition.hpp
    include/nodes/OperationNode.hpp
    include/nodes/Initializers.hpp
    include/nodes/Node.hpp
    include/nodes/PlaceholderNode.hpp
    include/nodes/ValueNode.hpp
    include/nodes/operations/NegateOperation.hpp
    include/nodes/operations/ReLUOperation.hpp
    include/nodes/operations/AdditionOperation.hpp
    include/nodes/operations/MatrixMultiplyOperation.hpp
    src/network/NeuralNetwork.cpp
    src/network/Session.cpp
    src/nodes/operations/MatrixMultiplyOperation.cpp
    src/nodes/operations/AdditionOperation.cpp
    src/nodes/operations/ReLUOperation.cpp
    src/nodes/operations/SoftmaxOperation.cpp
    src/nodes/Node.cpp
    src/nodes/OperationNode.cpp
    src/nodes/ValueNode.cpp
)
add_library(Core STATIC "${CORE_INCLUDES}")
target_link_libraries(Core PRIVATE armadillo fmt::fmt)
target_include_directories(Core PUBLIC include Lib/fmt/include)

# Ensure the script exists and is executable
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/generate_typelist.sh")
    message(FATAL_ERROR "generate_typelist.sh not found.")
endif()

# Set the paths for the script
set(OPERATIONS_DIR "${CMAKE_SOURCE_DIR}/Core/include/nodes/operations")
set(NEW_TYPELIST_FILE "${CMAKE_SOURCE_DIR}/Core/include/network/OperationTypelist_new.hpp")
set(ORIGINAL_TYPELIST_FILE "${CMAKE_SOURCE_DIR}/Core/include/network/OperationTypelist.hpp")

# Custom target to generate the typelist
add_custom_target(
    generate_typelist
    COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/cmake/generate_typelist.sh ${OPERATIONS_DIR} ${NEW_TYPELIST_FILE} ${ORIGINAL_TYPELIST_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating typelist..."
)

# Custom target to generate the typelist
add_custom_target(
    accept_typelist
    COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/cmake/accept_typelist.sh ${NEW_TYPELIST_FILE} ${ORIGINAL_TYPELIST_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Accepting typelist..."
)

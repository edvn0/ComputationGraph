#!/bin/bash

# Check if the operation name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <OperationName>"
    exit 1
fi

OPERATION_NAME=$1
OPERATIONS_DIR="Core/include/nodes/operations"
SRC_DIR="Core/src/nodes/operations"
CMAKE_FILE="Core/CMakeLists.txt"

# Copy template files
cp "${OPERATIONS_DIR}/ReLUOperation.hpp" "${OPERATIONS_DIR}/${OPERATION_NAME}Operation.hpp"
cp "${SRC_DIR}/ReLUOperation.cpp" "${SRC_DIR}/${OPERATION_NAME}Operation.cpp"

# Rename within files
sed -i "s/ReLUOperation/${OPERATION_NAME}Operation/g" "${OPERATIONS_DIR}/${OPERATION_NAME}Operation.hpp"
sed -i "s/ReLUOperation/${OPERATION_NAME}Operation/g" "${SRC_DIR}/${OPERATION_NAME}Operation.cpp"

# Update CMakeLists.txt
# Find the line number where CORE_INCLUDES ends
line_num=$(grep -n "set(CORE_INCLUDES" "${CMAKE_FILE}" | cut -d : -f 1)

# Insert new paths in CORE_INCLUDES
sed -i "${line_num}a \    include/nodes/operations/${OPERATION_NAME}Operation.hpp" "${CMAKE_FILE}"
sed -i "${line_num}a \    src/nodes/operations/${OPERATION_NAME}Operation.cpp" "${CMAKE_FILE}"

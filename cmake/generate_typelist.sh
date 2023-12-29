#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <operations_dir> <new_typelist_file> <original_typelist_file>"
    exit 1
}

# Check if sufficient arguments are provided
if [ "$#" -ne 3 ]; then
    usage
fi

# Assign arguments to variables
operations_dir=$1
new_typelist_file=$2
original_typelist_file=$3

# Check if operations directory exists and is a directory
if [ ! -d "$operations_dir" ]; then
    echo "Error: Operations directory '$operations_dir' does not exist or is not a directory."
    usage
fi

# Function to generate a new typelist file
generate_typelist() {
    local class_name_regex="class ([a-zA-Z0-9_]+)"
    local typelist_start="using OperationTypes = Typelist<"
    local file_list=($(find "$operations_dir" -name "*.hpp"))
    local total_files=${#file_list[@]}
    local count=0

    # Check if there are any .hpp files in the directory
    if [ $total_files -eq 0 ]; then
        echo "Error: No .hpp files found in the operations directory."
        exit 1
    fi

    # Write initial content (pragma, includes, namespace)
    echo "#pragma once" > "$new_typelist_file"
    echo "" >> "$new_typelist_file"
    echo "#include \"core/Typelist.hpp\"" >> "$new_typelist_file"

    # Include each operation header file
    for file in "${file_list[@]}"; do
        local filename=$(basename "$file")
        echo "#include \"nodes/operations/$filename\"" >> "$new_typelist_file"
    done

    echo "" >> "$new_typelist_file"
    echo "namespace Core {" >> "$new_typelist_file"
    echo "" >> "$new_typelist_file"
    echo "$typelist_start" >> "$new_typelist_file"

    # Loop over .hpp files and extract class names
    for file in "${file_list[@]}"; do
        ((count++))
        if [[ -f $file ]]; then
            while IFS= read -r line; do
                if [[ $line =~ $class_name_regex ]]; then
                    # Append class name to the typelist
                    if [[ $count -eq $total_files ]]; then
                        echo "                                ${BASH_REMATCH[1]}" >> "$new_typelist_file"
                    else
                        echo "                                ${BASH_REMATCH[1]}," >> "$new_typelist_file"
                    fi
                fi
            done < "$file"
        fi
    done

    # Finish the new typelist
    echo "                                >;" >> "$new_typelist_file"
    echo "" >> "$new_typelist_file"
    echo "} // namespace Core" >> "$new_typelist_file"
}

# Generate the typelist
generate_typelist

# Output message
echo "New typelist generated at $new_typelist_file."

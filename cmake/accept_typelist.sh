move_file() {
    mv "$1" "$2"
    echo "New typelist generated at $2."
}

# Check if the number of arguments is correct
if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    echo "Usage: ./accept_typelist.sh <new_typelist_file> <original_typelist_file>"
    exit 1
fi

# Check if the first argument is a file and that it exists
if [ ! -f "$1" ]; then
    echo "Error: From-typelist file is not a file."
    echo "Usage: ./accept_typelist.sh <new_typelist_file> <original_typelist_file>"
    exit 1
fi

# Call the function with the file paths as arguments
move_file "$1" "$2"

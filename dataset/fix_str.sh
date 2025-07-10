#!/bin/bash

# Usage: ./fix_str.sh <target-directory>

TARGET_DIR="$1"

if [[ -z "$TARGET_DIR" ]]; then
    echo "Usage: $0 <target-directory>"
    exit 1
fi

flatten_once() {
    local dir="$1"

    # Check contents of the directory
    entries=("$dir"/*)
    if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
        inner="${entries[0]}"
        echo "Moving: $inner  -->  $dir"
        
        # Move files from the inner directory to the parent
        mv "$inner"/* "$dir"/

        # Remove the empty inner directory
        rmdir "$inner"

        # Repeat the process in case there's more nesting
        flatten_once "$dir"
    fi
}

export -f flatten_once

# Traverse all subdirectories and apply flattening
find "$TARGET_DIR" -type d -exec bash -c 'flatten_once "$0"' {} \;

echo "All directories have been flattened."

#!/bin/bash

# Set default values for input directory and output file
input_dir="."
output_file="output.txt"

# Function to display help message
usage() {
  echo "Usage: $0 [input_dir] [--output=output_file]"
  echo "  input_dir:    The directory to process. Defaults to the current directory (.)."
  echo "  --output=output_file: The output file to generate. Defaults to output.txt."
  echo "  Example: $0 ./budapp --output=all_code.txt"
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output=*) output_file="${1#*=}"; shift ;;
    -h|--help) usage ;;
    -*) echo "Error: Invalid option: $1"; usage ;;
    *) if [[ -d "$1" ]]; then input_dir="$1"; shift; else echo "Error: Invalid input directory: $1"; usage; fi ;;
  esac
done

# Check if the input directory exists
if [[ ! -d "$input_dir" ]]; then echo "Error: Input directory '$input_dir' does not exist."; usage; fi

# Check if output file exists and prompt if exists
if [[ -f "$output_file" ]]; then read -r -p "Output file '$output_file' already exists. Do you want to overwrite it? (y/n): " response; if [[ "$response" != "y" && "$response" != "Y" ]]; then echo "Aborting script."; exit 0; fi; fi

# Initialize the output file to an empty file
> "$output_file"

# Recursively find all Python files, sort them by path, and append contents, excluding migrations dir
find "$input_dir" -type f -name "*.py" ! -path "$input_dir/migrations/*" ! -path "$input_dir/*/migrations/*" | sort | while IFS= read -r file; do
  echo "File: $file" >> "$output_file"
  cat "$file" | sed '/^"""/d' | sed '/^#  /d' >> "$output_file"
  echo "" >> "$output_file"
done

# Remove lines starting with '#' from the output file
sed -i '/^#/d' "$output_file"

# Notify user about the result
echo "Single file generated successfully to: '$output_file' with comments and docstrings removed and migrations excluded."

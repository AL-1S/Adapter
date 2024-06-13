import os

def count_lines_in_expanded_trans_files(directory):
    # Initialize a counter for total lines
    total_lines = 0
    
    # Iterate through files in the specified directory
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith("expanded.trans.txt"):
                filepath = os.path.join(root, filename)
                try:
                    # Count the lines in the file
                    with open(filepath, "r", encoding="utf-8") as file:
                        lines = file.readlines()
                        total_lines += len(lines)
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")
    
    return total_lines

# Specify the directory to search
directory_to_search = "/Work21/2024/tempuser/dataset/LibriSpeech/train-clean-100"

# Get the total lines in expanded.trans files
total_lines_in_expanded_trans = count_lines_in_expanded_trans_files(directory_to_search)

print(f"Total lines in expanded.trans files: {total_lines_in_expanded_trans}")

# 指定原始文件路径和结果文件路径
original_file_path = '/Work21/2024/tempuser/preprocess/invalid_punctuated_files.txt'
result_file_path = '/Work21/2024/tempuser/preprocess/invalid_files.txt'

# 要移除的路径前缀
prefix_to_remove = '/Work21/2024/tempuser/dataset/LibriSpeech/train-clean-100/'

# 打开原始文件和结果文件
with open(original_file_path, 'r', encoding='utf-8') as original_file, \
     open(result_file_path, 'w', encoding='utf-8') as result_file:
    # 逐行读取原始文件
    for line in original_file:
        # 移除每一行的指定前缀
        new_line = line.strip().replace(prefix_to_remove, '')
        # 将处理后的行写入结果文件
        result_file.write(new_line + '\n')

# 指定原始文件路径和结果文件路径
original_file_path = '/Work21/2024/tempuser/preprocess/missing_files.txt'
result_file_path = '/Work21/2024/tempuser/preprocess/miss_files.txt'

# 打开原始文件和结果文件
with open(original_file_path, 'r', encoding='utf-8') as original_file, \
     open(result_file_path, 'w', encoding='utf-8') as result_file:
    # 逐行读取原始文件
    for line in original_file:
        # 检查每一行是否包含 '/'
        if '/' not in line:
            # 如果不包含，则将该行写入结果文件
            result_file.write(line)

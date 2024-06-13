import os

# 设置数据集路径
dataset_path = '/Work21/2024/tempuser/dataset/LibriSpeech/train-clean-100'

# 初始化列表以存储文件路径
missing_files = []
invalid_files = []

# 遍历目录
for root, dirs, files in os.walk(dataset_path):
    # 检查是否有以 'punctuated.trans' 结尾的 txt 文件
    punctuated_file_exists = any(file.endswith('punctuated.trans.txt') for file in files)
    
    # 如果不存在，将根目录添加到缺失文件列表
    if not punctuated_file_exists:
        missing_files.append(root)
    else:
        # 如果存在，检查文件中的每一行是否以数字开头或是否存在空行
        for file in files:
            if file.endswith('punctuated.trans.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    stripped_line = line.strip()
                    # 如果行为空或行的第一个字符不是数字
                    if not stripped_line or not stripped_line[0].isdigit():
                        invalid_files.append(file_path)
                        break

# 将缺失的文件路径写入 txt 文件 1
with open('missing_punctuated_files.txt', 'w', encoding='utf-8') as f:
    for path in missing_files:
        f.write(path + '\n')

# 将无效的文件路径写入 txt 文件 2
with open('invalid_punctuated_files.txt', 'w', encoding='utf-8') as f:
    for path in invalid_files:
        f.write(path + '\n')

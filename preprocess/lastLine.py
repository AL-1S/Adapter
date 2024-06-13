import os

# 设置文件夹路径
folder_path = '/Work21/2024/tempuser/dataset/LibriSpeech/train-clean-100'

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('expanded.trans.txt'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r+', encoding='utf-8') as f:
                lines = f.readlines()
                # 确保文件至少有一行
                if lines:
                    # 删除最后一行的内容
                    lines[-1] = lines[-1][:-1]
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()

print("所有文件的最后一行已被处理。")

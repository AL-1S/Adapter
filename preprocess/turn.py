from openai import OpenAI
import os

# 指定数据集路径
dataset_path = '/Work21/2024/tempuser/dataset/LibriSpeech/train-clean-100'

# 创建 OpenAI 客户端实例
client = OpenAI(
    # 将这里换成你在便携AI聚合API后台生成的令牌
    api_key="sk-GvSo7Bpef9pgI4QU642aB1Ae57A940D99509Bc459d8a72E5",
    # 这里将官方的接口访问地址替换成便携AI聚合API的入口地址
    base_url="https://api.bianxieai.com/v1"
)

# 初始化会话消息
session_messages = [
    {
        "role": "user",
        "content": "请给句子添加标点，并修正字母的大小写，使句子变为正常的句子。注意，你的回答只能是该句子，不能包含其他任何东西。"
    }
]

# 遍历数据集目录
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.trans.txt'):
            # 构建完整的文件路径
            file_path = os.path.join(root, file)
            # 输出文件路径
            print(f"Processing file: {file_path}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 准备存储处理结果的列表
            punctuated_lines = []

            # 初始化进度计数器
            processed_count = 0
            total_lines = len(lines)
            
            for line in lines:
                # 检查是否为空行
                if not line.strip():
                    continue  # 如果是空行，则跳过
                # 分离编号和文本
                line_num, text = line.strip().split(' ', 1)
                
                # 重置会话消息，只包含当前需要处理的句子
                session_messages = [
                    {
                        "role": "user",
                        "content": "请给句子添加标点，并修正字母的大小写，使句子变为正常的句子。"
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ]
                
                # 使用 OpenAI 客户端创建聊天完成
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=session_messages
                )
                # 获取处理后的文本
                punctuated_text = completion.choices[0].message.content
                # 确保处理后的文本在同一行且编号后只有一个空格
                punctuated_line = f"{line_num} {punctuated_text}"
                # 添加到结果列表
                punctuated_lines.append(punctuated_line)

                # 更新进度计数器
                processed_count += 1
                print(f"Processed {processed_count}/{total_lines} lines.")
            
            # 将处理结果写入新文件
            with open(file_path.replace('.trans.txt', '_punctuated.trans.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(punctuated_lines))



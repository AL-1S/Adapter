import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型路径
model_path = '/Work21/2024/tempuser/opt-125m'

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 指定数据集路径
dataset_path = '/Work21/2024/tempuser/dataset/LibriSpeech/train-clean-100/26/496'

# 遍历数据集目录
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('punctuated.trans.txt'):
            # 构建完整的文件路径
            file_path = os.path.join(root, file)
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 准备存储推理结果的列表
            expanded_lines = []

            for index, line in enumerate(lines):
                # 分离编号和文本
                line_num, text = line.strip().split(' ', 1)
                # 使用分词器编码文本
                input_ids = tokenizer.encode(text, return_tensors='pt')
                # 计算输入句子的token数量
                input_length = len(input_ids[0])
                # 设置生成句子的最大长度为输入句子长度的2.5倍
                max_length = int(input_length * 2.5)
                # 使用采样技术生成文本
                outputs = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.9, top_k=50, top_p=0.95)
                # 解码生成的文本
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 确保生成的文本在同一行且编号后只有一个空格
                expanded_line = f"{line_num} {generated_text.replace(text, '').strip()}"
                # 添加到结果列表
                expanded_lines.append(expanded_line)
                print(f"已完成 {index + 1}/{len(lines)} 行")  # 显示当前进度
                
            # 将推理结果写入新文件
            with open(file_path.replace('.trans.txt', '_expanded.trans.txt'), 'w', encoding='utf-8') as f:
                for expanded_line in expanded_lines:
                    # 检查是否为空行
                    if expanded_line.strip():
                        # 合并多行文本
                        merged_text = ' '.join(expanded_line.splitlines())
                        f.write(merged_text + '\n')

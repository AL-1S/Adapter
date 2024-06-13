import torch
from transformers import AutoTokenizer, AutoModel

# 指定模型路径
model_path = '/Work21/2024/tempuser/opt-125m'

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 初始化模型
model = AutoModel.from_pretrained(model_path)

# 生成文本
prompt = "And of the last, the bill itself said."

# 将文本转换为输入张量
inputs = tokenizer(prompt, return_tensors='pt')

# 获取词嵌入
with torch.no_grad():
    outputs = model(**inputs)

# 获取嵌入的形状
embedding_shape = outputs.last_hidden_state.shape
print(embedding_shape)

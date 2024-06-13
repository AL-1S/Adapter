from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型路径
model_path = '/Work21/2024/tempuser/opt-125m'

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 生成文本
prompt = "Among the wealthiest of the people, and such people as were unencumbered with trades and business, but of the rest, the generality stayed and seemed to abide the worst, so that."  # 你可以替换这里的文本为你想要扩写的内容

# 计算原始提示的token数量
input_ids = tokenizer.encode(prompt, return_tensors='pt')
original_length = len(input_ids[0])

# 设置max_length为原始长度的2.5倍
max_length = int(original_length * 3)

# 使用束搜索生成文本
outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1, num_beams=5)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印生成的文本
print(generated_text)

from transformers import pipeline

# 创建文本生成pipeline
model_path = '/Work21/2024/tempuser/opt-125m'
generator = pipeline('text-generation', model=model_path)


input_sentence = "Very fond of both daughters, but particularly of Emma"

prompt = "Repeat the following phrase: "+ input_sentence

repeated_sentence = generator(prompt, max_length=200, num_return_sequences=1)

# 打印重复的句子
print(repeated_sentence[0]['generated_text'])




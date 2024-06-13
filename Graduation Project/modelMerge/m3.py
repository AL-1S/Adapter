import os
import torch
from transformers import WhisperModel, WhisperProcessor, OPTForCausalLM, AutoTokenizer
from torch import nn
import torchaudio
import numpy as np


# 加载本地的 Whisper 模型和处理器
local_whisper_path = "/Work21/2024/tempuser/whisper"
whisper_model = WhisperModel.from_pretrained(local_whisper_path)
whisper_processor = WhisperProcessor.from_pretrained(local_whisper_path)

class ModalityAdapter(nn.Module):
    def __init__(self):
        super(ModalityAdapter, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=512, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.bottleneck = nn.Linear(512, 768)  # OPT-125M的隐藏维数为768

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平特征以匹配瓶颈层的输入
        x = self.bottleneck(x)
        return x

# 实例化模态适配器
modality_adapter = ModalityAdapter()

# 加载本地的 OPT-125M 模型和分词器
local_opt_path = "/Work21/2024/tempuser/opt-125m"
opt_model = OPTForCausalLM.from_pretrained(local_opt_path)
tokenizer = AutoTokenizer.from_pretrained(local_opt_path)

# 数据集目录
dataset_dir = "/Work21/2024/tempuser/dataset/LibriSpeech/train-clean-100"

# 初始化进度计数器
total_sentences = sum([len(open(os.path.join(r, file), 'r', encoding='utf-8').readlines()) for r, d, files in os.walk(dataset_dir) for file in files if file.endswith("expanded.trans")])
processed_sentences = 0

# 遍历数据集目录
for subdir, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith("expanded.trans"):
            # 读取标记文本文件
            with open(os.path.join(subdir, file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                # 更新进度
                processed_sentences += 1
                print(f"Processing sentence {processed_sentences} of {total_sentences}")
            
            for line in lines:
                # 分割编号和文本
                audio_id, _, transcript = line.partition(" ")
                audio_file_path = os.path.join(subdir, audio_id + ".flac")
                
                # 使用 torchaudio 加载音频文件并获取波形数据
                waveform, sample_rate = torchaudio.load(audio_file_path)
                
                # 将波形数据转换为浮点数数组
                waveform = waveform.numpy().astype(np.float32)
                
                # 使用 WhisperProcessor 预处理音频波形数据
                audio_input = whisper_processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
                
                # 使用 Whisper 的 encoder 处理预处理后的音频
                with torch.no_grad():
                    # 注意这里我们只调用编码器
                    encoded_audio = whisper_model.encoder(**audio_input).last_hidden_state
                


                
                # 使用模态适配器处理 encoder 的输出
                adapted_features = modality_adapter(encoded_audio)
                
                # 将标记文本转换为 token ids
                labels = tokenizer(transcript, return_tensors="pt").input_ids
                
                # 使用 OPT-125M 模型进行文本生成
                # 这里使用 teacher forcing，即在训练过程中提供正确的输出标签
                input_ids = labels[:, :-1]  # 输入标记，去除最后一个标记
                outputs = opt_model(input_ids=input_ids, labels=labels[:, 1:])  # 预测下一个标记
                
                # 计算损失并进行反向传播
                loss = outputs.loss
                loss.backward()
                
                # 保存模态适配器的权重
                save_path = '/Work21/2024/tempuser/modelMerge/modality_adapter_weights.pth'
                torch.save(modality_adapter.state_dict(), save_path)
                
                print(f"Finished processing sentence {processed_sentences} of {total_sentences}")
            
            print("Finished processing all files.")

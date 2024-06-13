import torch
from transformers import WhisperModel, WhisperFeatureExtractor
import whisper
import numpy as np
import torch.nn.functional as F

# 指定模型所在的本地路径
local_model_path = "/Work21/2024/tempuser/whisper_small"

# 初始化 Whisper 模型和特征提取器
model = WhisperModel.from_pretrained(local_model_path)
feature_extractor = WhisperFeatureExtractor.from_pretrained(local_model_path)
audio_path = "/Work21/2024/tempuser/dataset/LibriSpeech/train-clean-100/7148/7763/7148-7763-0003.flac"

# 加载音频文件并转换为梅尔频谱
audio = whisper.load_audio(audio_path)
audio_len=len(audio)
print(audio.shape)
audio = whisper.pad_or_trim(audio)
print(audio.shape)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
print(mel.shape)
with torch.no_grad():
    mel = mel.unsqueeze(0)
    print(mel.shape)
    encoder_outputs = model.encoder(mel)
    features = encoder_outputs.last_hidden_state
    features_len=int(audio_len/320)
    trimmed_features = features[:, :features_len]
    print(trimmed_features.shape)  
    padding_value = 1 
    total_padding = 100
    padding = (0, 0, 0, total_padding)
    padded_label = F.pad(trimmed_features, padding, "constant", padding_value)
    

# 打印/返回特征向量
print(padded_label.shape)  
# print(features)  


# # 示例：处理音频文件并获取编码后的隐藏状态
# audio_path = "/Work21/2024/tempuser/dataset/LibriSpeech/train-clean-100/27/123349/27-123349-0000.flac"
# hidden_states = extract_features(audio_path)
# print(hidden_states.shape)


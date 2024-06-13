import os
import pandas as pd
import whisper
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import OPTForCausalLM, AutoTokenizer, WhisperModel, WhisperTokenizer
import torch.nn as nn
from torch.optim import Adam
from typing import Optional, List
import torch.nn.functional as F

class ModalityAdapter(nn.Module):
    def __init__(self, input_dim=384, output_dim=768, bottleneck_dim=512):
        super(ModalityAdapter, self).__init__()
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        # 三个一维卷积层，用于时间长度减少
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 432, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(432, 480, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(480, bottleneck_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        # 特征维度升维到output_dim
        self.up_scale = nn.Linear(bottleneck_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        # 调整形状以匹配线性层的期望输入
        # 将特征维度和时间维度交换
        x = x.contiguous().view(x.size(0), -1, self.bottleneck_dim) # 调整形状
        # 特征维度升维
        x = self.up_scale(x)
        x = x.view(x.size(0), -1, self.output_dim)
        return x

# 数据集类
class LibriSpeechDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio_path = self.dataframe.iloc[idx, 0]
        text = self.dataframe.iloc[idx, 1]
        return audio_path, text
    
# forward子类
class OPTForCausalLMWithEmbeddings(OPTForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if inputs_embeds is not None:
            return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        

# 预处理和加载数据
def load_data(dataset_dir):
    data = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith("expanded.trans.txt"):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        audio_file = os.path.join(root, parts[0] + ".flac")
                        if os.path.exists(audio_file):
                            data.append((audio_file, parts[1]))
    return pd.DataFrame(data, columns=['audio_path', 'text'])

# 提取特征
def extract_features(audio_path,  model_whisper, device):
    audio = whisper.load_audio(audio_path)
    print(audio.shape)
    audio_len=len(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)
    with torch.no_grad():
        mel = mel.unsqueeze(0)
        # print(mel.shape)
        encoder_outputs = model_whisper.encoder(mel)
        features = encoder_outputs.last_hidden_state
        features_len=int(audio_len/320)
        trimmed_features = features[:, :features_len]
    return trimmed_features

# 训练模型
def train(model_whisper, model_adapter, model_with_embeddings,  model_opt, tokenizer_opt, data_loader, device):
    ## criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model_adapter.parameters(), lr=0.001)
    model_whisper.eval()
    model_opt.eval()
    model_adapter.train()

    for epoch in range(1):  
        for i, (audio_paths, texts) in enumerate(data_loader):
            optimizer.zero_grad()
            batch_loss = 0
            for audio_paths, texts in zip(audio_paths, texts):
                koe_features = extract_features(audio_paths,  model_whisper, device)
                print("audio_features的维度:", koe_features.shape)

                adapted_features = model_adapter(koe_features)
                feature_length = adapted_features.size(1)
                print("adapted_features的维度:", adapted_features.shape)
                print("adapted_features的类型:", adapted_features.dtype)
                
                ## inputs_tensor = tokenizer_opt(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                inputs = tokenizer_opt(texts, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs["input_ids"]
                labels = input_ids[:, 1:].contiguous()
                labels = F.pad(labels, pad=(0, 1), value=tokenizer_opt.pad_token_id)  # 右侧填充
                # print(labels.shape)
                labels_padded = F.pad(labels, pad=(0, max(0, feature_length - labels.size(1))), value=-1)
                # inputs = {k: v.to(device) for k, v in inputs.items()}
                # word_embedding = model_opt(**inputs)
                # labels = word_embedding.hidden_states[-1]
                # print("未pad的词嵌入", labels.shape)
                print("填充标签长", labels_padded.shape)
                # labels_length = labels.size(1)
                # padding_value = -1 
                # total_padding = feature_length - labels_length
                # padding = (0, 0, 0, total_padding)
                # padded_labels = F.pad(labels, padding, "constant", padding_value)

                # print("词嵌入的维度:", padded_labels.shape)
                # print("词嵌入的类型:", padded_labels.dtype)

                outputs = model_with_embeddings(inputs_embeds=adapted_features, labels=labels_padded)
                batch_loss += outputs.loss
                # print("shuchu", outputs.shape)
                # batch_loss += outputs.loss
                # labels = labels[:, 1:, :].argmax(-1) 
                # outputs = model_opt(inputs_embeds=adapted_features[:, :-1, :], labels=labels)
                

            batch_loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Step {i+1}, Loss: {batch_loss.item()}")
            torch.save(model_adapter.state_dict(), f'model_adapter_epoch_{epoch+1}.pth')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = "/Work21/2024/tempuser/dataset/LibriSpeech/train-clean-100"

    # 加载模型
    model_whisper = WhisperModel.from_pretrained("/Work21/2024/tempuser/whisper",output_hidden_states=True).to(device)
    model_opt = OPTForCausalLM.from_pretrained("/Work21/2024/tempuser/opt-125m",output_hidden_states=True).to(device)
    tokenizer_opt = AutoTokenizer.from_pretrained("/Work21/2024/tempuser/opt-125m")
    model_adapter = ModalityAdapter().to(device)
    model_with_embeddings = OPTForCausalLMWithEmbeddings(config=model_opt.config)
    model_with_embeddings.load_state_dict(model_opt.state_dict())
    model_with_embeddings = model_with_embeddings.to(device)

    # 加载数据
    df = load_data(dataset_dir)
    dataset = LibriSpeechDataset(df)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 训练模型
    train(model_whisper, model_adapter, model_with_embeddings,  model_opt, tokenizer_opt,  data_loader, device)

if __name__ == "__main__":
    main()
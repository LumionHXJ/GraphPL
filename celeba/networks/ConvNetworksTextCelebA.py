import random
import torch
import torch.nn as nn

from celeba.networks.FeatureExtractorText import FeatureExtractorText
from celeba.networks.FeatureCompressor import LinearFeatureCompressor

from ..constants import styles, modality_dims
DIM_TEXT = [16, 32, 64, 128, 256, 512, 512] # 7 layers
class_dim = modality_dims['text']
style_dim = styles['text']

class EncoderText(nn.Module):
    def __init__(self, flags):
        super(EncoderText, self).__init__();
        self.feature_extractor = FeatureExtractorText(flags.num_features, DIM_TEXT)
        self.feature_compressor = LinearFeatureCompressor(DIM_TEXT[-1],
                                                          style_dim,
                                                          class_dim)

    def forward(self, x_text):
        h_text = self.feature_extractor(x_text)
        mu_style, logvar_style, mu_content, logvar_content = self.feature_compressor(h_text);
        return mu_style, logvar_style, mu_content, logvar_content, h_text;


class DecoderText(nn.Module):
    teacher_forcing = True
    def __init__(self, flags, padding_index=55):
        super(DecoderText, self).__init__()
        self.flags = flags
        self.style_dim = style_dim
        self.max_seq_len = flags.len_sequence  # The maximum length of the output sequence
        self.embedding = nn.Linear(flags.num_features, DIM_TEXT[-1])
        self.padding_index = padding_index
        self.teacher_forcing_rate = flags.teacher_forcing
        self.num_layers = 4
        self.fc_h = nn.Linear(style_dim+class_dim, self.num_layers * DIM_TEXT[-1]) # hidden state
        self.fc_c = nn.Linear(style_dim+class_dim, self.num_layers * DIM_TEXT[-1]) # cell state
        self.lstm = nn.LSTM(input_size=DIM_TEXT[-1], 
                            hidden_size=DIM_TEXT[-1], 
                            num_layers=self.num_layers, 
                            batch_first=True)
        self.fc_o = nn.Linear(DIM_TEXT[-1], flags.num_features)
        self.out_act = nn.Softmax(dim=-1)

    def forward(self, style_latent_space, class_latent_space, target_sequence=None):
        if self.style_dim > 0:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        bs = z.size(0)
        # Initialize the hidden state and cell state
        hidden = (self.fc_h(z).view(bs, self.num_layers, -1).permute(1,0,2).contiguous(),
                  self.fc_c(z).view(bs, self.num_layers, -1).permute(1,0,2).contiguous())

        if target_sequence is not None and (random.uniform(0, 1) < self.teacher_forcing_rate or not self.training): # ! teacher forcing rate?
            # Shift right while using teacher forcing
            lstm_input = torch.cat((torch.zeros(bs, 1, DIM_TEXT[-1]).to(z.device), 
                                    self.embedding(target_sequence[:, :-1])), dim=1) # B, L, D      
            lstm_out, (hn, cn) = self.lstm(lstm_input, hidden)
            # weight tying -> probs
            out = self.fc_o(lstm_out) # B, L, V
            log_prob = self.out_act(out)
        else: 
            # auto regressive manner during inference: 
            # ! ignoring [space] at first place
            # ! output discrete space instead of continuous
            log_prob = []
            lstm_input = torch.zeros(bs, 1, DIM_TEXT[-1]).to(z.device)
            for t in range(self.max_seq_len):
                lstm_out, hidden = self.lstm(lstm_input, hidden)
                out = self.fc_o(lstm_out)
                if t == 0: # no padding token at beginning
                    out[..., self.padding_index] = -1e10
                out = torch.where(out == out.max(dim=-1, keepdim=True)[0], 1.0, 0.0)
                log_prob.append(out)
                lstm_input = self.embedding(out)
            log_prob = torch.cat(log_prob, dim=1) # B, L, V
        return [log_prob]

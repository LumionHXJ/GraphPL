import random
import torch
import torch.nn as nn
import glog as logger
from ..constants import modality_dims
class_dim = modality_dims['text']

class FeatureEncText(nn.Module):
    def __init__(self, dim, num_features):
        super(FeatureEncText, self).__init__()
        self.dim = dim
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, self.dim, kernel_size=1),
            nn.BatchNorm1d(self.dim),
            nn.ReLU(),
            nn.Conv1d(self.dim, self.dim, kernel_size=4, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(self.dim),
            nn.ReLU(),
            nn.Conv1d(self.dim, self.dim, kernel_size=4, stride=2, padding=0, dilation=1),
            nn.BatchNorm1d(self.dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.transpose(-2,-1);
        out = self.conv(x)
        h = out.view(-1, self.dim)
        return h;

class EncoderText(nn.Module):
    def __init__(self, flags, style_dim):
        super(EncoderText, self).__init__()
        logger.info(f"TEXT style dim: {style_dim}, class dim: {class_dim}")
        self.flags = flags
        self.style_dim = style_dim
        self.text_feature_enc = FeatureEncText(flags.dim, flags.num_features);
        if self.style_dim > 0:
            # style
            self.style_mu = nn.Linear(in_features=flags.dim, out_features=style_dim, bias=True)
            self.style_logvar = nn.Linear(in_features=flags.dim, out_features=style_dim, bias=True)
        # class
        self.class_mu = nn.Linear(in_features=flags.dim, out_features=class_dim, bias=True)
        self.class_logvar = nn.Linear(in_features=flags.dim, out_features=class_dim, bias=True)
    def forward(self, x):
        h = self.text_feature_enc(x);
        if self.style_dim > 0:
            return self.style_mu(h), self.style_logvar(h), self.class_mu(h), self.class_logvar(h)
        else:
            return None, None, self.class_mu(h), self.class_logvar(h)

class DecoderText(nn.Module):
    teacher_forcing = True
    def __init__(self, flags, style_dim, padding_index=-2):
        super(DecoderText, self).__init__()
        self.flags = flags
        self.style_dim = style_dim
        self.max_seq_len = flags.len_sequence  # The maximum length of the output sequence
        self.embedding = nn.Linear(flags.num_features, self.flags.dim)
        self.padding_index = padding_index
        self.num_layers = 1
        self.teacher_forcing_rate = flags.teacher_forcing
        self.fc_h = nn.Linear(style_dim+class_dim, self.num_layers * flags.dim) # hidden state
        self.fc_c = nn.Linear(style_dim+class_dim, self.num_layers * flags.dim) # cell state
        self.lstm = nn.LSTM(input_size=flags.dim, 
                            hidden_size=flags.dim, 
                            num_layers=self.num_layers, 
                            batch_first=True)
        self.fc_o = nn.Linear(flags.dim, flags.num_features)
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

        if target_sequence is not None and (random.uniform(0, 1) < self.teacher_forcing_rate or not self.training): # may using teacher forcing in inference
            # Shift right while using teacher forcing
            lstm_input = torch.cat((torch.zeros(bs, 1, self.flags.dim).to(z.device), 
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
            lstm_input = torch.zeros(bs, 1, self.flags.dim).to(z.device)
            for t in range(self.max_seq_len):
                lstm_out, hidden = self.lstm(lstm_input, hidden)
                out = self.fc_o(lstm_out)
                if t == 0:
                    out[..., self.padding_index] = -1e10
                out = torch.where(out == out.max(dim=-1, keepdim=True)[0], 1.0, 0.0)
                log_prob.append(out)
                lstm_input = self.embedding(out)
            log_prob = torch.cat(log_prob, dim=1) # B, L, V
        return [log_prob]
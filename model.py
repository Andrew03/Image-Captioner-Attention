import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import copy

def to_var(x, useCuda=True, volatile=False):
  if useCuda and torch.cuda.is_available():
    x = x.cuda()
  return Variable(x, volatile=volatile)

class EncoderCNN(nn.Module):
  def __init__(self):
    super(EncoderCNN, self).__init__()
    vgg = models.vgg16(pretrained=True).eval()
    for param in vgg.parameters():
      param.requires_grad = False
    self.vgg = nn.Sequential(*(vgg.features[i] for i in range(29)))

  def forward(self, images):
    features = self.vgg(images)
    features_reshaped = features.view(-1, 512, 196)
    features_transposed = features_reshaped.transpose(1, 2)
    return features_transposed

class DecoderRNN(nn.Module):
  def __init__(self, vis_dim, vis_num, embed_dim, hidden_dim, vocab_size, num_layers, dropout=0, useCuda=True):
    super(DecoderRNN, self).__init__()
    self.vis_dim = vis_dim
    self.vis_num = vis_num
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.useCuda = useCuda

    self.init_h = nn.Linear(vis_dim, hidden_dim, bias=False)
    self.init_c = nn.Linear(vis_dim, hidden_dim, bias=False)
    self.attn_vw = nn.Linear(vis_dim, 1, bias=False)
    self.attn_hw = nn.Linear(hidden_dim, 1, bias=False)

    self.embed = nn.Embedding(vocab_size, embed_dim)
    self.lstm = nn.LSTM(vis_dim + embed_dim, hidden_dim)
    self.output = nn.Linear(hidden_dim, vocab_size)

  def _init_hidden(self, features):
    hidden = torch.sum(self.init_h(features), 1) / self.vis_num
    cell = torch.sum(self.init_c(features), 1) / self.vis_num
    return hidden.unsqueeze(0), cell.unsqueeze(0)

  def _compute_attention(self, features, hidden_state):
    att_vw = self.attn_vw(features)
    att_hw = self.attn_hw(hidden_state.squeeze(0)).unsqueeze(1)
    attention = att_vw + att_hw.repeat(1, self.vis_num, 1)
    attention_softmax = F.softmax(attention, dim=1)
    return torch.sum(features * attention_softmax, 1)


  def forward(self, features, captions):
    hidden = self._init_hidden(features)
    word_embeddings = self.embed(captions)
    word_embeddings = word_embeddings.transpose(0, 1)
    word_space = None
    for i, embedding in enumerate(word_embeddings):
      attention = self._compute_attention(features, hidden[0])
      input = torch.cat([attention, embedding], 1).unsqueeze(0)
      out, hidden = self.lstm(input, hidden)
      words = self.output(out)
      word_space = torch.cat([word_space, words], 0) if word_space is not None else words
    return F.log_softmax(word_space, dim=2), F.softmax(word_space, dim=2)

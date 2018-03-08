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
    vgg = models.vgg19(pretrained=True).eval()
    convs = list(vgg.children())[0]
    self.vgg = nn.Sequential(*list(convs)[:-1])
    """
    vgg = models.vgg16(pretrained=True).eval()
    for param in vgg.parameters():
      param.requires_grad = False
    self.vgg = nn.Sequential(*(vgg.features[i] for i in range(29)))
    """

  def forward(self, images):
    features = self.vgg(images)
    features_reshaped = features.view(-1, 512, 196)
    features_transposed = features_reshaped.transpose(1, 2)
    #print(features_transposed.size())
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

    # remember to divide output by vis_num
    self.init_h = nn.Linear(vis_dim, hidden_dim, bias=False)
    self.init_c = nn.Linear(vis_dim, hidden_dim, bias=False)
    # cat features with previous hidden state
    # remember to softmax and multiply and then pass in
    self.attn = nn.Linear(vis_dim + hidden_dim, 1, bias=False)

    self.embed = nn.Embedding(vocab_size, embed_dim)
    self.lstm = nn.LSTM(embed_dim, hidden_dim)
    self.fc_out = nn.Linear(hidden_dim, vocab_size)

    # attention
    # try with biases later
    self.att_vw = nn.Linear(vis_dim, vis_dim, bias=False)
    self.att_hw = nn.Linear(hidden_dim, vis_dim, bias=False)
    self.att_bias = nn.Parameter(torch.zeros(vis_num))
    self.att_w = nn.Linear(self.vis_dim, 1, bias=False)

  def _init_hidden(self, features):
    hidden = self.init_h(features) / self.vis_num
    cell = self.init_c(features) / self.vis_num
    return hidden, cell

  def forward(self, features, captions):
    hiddens = self._init_hidden(features)
    hidden = hiddens
    print(hiddens[0].size())
    word_embeddings = self.embed(captions)
    print(word_embeddings.size())
    outs = []
    for j, i in enumerate(word_embeddings.transpose(0, 1)):
      print(j)
      out, hidden = self.lstm(i.unsqueeze(0), hidden)
      if (len(outs) == 0):
        outs = out
        hiddens = hidden
      else:
        outs = torch.cat((outs, out), 0)
        hiddens = (torch.cat((hiddens[0], hidden[0]), 0),
                   torch.cat((hiddens[1], hidden[1]), 0))
    return outs, hiddens

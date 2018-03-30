import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

def to_var(x, useCuda=True, volatile=False):
  if useCuda and torch.cuda.is_available():
    x = x.cuda()
  return Variable(x, volatile=volatile)

def create_predict_input_captions(captions, useCuda=True):
  if torch.cuda.is_available() and useCuda:
    return Variable(torch.cuda.LongTensor(captions))
  return Variable(torch.LongTensor(captions))
  
  x = Variable(torch.LongTensor(captions)) if type(captions) == int else captions
  return x.cuda() if torch.cuda.is_available() and useCuda else x

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
    self.attn_vw = nn.Linear(vis_dim, 1)
    self.attn_hw = nn.Linear(hidden_dim, 1)

    self.embed = nn.Embedding(vocab_size, embed_dim)
    self.lstm = nn.LSTM(vis_dim + embed_dim, hidden_dim, batch_first=True)
    self.output = nn.Linear(hidden_dim, vocab_size)

  def _init_hidden(self, features):
    hidden = torch.sum(self.init_h(features), 1) / self.vis_num
    cell = torch.sum(self.init_c(features), 1) / self.vis_num
    return hidden.unsqueeze(0), cell.unsqueeze(0)

  def _compute_attention(self, features, hidden_state):
    """
    features: B x vis_num x vis_dim
    hidden_state: (1 x B x hidden_size, 1 x B x hidden_size)
    """
    # add in L1 norm (sum up everything and divide everything by sum
    # B x vis_num x 1
    att_vw = self.attn_vw(features)
    # B x vis_num x 1
    att_hw = self.attn_hw(hidden_state.transpose(0, 1).repeat(1, self.vis_num, 1))
    # B x vis_num x 1
    attention = att_vw + att_hw
    attention_softmax = F.softmax(attention, dim=1)
    # B x vis_dim
    return torch.sum(features * attention_softmax, 1)


  def forward(self, features, captions):
    """
    features: B x vis_num x vis_dim
    captions: B x seq_length
    """
    hidden = self._init_hidden(features)
    word_embeddings = self.embed(captions)
    word_space = None
    lengths = len(captions[0])
    for i in range(lengths):
      embedding = torch.index_select(word_embeddings, 1, Variable(torch.cuda.LongTensor([i])))
      # try using this instead of the whole chunk
      #word_distribution, word_probabilities, hidden = self._forward(features, embedding, hidden)
      attention = self._compute_attention(features, hidden[0]).unsqueeze(1)
      input = torch.cat([attention, embedding], 2)
      out, hidden = self.lstm(input, hidden)
      words = self.output(out)
      word_space = torch.cat([word_space, words], 1) if word_space is not None else words
    word_space = pack_padded_sequence(word_space, [lengths for i in range(len(captions))], batch_first=True)[0]
    return F.log_softmax(word_space, dim=1), F.softmax(word_space, dim=1)

  def sample(self, features, beam_size=1, start_token=0, end_token=1):
    beam_size = 1
    hidden = self._init_hidden(features)
    captions = []
    caption = create_predict_input_captions([start_token], self.useCuda)
    score = 0
    for i in range(20):
      embedding = self.embed(caption)
      attention = self._compute_attention(features, hidden[0])
      input = torch.cat([attention, embedding], 1).unsqueeze(1)
      out, hidden = self.lstm(input, hidden)
      words = self.output(out)
      word_distribution = F.log_softmax(words, dim=2)
      caption_score, caption_indices = word_distribution.topk(beam_size)
      next_caption = caption_indices[0][0].data[0]
      caption = create_predict_input_captions([next_caption], self.useCuda)
      captions.append(next_caption)
      score += caption_score
      if next_caption == end_token:
        break
    return captions, score

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
    """
    for batch in features:
      for feature_vec in batch:
        feature_sum = torch.sum(feature_vec)
        feature_vec /= feature_sum
        feature_sum = torch.sum(feature_vec)
        print(feature_sum)
    """
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
    hidden = self._init_hidden(features)
    completed_phrases = []
    best_phrases = []
    score = 0

    initial_caption = create_predict_input_captions([start_token], self.useCuda)
    embedding = self.embed(initial_caption)
    attention = self._compute_attention(features, hidden[0])
    input = torch.cat([attention, embedding], 1).unsqueeze(1)
    out, hidden = self.lstm(input, hidden)
    words = self.output(out)
    word_scores = F.softmax(words, dim=2)
    top_scores, top_captions = word_scores.topk(beam_size)
    best_phrases = [[top_scores[0][0].data[i], [top_captions[0][0].data[i]]] for i in range(beam_size)]
    next_captions = top_captions.resize(beam_size, 1)
    hidden = (hidden[0].repeat(1, beam_size, 1), hidden[1].repeat(1, beam_size, 1))

    for index in range(20):
      best_candidates = []
      embedding = self.embed(next_captions)
      attention = self._compute_attention(features, hidden[0]).unsqueeze(1)
      input = torch.cat([attention, embedding], 2)
      out, hidden = self.lstm(input, hidden)
      words = self.output(out)
      word_scores = F.softmax(words, dim=2)
      top_scores, top_captions = word_scores.topk(beam_size)
      len_phrases = len(best_phrases[0][1])
      for i in range(len(best_phrases)):
        for j in range(beam_size):
          best_candidates.extend([[best_phrases[i][0] + top_scores[i][0].data[j],
            best_phrases[i][1] + [top_captions[i][0].data[j]],
            i]])
      top_candidates = sorted(best_candidates, key=lambda score_caption: score_caption[0])[-beam_size:]
      temp_candidates = []
      for phrase in top_candidates:
        if phrase[1][-1] == end_token:
          completed_phrases.append([phrase[0] / len(phrase[1]), phrase[1]])
        else:
          temp_candidates.append(phrase)
      top_candidates = temp_candidates
      if len(completed_phrases) >= beam_size:
        return sorted(completed_phrases, key=lambda score_caption: score_caption[0], reverse=True)[:beam_size]
      best_phrases = [[phrase[0], phrase[1]] for phrase in top_candidates]
      next_captions = create_predict_input_captions([[phrase[1][-1]] for phrase in top_candidates], self.useCuda)
      hidden_0 = (torch.stack([hidden[0][0].select(0, phrase[2]) for phrase in top_candidates]).unsqueeze(0))
      hidden_1 = (torch.stack([hidden[1][0].select(0, phrase[2]) for phrase in top_candidates]).unsqueeze(0))
      hidden = (hidden_0, hidden_1)
    return sorted(completed_phrases, key=lambda score_caption: score_caption[0], reverse=True)[:beam_size]

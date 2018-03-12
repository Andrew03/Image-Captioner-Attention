import torch
import model
import pickle
import torchvision.transforms as transforms
from torch.autograd import Variable
from build_vocab import Vocabulary
from batch_data import BatchedData
from batched_data_loader import get_loader

def to_var(x, useCuda=True, volatile=False):
  if torch.cuda.is_available() and useCuda:
    x = x.cuda()
  return Variable(x, volatile=volatile)

transform = transforms.Compose([
  transforms.Resize(256),
  transforms.RandomCrop(224),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

with open("../ImageCaptioner/data/vocab/vocab_occurrence_5.pkl", 'rb') as f1, \
     open("data/batched_data/train_batch_32_size_4000.pkl", 'rb') as f2:
       vocab = pickle.load(f1)
       batched_train_set = pickle.load(f2)
batched_train_loader = get_loader("../ImageCaptioner/data/train2014",
                                  "../ImageCaptioner/data/annotations/captions_train2014.json",
                                  batched_train_set,
                                  vocab,
                                  transform,
                                  shuffle=True,
                                  num_workers=3)

encoder = model.EncoderCNN()
decoder = model.DecoderRNN(512, 196, 512, 512, len(vocab), 1)
if torch.cuda.is_available():
  encoder = encoder.cuda()
  decoder = decoder.cuda()
features = []
for i, (images, captions, lengths, ids) in enumerate(batched_train_loader):
  if i == 1:
    break
  images = to_var(images, volatile=True)
  captions = to_var(captions, volatile=True)
  features = encoder(images)
  print("initial features size: " + str(features.size()))
  out, hidden = decoder(features, captions)
  print(out.size())
  print(hidden.size())

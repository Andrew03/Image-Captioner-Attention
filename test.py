import torch
import model
import pickle
import torchvision.transforms as transforms
from torch.autograd import Variable
from build_vocab import Vocabulary
from batch_data import BatchedData
from batched_data_loader import get_loader
from pycocotools.coco import COCO

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

with open("../ImageCaptioner/data/vocab/vocab_occurrence_5.pkl", 'rb') as f1,\
    open("../ImageCaptioner/data/batched_data/val_batch_1.pkl", "rb") as f2:
       vocab = pickle.load(f1)
       batched_val_set = pickle.load(f2)
coco_caps = COCO("../ImageCaptioner/data/annotations/captions_val2014.json")
batched_val_loader = get_loader("../ImageCaptioner/data/val2014",
                                  "../ImageCaptioner/data/annotations/captions_val2014.json",
                                  batched_val_set,
                                  vocab,
                                  transform,
                                  shuffle=True,
                                  num_workers=3)

encoder = model.EncoderCNN()
decoder = model.DecoderRNN(512, 196, 512, 512, len(vocab), 1)
if torch.cuda.is_available():
  encoder = encoder.cuda()
  decoder = decoder.cuda()

checkpoint = torch.load("noNorm/model_batch_100_dims_512x512_lr_0.0001/checkpoint_25.pt")
decoder.load_state_dict(checkpoint['state_dict'])
checkpoint = None
torch.cuda.empty_cache()

for i, (images, captions, lengths, ids) in enumerate(batched_val_loader):
  if i == 1:
    break
  print("actual captions are: ")
  annIds = coco_caps.getAnnIds(imgIds=ids)
  anns = coco_caps.loadAnns(annIds)
  for ann in anns:
    print(ann['caption'])
  images = to_var(images, volatile=True)
  captions = to_var(captions, volatile=True)
  features = encoder(images)
  caption, _ = decoder.sample(features)
  print(caption)

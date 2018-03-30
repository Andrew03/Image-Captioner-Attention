import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
import argparse
from torch.nn.utils.rnn import pack_padded_sequence
from pycocotools.coco import COCO
import model
from build_vocab import Vocabulary
from batch_data import BatchedData
from batched_data_loader import get_loader

def to_var(x, useCuda=True, volatile=False):
  if torch.cuda.is_available() and useCuda:
    x = x.cuda()
  return Variable(x, volatile=volatile)

def caption_id_to_string(caption, vocab):
  output = ""
  for word in caption:
    output += vocab(word) + " "
  return output

def main(args):
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])
  ])

  with open(args.vocab_path, "rb") as f1, \
    open(args.batched_file_path, "rb") as f2:
      vocab = pickle.load(f1)
      batched_val_set = pickle.load(f2)

  coco_caps = COCO(args.caption_path)
  batched_val_loader = get_loader(args.image_dir,
                                  args.caption_path,
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

  checkpoint = torch.load(args.load_checkpoint)
  decoder.load_state_dict(checkpoint["state_dict"])
  checkpoint = None
  torch.cuda.empty_cache()

  for i, (images, captions, lengths, ids) in enumerate(batched_val_loader):
    if i == args.num_runs:
      break
    print("\nactual captions for batch " + str(i) + " are: ")
    annIds = coco_caps.getAnnIds(imgIds=ids)
    anns = coco_caps.loadAnns(annIds)
    for ann in anns:
      print(ann["caption"])
    images = to_var(images, volatile=True)
    captions = to_var(captions, volatile=True)
    features = encoder(images)
    caption, _ = decoder.sample(features, args.beam_size)
    print("predicted captions are: ")
    print(caption_id_to_string(caption, vocab))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--caption_path', type=str,
                      default='../ImageCaptioner/data/annotations/captions_val2014.json',
                      help='Path for annotation file. Default value of ../ImageCaptioner/data/annotations/captions_val2014.json')
  parser.add_argument('--image_dir', type=str,
                      default='../ImageCaptioner/data/val2014',
                      help='Path to image directory. Default value of ../ImageCaptioner/data/val2014')
  parser.add_argument('--batched_file_path', type=str,
                      default='../ImageCaptioner/data/batched_data/val_batch_1.pkl',
                      help='Path to batched data file. Default value of ../ImageCaptioner/data/batched_data/val_batch_1.pkl')
  parser.add_argument('--vocab_path', type=str,
                      default='../ImageCaptioner/data/vocab/vocab_occurrence_5.pkl',
                      help='Path to vocab. Default value of ../ImageCaptioner/data/vocab/vocab_occurrence_5.pkl')
  parser.add_argument('--embedding_dim', type=int,
                      default=512,
                      help='Size of the embedding layer. Default value of 512')
  parser.add_argument('--hidden_dim', type=int,
                      default=512,
                      help='Size of the hidden layer. Default value of 512')
  parser.add_argument('--num_layers', type=int,
                      default=1,
                      help='Number of layers in the lstm. Default value of 1')
  parser.add_argument('-disable_cuda', action='store_true',
                      default=False,
                      help='Set if cuda should not be used')
  parser.add_argument('--load_checkpoint', type=str,
                      required=True,
                      help='Saved checkpoint file name. Required')
  parser.add_argument('--num_runs', type=int,
                      default=10,
                      help='Number of runs to go through. Default value of 5')
  parser.add_argument('--beam_size', type=int,
                      default=10,
                      help='Size of beam to generate with. Default value of 5')

  args = parser.parse_args()
  main(args)

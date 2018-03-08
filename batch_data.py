import nltk
import pickle
import argparse
import random
import math
from collections import Counter
from pycocotools.coco import COCO


class BatchedData(object):
  def __init__(self, batch_size):
    self.batched_data = []
    self.index = 0
    self.batch_size = batch_size

  def add_batch(self, batch):
    if len(batch) == self.batch_size:
      self.batched_data.append(batch)
    else:
      print("not the correct size batch!")

  def __call__(self, index):
    if not index < len(self.batched_data):
      return []
    return self.batched_data[index]

  def __len__(self):
    return len(self.batched_data)
    
def batch_data(caption_path, batch_size):
  coco = COCO(caption_path)
  ids = coco.anns.keys()
  batched_set = {}
  for i, id in enumerate(ids):
    caption = str(coco.anns[id]['caption'])
    # accounting for <SOS> and <EOS> tokens
    caption_len = len(nltk.tokenize.word_tokenize(caption.lower())) + 2
    if caption_len not in batched_set.keys():
      batched_set[caption_len] = []
    batched_set[caption_len].append(id)

    if i % 1000 == 0:
      print("Examined [%d/%d] captions" %(i, len(ids)))

  batched_data = BatchedData(batch_size)
  curr_size = 0
  for i in batched_set.keys():
    if len(batched_set[i]) >= batch_size:
      batch = batched_set[i]
      random.shuffle(batch)
      for j in range(math.floor(len(batch) / batch_size)):
        if args.max_size is None or curr_size < args.max_size:
          batched_data.add_batch(batch[batch_size * j : batch_size * (j+1)])
          curr_size += 1
  random.shuffle(batched_data.batched_data)
  return batched_data

def main(args):
  batched_data = batch_data(caption_path=args.caption_path,
                           batch_size=args.batch_size)
  save_path = args.save_path
  with open(save_path, 'wb') as f:
    pickle.dump(batched_data, f)
  print("Total batched data set size: %d" %len(batched_data))
  print("Saved the batched data set at '%s'" %save_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--caption_path', type=str,
                      default='../ImageCaptioner/data/annotations/captions_train2014.json',
                      help='Path for annotation file. Default value of ../ImageCaptioner/data/annotations/captions_train2014.json')
  parser.add_argument('--save_path', type=str,
                      required=True,
                      help='Path to save batched data')
  parser.add_argument('--batch_size', type=int,
                      default=32,
                      help='Size of a batch. Default value of 32')
  parser.add_argument('--max_size', type=int,
                      default=None,
                      help='Maximum number of batches in the batched data set. Defaults to no maximum')
  args = parser.parse_args()
  main(args)

import torch
import torch.utils.data as data
import os
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO

class BatchedCocoDataset(data.Dataset):
  def __init__(self, image_path, caption_path, batched_captions, vocab, transform=None):
    self.image_path = image_path
    self.coco = COCO(caption_path)
    self.batched_captions = batched_captions
    self.vocab = vocab
    self.transform = transform

  def __getitem__(self, index):
    images = []
    captions = []
    lengths = []
    img_ids = []
    for ann_id in self.batched_captions(index):
      caption = self.coco.anns[ann_id]['caption']
      img_id = self.coco.anns[ann_id]['image_id']
      path = self.coco.loadImgs(img_id)[0]['file_name']

      image = Image.open(os.path.join(self.image_path, path)).convert('RGB')
      if self.transform is not None:
        image = self.transform(image)
      images.append(image)
      
      tokens = nltk.tokenize.word_tokenize(str(caption).lower())
      caption = [self.vocab('<SOS>')] + [self.vocab(token) for token in tokens] + [self.vocab('<EOS>')]
      captions.append(caption)
      lengths.append(len(caption))
      img_ids.append(img_id)
    return torch.stack(images, 0), torch.LongTensor(captions), lengths, img_ids

  def __len__(self):
    return len(self.batched_captions)

def collate_fn(data):
  images, captions, lengths, img_ids = zip(*data)
  return images[0], captions[0], lengths[0], img_ids[0]

def get_loader(image_path, caption_path, batched_captions, vocab, transform, shuffle, num_workers):
  dataset = BatchedCocoDataset(image_path=image_path,
                                caption_path=caption_path,
                                batched_captions=batched_captions,
                                vocab=vocab,
                                transform=transform)

  return torch.utils.data.DataLoader(dataset=dataset,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn)

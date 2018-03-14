import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.autograd as autograd
import pickle
import argparse
import os
import smtplib
import getpass
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')
import model
from tqdm import tqdm
tqdm.monitor_interval = 0
from torch.nn.utils.rnn import pack_padded_sequence
from build_vocab import Vocabulary
from batch_data import BatchedData
from batched_data_loader import get_loader

def to_var(x, useCuda=True, volatile=False):
  if torch.cuda.is_available() and useCuda:
    x = x.cuda()
  return autograd.Variable(x, volatile=volatile)

def evaluate(images, captions, encoder_cnn, decoder_rnn, loss_function, useCuda=True, volatile=False):
  images = to_var(images, useCuda, volatile)
  features = encoder_cnn(images)
  inputs = to_var(captions, useCuda, volatile)[:, :-1]
  targets = to_var(captions[:, 1:], useCuda, volatile)
  len_targets = len(targets[0])
  targets = pack_padded_sequence(targets, [len_targets for i in range(len(captions))], batch_first=True)[0]
  predictions, _ = decoder_rnn(features, inputs)
  predictions = pack_padded_sequence(predictions, [len(predictions[i]) for i in range(len(predictions))], batch_first=True)[0]
  loss = loss_function(predictions, targets)
  return loss

def train(images, captions, encoder_cnn, decoder_rnn, loss_function, optimizer, useCuda):
  loss = evaluate(images, captions, encoder_cnn, decoder_rnn, loss_function, useCuda)
  loss.backward()
  optimizer.step()
  return loss

def validate(val_loader, encoder_cnn, decoder_rnn, loss_function, useCuda):
  sum_loss = 0
  for i, (images, captions, lengths, ids) in enumerate(val_loader, 1):
    loss = evaluate(images, captions, encoder_cnn, decoder_rnn, loss_function, useCuda, volatile=False)
    sum_loss += loss.data.select(0, 0)
    if i == 100:
      break
  return sum_loss / 100

def validate_full(val_loader, encoder_cnn, decoder_rnn, loss_function, useCuda, epoch, num_epochs):
  decoder_rnn = decoder_rnn.copy()
  progress_bar = tqdm(iterable=val_loader, desc='Epoch [%i/%i] (Val)' %(epoch, num_epochs), position=1)
  sum_loss = 0
  for i, (images, captions, lengths, ids) in enumerate(progress_bar):
    loss = evaluate(images, captions, encoder_cnn, decoder_rnn, loss_function, useCuda, volatile=False)
    sum_loss += loss.data.select(0, 0)
    progress_bar.set_postfix(loss=sum_loss / (i if i > 0 else 1))
  return sum_loss / len(val_loader)

def main(args):
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])
  ])

  useCuda = not args.disable_cuda

  with open(args.vocab_path, 'rb') as vocab_path, \
      open(args.batched_train_path, 'rb') as batched_train_path, \
      open(args.batched_val_path, 'rb') as batched_val_path:
    vocab = pickle.load(vocab_path)
    batched_train_set = pickle.load(batched_train_path)
    batched_val_set = pickle.load(batched_val_path)

  batched_train_loader = get_loader(args.train_image_dir, args.train_caption_path, batched_train_set, vocab, transform, shuffle=True, num_workers=3)
  batched_val_loader = get_loader(args.val_image_dir, args.val_caption_path, batched_val_set, vocab, transform, shuffle=True, num_workers=1)
  batched_val_loader_full = get_loader(args.val_image_dir, args.val_caption_path, batched_val_set, vocab, transform, shuffle=True, num_workers=1)

  encoder_cnn = model.EncoderCNN()
  decoder_rnn = model.DecoderRNN(512, 196, args.embedding_dim, args.hidden_dim, len(vocab), args.num_layers, args.dropout, useCuda=useCuda)
  if torch.cuda.is_available() and useCuda:
    encoder_cnn.cuda()
    decoder_rnn.cuda()
  loss_function = nn.NLLLoss()
  params = list(decoder_rnn.parameters())
  optimizer = optim.Adam(params, lr=args.lr)

  output_train_file = open(args.output_train_name, 'w')
  output_val_file = open(args.output_val_name, 'w')
  start_epoch = 0

  if args.load_checkpoint is not None:
    checkpoint = torch.load(args.load_checkpoint) if useCuda else torch.load(args.load_checkpoint, map_location=lambda storage, loc: storage)
    print("loading from checkpoint " + str(args.load_checkpoint))
    start_epoch = checkpoint['epoch']
    decoder_rnn.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None
    torch.cuda.empty_cache()

  for epoch in range(start_epoch, args.num_epochs):
    progress_bar = tqdm(iterable=batched_train_loader, desc='Epoch [%i/%i] (Train)' %(epoch, args.num_epochs))
    train_sum_loss = 0
    for i, (images, captions, lengths, ids) in enumerate(progress_bar, 1):
      loss = train(images, captions, encoder_cnn, decoder_rnn, loss_function, optimizer, useCuda)
      # tqdm.write(str(loss.data.select(0, 0)))
      train_sum_loss += loss.data.select(0, 0)
      progress_bar.set_postfix(loss=train_sum_loss/((i % 100) + 1))
      if i % 100 == 0:
        output_train_file.write("%d, %5.4f\n" %(epoch * len(batched_train_loader) + i, train_sum_loss / 100))
        train_sum_loss = 0
        if i % 1000 == 0:
          temp_loss = validate(batched_val_loader, encoder_cnn, decoder_rnn, loss_function, useCuda)
          output_val_file.write("%d, %5.4f\n" %(i, temp_loss))
    # end of batch
    output_train_file.write("%d, %5.4f\n" %((epoch + 1) * len(batched_train_loader), train_sum_loss / len(batched_train_loader) / 100))

    val_sum_loss = 0
    for i, (images, captions, lengths, ids) in enumerate(batched_val_loader_full, 1):
      loss = evaluate(images, captions, encoder_cnn, decoder_rnn, loss_function, optimizer, useCuda)
      val_sum_loss += loss.data.select(0, 0)
      progress_bar.set_postfix(loss=val_sum_loss/i)
    output_val_file.write("%d, %5.4f\n" %((epoch + 1) * len(batched_train_loader), val_sum_loss))

    torch.save({'epoch': epoch + 1,
                'state_dict': decoder_rnn.state_dict(),
                'optimizer': optimizer.state_dict()},
                "checkpoint_" + str(epoch + 1) + ".pt")

  output_train_file.close()
  output_val_file.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_caption_path', type=str,
                      default='../ImageCaptioner/data/annotations/captions_train2014.json',
                      help='Path for train annotation file. Default value of ../ImageCaptioner/data/annotations/captions_train2014.json')
  parser.add_argument('--train_image_dir', type=str,
                      default='../ImageCaptioner/data/train2014',
                      help='Path to train image directory. Default value of ../ImageCaptioner/data/train2014')
  parser.add_argument('--batched_train_path', type=str,
                      default='./data/batched_data/train_batch_32_size_4000.pkl',
                      help='Path to batched train data file. Default value of ./data/batched_data/train_batch_32_size_4000.pkl')
  parser.add_argument('--val_caption_path', type=str,
                      default='../ImageCaptioner/data/annotations/captions_val2014.json',
                      help='Path for val annotation file. Default value of ../ImageCaptioner/data/annotations/captions_val2014.json')
  parser.add_argument('--val_image_dir', type=str,
                      default='../ImageCaptioner/data/val2014',
                      help='Path to val image directory. Default value of ../ImageCaptioner/data/val2014')
  parser.add_argument('--batched_val_path', type=str,
                      default='./data/batched_data/val_batch_32_size_4000.pkl',
                      help='Path to batched val data file. Default value of ./data/batched_data/val_batch_32_size_4000.pkl')
  parser.add_argument('--vocab_path', type=str,
                      required=True,
                      help='Path to vocab. Required')
  parser.add_argument('--output_train_name', type=str,
                      required=True,
                      help='Output train file name. Required.')
  parser.add_argument('--output_val_name', type=str,
                      required=True,
                      help='Output val file name. Required.')
  parser.add_argument('--num_epochs', type=int,
                      default=10,
                      help='Number of epochs to train for. Default value of 10.')
  parser.add_argument('--embedding_dim', type=int,
                      default=512,
                      help='Size of the embedding layer. Default value of 512')
  parser.add_argument('--hidden_dim', type=int,
                      default=512,
                      help='Size of the hidden layer. Default value of 512')
  parser.add_argument('--num_layers', type=int,
                      default=1,
                      help='Number of layers in the lstm. Default value of 1')
  parser.add_argument('--lr', type=float,
                      default=0.001,
                      help='Learning rate. Default value of 0.001')
  parser.add_argument('--grad_clip', type=float,
                      default=5,
                      help='Maximum gradient. Default value of 5')
  parser.add_argument('--dropout', type=float,
                      default=0.0,
                      help='Dropout value for the decoder. Default value of 0.0')
  parser.add_argument('-disable_cuda', action='store_true',
                      default=False,
                      help='Set if cuda should not be used')
  parser.add_argument('--load_checkpoint', type=str,
                      default=None,
                      help='Saved checkpoint file name. Default value of none.')

  args = parser.parse_args()
  main(args)

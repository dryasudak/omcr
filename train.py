import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import NNConv
from torch_geometric.data import (InMemoryDataset, download_url, Data)


class RxnDataset(InMemoryDataset):
  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, filename=None):
    self.raw_file = filename + '.txt'
    self.processed_file = filename + '.pt'
    self.raw_url = './data/rxn/raw/' + self.raw_file

    super(RxnDataset, self).__init__(root, transform, pre_transform, pre_filter)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self):
    return [self.raw_file]

  @property
  def processed_file_names(self):
    return [self.processed_file]

  def download(self):
    file_path = download_url(self.raw_url, self.raw_dir)


class NNModel(nn.Module):
   def __init__(self, in_channels, edge_channels, hidden_channels=16, out_channels=None):
      super().__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.edge_channels = edge_channels
      self.hidden_channels = hidden_channels

      def gen_nn():
         return nn.Sequential(
                nn.Embedding(edge_channels, 8), nn.ReLU(),
                nn.Linear(8, hidden_channels*hidden_channels))

      self.embedding = nn.Embedding(
         in_channels, hidden_channels)
      self.conv1 = NNConv(
         in_channels=hidden_channels,
         out_channels=hidden_channels,
         nn=gen_nn(),
         aggr="mean")
      self.conv2 = NNConv(
         in_channels=hidden_channels,
         out_channels=hidden_channels,
         nn=gen_nn(),
         aggr="mean")
      self.conv3 = NNConv(
         in_channels=hidden_channels,
         out_channels=hidden_channels,
         nn=gen_nn(),
         aggr="mean")
      self.conv4 = NNConv(
         in_channels=hidden_channels,
         out_channels=hidden_channels,
         nn=gen_nn(),
         aggr="mean")
         
      self.f = nn.Linear(hidden_channels, 1)


   def forward(self, data):
      loop_edge_index = data.loop_edge
      loop_pair_index = data.loop_pair
      metal_idx = data.metal_idx

      x = self.embedding(data.x)
      x = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
      x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
      x = F.relu(self.conv3(x, data.edge_index, data.edge_attr))
      x = self.conv4(x, data.edge_index, data.edge_attr)
      fhM = self.f(x[metal_idx, :])

      def calc_x_1(source, target, add: bool):
         left  = torch.cat((x1[metal_idx1], x1[metal_idx1], x1[source]), dim=0)
         right = torch.cat((x1[source],     x1[target],    -x1[target]), dim=0)
         product  = torch.sum(left * right).unsqueeze(0)
         if add:
            ret = torch.add(product, -fhM1)
         else:
            ret = -product
         return ret

      res_vector = []
      for ibat in range(len(loop_edge_index)):
        fhM1 = fhM[ibat]
        loop_edge1 = loop_edge_index[ibat]
        loop_pair1 = loop_pair_index[ibat]
        metal_idx1 = metal_idx[ibat]
        res1 = []
        x1 = x[data.batch==ibat].view(-1, self.hidden_channels)
        res1.extend([calc_x_1(i,j, True ) for i,j in loop_edge1])
        res1.extend([calc_x_1(i,j, False) for i,j in loop_pair1])
        res_vector.append(torch.cat(res1, dim=0))

      return res_vector


current_dir = os.getcwd() + "/"
basedir = current_dir + 'data/rxn/'


def train_test_split(args, r_val=0.1, r_test=0.2):
  random.seed(0)
  filename = args.filename
  rxns = RxnDataset(root=basedir, filename=filename)
#  for i,rxn in enumerate(rxns[:]):
#    if i%100==0: print(i+1,rxn.x)
#  raise RuntimeError

  rand_perm = [i for i in range(len(rxns))]

  n_test = args.ntest
  n_rxn = len(rxns)
  if n_test==None:
    n_test = int(n_rxn * r_test)
  n_val = int(n_rxn * r_val)
  n_train = n_rxn - n_val - n_test

  train_data = []
  for data1 in rxns[rand_perm[n_test+n_val:n_rxn]]:
    target = data1.action_indices[data1.labind]
    data1.action_indices = torch.tensor([target],dtype=torch.int64)
    train_data.append(data1)
  random.shuffle(train_data)

  val_data = []
  for ind1 in rand_perm[n_test:n_test+n_val]:
    data1 = rxns[ind1]
    data1.ipos = ind1
    labind = data1.labind.item()
    if labind==0:
      val_data.append(data1)

  test_data = []
  for ind1 in rand_perm[0:n_test]:
    data1 = rxns[ind1]
    data1.ipos = ind1
    labind = data1.labind.item()
    if labind==0:
      test_data.append(data1)

  print('%d data split into %d train, %d val, %d test data.'
     % (n_rxn, n_train, len(val_data), len(test_data)))
  return train_data,val_data,test_data


def train():
  model.train()
  loss_all = 0
  acc_all = 0
  for data in train_loader:  #  process 64 graph at once
    data = data.to(device)
    target = data.action_indices
    estimate = model(data)  #  list of 64 torch tensor
    optimizer.zero_grad()
    loss = torch.Tensor([0]).to(device)
    for e1,t1 in zip(estimate,target):  #  loop over each graph
      loss += criterion(e1.unsqueeze(0), t1.unsqueeze(0))
      rxn_prob = nn.functional.softmax(e1, dim=0)
      pred_label = torch.argmax(rxn_prob)
      if pred_label==t1:
        acc_all += 1
    loss.backward()
    loss_all += loss.item()
    optimizer.step()
  ndata = len(train_loader.dataset)
  loss_avg = loss_all / ndata
  acc_avg = acc_all / ndata
  return loss_avg, acc_avg


def test(loader):
  model.eval()
  acc_all = 0
  for data in loader:
    data = data.to(device)
    estimate = model(data)
    for e1,t1 in zip(estimate,data.action_indices):
      rxn_prob = nn.functional.softmax(e1, dim=0)
      pred_label = torch.argmax(rxn_prob).item()
      if pred_label in t1:
        acc_all += 1
  ndata = len(loader.dataset)
  acc_avg = acc_all / ndata
  return acc_avg


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, help='the number of epoch to train', default=50)
  parser.add_argument('--hc', type=int, help='the dimensions of hidden channel', default=16)
  parser.add_argument('--batch', type=int, help='batch_size', default=64)
  parser.add_argument('--ntest', type=int, help='the number of test samples', default=None)
  parser.add_argument('--filename', type=str, help='dataset name', default='patent2500')
  args = parser.parse_args()


  train_data, val_data, test_data = train_test_split(args)
  batch_size = args.batch
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = NNModel(
    in_channels=108,
    edge_channels=5,
    hidden_channels=args.hc,
  ).to(device)

  criterion  = nn.CrossEntropyLoss()
  optimizer = optim.Adadelta(model.parameters(), lr=0.7)

  epoch_loss_log = {'epoch':[], 'loss':[], 'train_acc':[], 'val_acc':[], 'test_acc':[]}
  for epoch in range(args.epoch):
    lr = 0.7
    loss, train_acc = train()
    val_acc = test(val_loader)
    test_acc = test(test_loader)

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Train acc: {:.7f}, Val acc: {:.7f}, '
          'Test acc: {:.7f}'.format(epoch, lr, loss, train_acc, val_acc, test_acc), flush=True)

    epoch_loss_log['epoch'].append(epoch+1)
    epoch_loss_log['loss'].append(loss)
    epoch_loss_log['train_acc'].append(train_acc)
    epoch_loss_log['val_acc'].append(val_acc)
    epoch_loss_log['test_acc'].append(test_acc)



import logging
from tqdm import tqdm
import bcolz
import pickle
from torch import nn
import numpy as np
from torch.autograd import Variable
import  torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import argparse

class FFnet(nn.Module):
    def __init__(self,nb=34, xyz=0):
        super(FFnet, self).__init__()
        self.number_label = nb
        self.fc = nn.Linear(100 , self.number_label)

    def forward(self, x):
        x = self.fc(x)
        return x

class LoadTrigger(data.Dataset):
    subtype_events = []

    def __init__(self, type , glove, se):
        super(LoadTrigger, self).__init__()
        self.subtype_events = se
        TRAIN_PATH =  "/home/binhnguyen/PycharmProjects/StatisticWork/data/train.txt"
        TEST_PATH = "/home/binhnguyen/PycharmProjects/StatisticWork/data/test.txt"
        dictionary = dict()
        for word in glove:
            dictionary[word] = 1
        self.type = type
        if self.type == 0:
            self.train_data = []
            self.train_labels = []

            content = open(TRAIN_PATH).readlines()
            for sample in content:
                tokens = sample.strip().split("\t")
                event = tokens[1]
                trigger = tokens[3]
                if (trigger not in dictionary):
                    self.train_data.append(glove['<unk>'])
                    self.train_labels.append(self.subtype_events.index('Other'))
                else:
                    self.train_data.append(glove[trigger])
                    self.train_labels.append(self.subtype_events.index(event))
        else:
            if self.type == 1:
                self.test_data = []
                self.test_labels = []

                content = open(TEST_PATH).readlines()
                for sample in content:
                    tokens = sample.strip().split("\t")
                    event = tokens[1]
                    trigger = tokens[3]
                    if (trigger not in dictionary):
                        self.test_data.append(glove['<unk>'])
                        self.test_labels.append(self.subtype_events.index('Other'))
                    else:
                        self.test_data.append(glove[trigger])
                        self.test_labels.append(self.subtype_events.index(event))

    def __getitem__(self, index):
        if self.type == 0:
            return self.train_data[index] , self.train_labels[index]
        else:
            if self.type == 1:
                return self.test_data[index] , self.test_labels[index]

    def __len__(self):
        if self.type == 0:
            return len(self.train_data)
        else:
            if (self.type == 1):
                return len(self.test_data)


def load_glove():
    glove_path = "/home/binhnguyen/Downloads/glove.6B/"
    vectors = bcolz.open(glove_path + '6B.100.dat')[:]
    words = pickle.load(open(glove_path + '6B.100_words.pkl', 'rb'))
    word2idx = pickle.load(open(glove_path + '6B.100_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    glove['<unk>'] = np.random.normal(scale=0.6, size=(100,))
    print("Load data done!")
    return glove

def load_subtype_event():
    subtype_events = []
    fname = "/home/binhnguyen/PycharmProjects/StatisticWork/data/eventMap.txt"
    with open(fname) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    for sentence in content:
        if len(sentence) != 0:
            list_word = sentence.split(' ')
            subtype_events.append(list_word[0].replace(list_word[1] + ":", ""))
    subtype_events.append('Other')
    # print("Length subtype = " , len(subtype_events))
    return subtype_events


def main(args):
    level = logging.INFO
    format = '  %(message)s'
    handlers = [logging.FileHandler('classify_trigger_log.txt'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=format, handlers=handlers, filename="classify_trigger_log.txt")

    se = load_subtype_event()
    glove = load_glove()
    train = LoadTrigger(0 ,glove,se)
    test  = LoadTrigger(1 ,glove,se)

    train_data = DataLoader(train , batch_size=args.batch, shuffle=True)
    test_data = DataLoader(test , batch_size=args.batch, shuffle=True)

    model = FFnet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    log = open("log.txt", "a")

    for epoch in range(args.epoch):
        train_loss = 0.0

        for feature,label in tqdm(train_data,desc='Training'):
            feature,label = Variable(feature.float()) ,  Variable(label.squeeze())
            y_hat = model(feature)
            loss = criterion(y_hat , label)
            train_loss += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss = 0.0
        y = []
        output = []
        for feature, label in tqdm(test_data, desc='Testing'):
            y += label.numpy().tolist()
            feature, label = Variable(feature.float()), Variable(label.squeeze())
            y_hat = model(feature)
            loss = criterion(y_hat, label)
            test_loss += loss.data.item()
            y_hat = np.argmax(y_hat.data, axis=1)
            output += y_hat.tolist()
        test_acc = accuracy_score(y, output)

        logging.info("Epoch %2d . Test acc = %10.2f ." % (epoch , test_acc) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()
main(args)
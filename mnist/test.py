from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from weighted_cross_entropy import class_select, WeightedCrossEntropyLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class ClassifyNet(nn.Module):
    def __init__(self):
        super(ClassifyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    # The input x now contains two parts
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        self.features = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def get_features(self):
        return self.features

class TrustNet(nn.Module):
    def __init__(self, classifier):
        super(TrustNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*64+320, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self.classifier = classifier

    def forward(self, x):
        y = F.relu(F.max_pool2d(self.conv1(x), 2))
        y = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(y)), 2))
        y = y.view(-1, 4*4*64)

        self.classifier(x)
        y2 = self.classifier.get_features()

        x = F.relu(self.fc1(torch.cat((y, y2), 1)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.softmax(x)

    def set_classifier(self, classifier):
        self.classifier = classifier

classify_model = ClassifyNet()
confidence_model = TrustNet(classify_model)
if args.cuda:
    classify_model.cuda()
    trust_model.cuda()

classify_optimizer = optim.SGD(classify_model.parameters(), lr=args.lr, momentum=args.momentum)
confidence_optimizer = optim.SGD(confidence_model.parameters(), lr=args.lr, momentum=args.momentum)

def train_classification(epoch):
    classify_model.train()
    confidence_model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        classify_optimizer.zero_grad()
        output = classify_model(data)
        

        # loss = F.nll_loss(output, target)

        # Instance weighted loss
        confidence = confidence_model(data)
        confidence = confidence[:,1]
        # print(weights)
        classification_loss = WeightedCrossEntropyLoss()(output, target, confidence)
        # print(loss)
        classification_loss.backward()
        classify_optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Classificaion Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), classification_loss.data[0]))


def train_confidence(epoch):
    classify_model.eval()
    confidence_model.train()

    confidence_model.set_classifier(classify_model)
    for param in confidence_model.classifier.parameters():
        param.requires_grad = False


    negative_cases = []
    negative_targets = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = classify_model(data)

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        corrected = pred.eq(target.data.view_as(pred))

        for idx, row in enumerate(corrected.split(1)):
            # print(confidence_target[idx], row, confidence[idx])
            cor = row.numpy()
            # print(cor[0][0])
            if (cor[0][0]==0):
                negative_cases.append(np.expand_dims(data[idx,0,:].data.numpy(), axis=0))
                negative_targets.append(target[idx].data.numpy()[0])

    negative_targets = torch.from_numpy(np.array(negative_targets))
    negative_data = torch.from_numpy(np.stack(negative_cases, axis=0))

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # neg_target = torch.zeros(data.shape[0])
        selected = np.random.choice(negative_data.shape[0], data.shape[0])
        data = torch.cat((data, negative_data[selected, : ,:, :]), 0)
        # target = torch.cat((target, neg_target.type('torch.LongTensor')), 0)
        target = torch.cat((target, negative_targets[selected.tolist()]), 0)

        data, target = Variable(data), Variable(target)
        output = classify_model(data)

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        corrected = pred.eq(target.data.view_as(pred))


        confidence_optimizer.zero_grad()
        confidence = confidence_model(data)
        confidence = confidence[:,1]
        # print(trust_model(data))

        # confidence_target = Variable((1-torch.exp(class_select(output, target))).data)
        confidence_target = Variable((torch.exp(class_select(output, target))).data, requires_grad=False)

        # only optimize the error cases
        for idx, row in enumerate(corrected.split(1)):
            # print(confidence_target[idx], row, confidence[idx])
            cor = row.numpy()
            # print(cor[0][0])
            if (cor[0][0]==1):
                # confidence_target[idx].data -= confidence_target[idx].data-confidence[idx].data
                if (confidence[idx].data[0]>0.9):
                    confidence_target[idx].data -= confidence_target[idx].data-confidence[idx].data
                # print(confidence_target[idx])

        # for idx, row in enumerate(corrected.split(1)):
        #     print(confidence_target[idx], row, confidence[idx])

        # print(1-torch.exp(class_select(output, target)))
        confidence_loss = nn.MSELoss()(confidence, confidence_target)
        # print("before")
        # print(weight_loss)
        
        confidence_loss.backward()
        confidence_optimizer.step()

        # weights = trust_model(data)
        # weights = weights[:,1]
        # weight_loss = nn.MSELoss()(weights, weights_target)
        # print("after")
        # print(weight_loss)

        if batch_idx % args.log_interval == 0:
            print('Confidence Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), confidence_loss.data[0]))

def test():
    classify_model.eval()
    confidence_model.eval()
    classification_loss = 0
    confidence_loss = 0
    correct = 0
    # for data, target in test_loader:
    for data, target in train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = classify_model(data)
        confidence = confidence_model(data)
        confidence = confidence[:,1]

        # print(confidence)
        # confidence_target = Variable((1-torch.exp(class_select(output, target))).data)
        confidence_target = Variable((torch.exp(class_select(output, target))).data)
        # print(weights_target)

        # for row in confidence.split(1):
        #     if row.data[0]<0.7:
        #         print(row)

        classification_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        confidence_loss += F.mse_loss(confidence, confidence_target, size_average=False).data[0]
    
    print(confidence_loss)
    classification_loss /= len(train_loader.dataset)
    # confidence_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        classification_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    print('\nTest set: Average weight loss: {:.4f}\n'.format(
        confidence_loss))

def evaluate():
    classify_model.eval()
    confidence_model.eval()

    total_num = 0
    correct_num = 0
    accepted_total_num = 0
    accepted_correct_num = 0
    # for data, target in test_loader:
    for data, target in train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = classify_model(data)
        confidence = confidence_model(data)
        confidence = confidence[:,1]

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        corrected = pred.eq(target.data.view_as(pred))

        total_num += corrected.shape[0]
        correct_num += corrected.cpu().sum()

        for idx, row in enumerate(confidence.split(1)):
            # if (corrected[idx][0]==0):
                # print(row, corrected[idx])
            cor = row.data.numpy()
            if (cor[0]>0.3):
                accepted_total_num += 1

                if (corrected[idx][0]>0):
                    accepted_correct_num += 1
                    # print(corrected[idx])

    print("accuracy of total cases")
    print(total_num)
    print(correct_num)

    print("accuracy of accepted cases")
    print(accepted_total_num)
    print(accepted_correct_num)

for epoch in range(1, args.epochs + 1):
    train_classification(epoch)
    evaluate()

for epoch in range(1, 200):
    train_confidence(epoch)
    # test()
    evaluate()

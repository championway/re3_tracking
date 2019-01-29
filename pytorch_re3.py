
# coding: utf-8

# In[1]:


import os, sys
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.modules.normalization as norm
from torch.autograd import Variable
from process_data import ALOVDataset
import torch.optim as optim
import numpy as np


# In[2]:


LSTM_SIZE = 512


# In[3]:


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


# In[4]:


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# In[5]:


#alexnet = models.alexnet(pretrained=True)
class alexnet_conv_layers(nn.Module):
    def __init__(self):
        super(alexnet_conv_layers, self).__init__()
        input_channels = 3
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            norm.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75, k=1.0)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(96, out_channels=16, kernel_size=1, stride=1),
            nn.PReLU(),
            Flatten()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, groups=2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            norm.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75, k=1.0)
        )

        self.skip2 = nn.Sequential(
            nn.Conv2d(256, out_channels=32, kernel_size=1, stride=1),
            nn.PReLU(),
            Flatten()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=2),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, groups=2),
            nn.ReLU()
        )

        self.pool5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv5_flat = nn.Sequential(
            Flatten()
        )

        self.skip5 = nn.Sequential(
            nn.Conv2d(256, out_channels=64, kernel_size=1, stride=1),
            nn.PReLU(),
            Flatten()
        )

        self.conv6 = nn.Sequential(
            nn.Linear(37104 * 2, 2048),
            nn.ReLU()
        )

    def forward(self, x):
        x_out1 = self.conv1(x)
        x_out_skip1 = self.skip1(x_out1)

        x_out2 = self.conv2(x_out1)
        x_out_skip2 = self.skip2(x_out2)

        x_out3 = self.conv3(x_out2)
        x_out4 = self.conv4(x_out3)
        x_out5 = self.conv5(x_out4)

        x_out_skip5 = self.skip5(x_out5)

        x_out_pool =self.pool5(x_out5)
        x_out_pool = self.conv5_flat( x_out_pool)
        x_out = torch.cat((x_out_skip1, x_out_skip2, x_out_skip5, x_out_pool), dim=1)

        y_out1 = self.conv1(x)
        y_out_skip1 = self.skip1(y_out1)

        y_out2 = self.conv2(y_out1)
        y_out_skip2 = self.skip2(y_out2)

        y_out3 = self.conv3(y_out2)
        y_out4 = self.conv4(y_out3)
        y_out5 = self.conv5(y_out4)

        y_out_skip5 = self.skip5(y_out5)

        y_out_pool =self.pool5(y_out5)
        y_out_pool = self.conv5_flat(y_out_pool)
        y_out = torch.cat((y_out_skip1, y_out_skip2, y_out_skip5, y_out_pool), dim=1)

        final_out = torch.cat((x_out, y_out), dim=1)
        conv_out = self.conv6(final_out)
        return conv_out


# In[6]:


class Re3Net(nn.Module):
    def __init__(self):
        super(Re3Net,self).__init__()
        self.conv_layers = alexnet_conv_layers()
        
        #2048 from conv_layers? maybe 1024?
        self.lstm1 =nn.LSTMCell(2048, LSTM_SIZE)
        self.lstm2 = nn.LSTMCell(2048 + LSTM_SIZE, LSTM_SIZE)

        self.fc_final = nn.Linear(LSTM_SIZE,4)

        #self.h0=Variable(torch.rand(1,LSTM_SIZE)).cuda()
        #self.c0=Variable(torch.rand(1,LSTM_SIZE)).cuda()

    '''def init_hidden(self):
        self.h0 = Variable(torch.rand(1, LSTM_SIZE))
        self.c0 = Variable(torch.rand(1, LSTM_SIZE))'''

    def forward(self, x, prev_LSTM_state=False):
        out = self.conv_layers(x)
        
        h0 = 0
        c0 = 0
        
        h0 = Variable(torch.rand(x.shape[0],LSTM_SIZE)).cuda()
        c0 = Variable(torch.rand(x.shape[0],LSTM_SIZE)).cuda()
        
        lstm_out, h0 = self.lstm1(out, (h0, c0))

        lstm2_in = torch.cat((out, lstm_out), dim=1)

        lstm2_out, h1 = self.lstm2(lstm2_in, (h0, c0))

        out = self.fc_final(lstm2_out)
        return out


# In[7]:


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


# In[8]:


def evaluate(model, dataloader, criterion, epoch):

    model.eval()
    dataset = dataloader.dataset
    total_loss = 0

    for i in range(64):
        sample = dataset[i]
        sample['currimg'] = sample['currimg'][None,:,:,:]
        sample['previmg'] = sample['previmg'][None,:,:,:]
        x1, x2 = sample['previmg'], sample['currimg']
        y = sample['currbb']

        if use_gpu:
            x1 = Variable(x1.cuda())
            x2 = Variable(x2.cuda())
            y = Variable(y.cuda(), requires_grad=False)
        else:
            x1 = Variable(x1)
            x2 = Variable(x2)
            y = Variable(y, requires_grad=False)

        output = model(x1, x2)
        #print(output.size()) # [1,4]
        #print(y.size()) # [4]
        output = output.view(4)
        loss = criterion(output, y)
        total_loss += loss.data[0]
        if i % 10 == 0:
            print('[validation] epoch = %d, i = %d, loss = %f' % (epoch, i, loss.data[0]))

    seq_loss = total_loss/64
    return seq_loss


# In[9]:


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = RunningAverage()
    counter=0
    for i,data in enumerate(dataloader):
        optimizer.zero_grad()
        x1, x2, y = data['previmg'], data['currimg'], data['currbb']
        output = model(x1, x2)
        loss = loss_fn(output, y)
        loss.backward(retain_graph=True)
        # performs updates using calculated gradients
        optimizer.step()
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output = output.data.cpu().numpy()
            # compute all metrics on this batch
            summary_batch = {}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)
            logging.info('- Average Loss for iteration {} is {}'.format(i,loss.data[0]/params.batch_size))

        # update the average loss
        loss_avg.update(loss.data[0])
        counter+=1

    print(counter)
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


# In[10]:


def train_model(net, dataloader, optim, loss_function, num_epochs):

    dataset_size = dataloader.dataset.len
    for epoch in range(num_epochs):
        net.train()
        curr_loss = 0.0

        # currently training on just ALOV dataset
        i = 0
        for data in dataloader:

            x1, x2, y = data['previmg'], data['currimg'], data['currbb']
            if use_gpu:
                x1, x2, y = Variable(x1.cuda()), Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)
            else:
                x1, x2, y = Variable(x1), Variable(x2), Variable(y, requires_grad=False)

            optim.zero_grad()

            output = net(x1,x2)
            #print(output.size()) # [1,4]
            #print(y.size()) # [4]
            loss = loss_function(output, y)

            loss.backward(retain_graph=True)
            optim.step()
            if i%20 == 0:
                print('[training] epoch = %d, i = %d/%d, loss = %f' % (epoch, i, dataset_size			,loss.data[0]) )
                sys.stdout.flush()
            i = i + 1
            curr_loss += loss.data[0]
        epoch_loss = curr_loss / dataset_size
        print('Loss: {:.4f}'.format(epoch_loss))
        
        path = save_directory + '_batch_' + str(epoch) + '_loss_' + str(round(epoch_loss, 3)) + '.pth'
        torch.save(net.state_dict(), path)

        val_loss = evaluate(net, dataloader, loss_function, epoch)
        print('Validation Loss: {:.4f}'.format(val_loss))
    return net


# In[11]:


# Convert numpy arrays to torch tensors
class ToTensor(object):
    def __call__(self, sample):
        prev_img, curr_img = sample['previmg'], sample['currimg']
        # swap color axis because numpy image: H x W x C ; torch image: C X H X W
        prev_img = prev_img.transpose((2, 0, 1))
        curr_img = curr_img.transpose((2, 0, 1))
        if 'currbb' in sample:
            currbb = sample['currbb']
            return {'previmg': torch.from_numpy(prev_img).float(),
                    'currimg': torch.from_numpy(curr_img).float(),
                    'currbb': torch.from_numpy(currbb).float()
                    }
        else:
            return {'previmg': torch.from_numpy(prev_img).float(),
                    'currimg': torch.from_numpy(curr_img).float()
                    }


# To normalize the data points
class Normalize(object):
    def __call__(self, sample):

        prev_img, curr_img = sample['previmg'], sample['currimg']
        self.mean = [104, 117, 123]
        prev_img = prev_img.astype(float)
        curr_img = curr_img.astype(float)
        prev_img -= np.array(self.mean).astype(float)
        curr_img -= np.array(self.mean).astype(float)

        if 'currbb' in sample:
            currbb = sample['currbb']
            currbb = currbb*(10./227);
            return {'previmg': prev_img,
                    'currimg': curr_img,
                    'currbb': currbb}
        else:
            return {'previmg': prev_img,
                    'currimg': curr_img
}
transform = transforms.Compose([Normalize(), ToTensor()])


# In[12]:


save_directory = 'saved_models/'
save_model_step = 5
learning_rate = 0.00001
use_gpu = True
num_epochs = 100


# In[13]:


alov = ALOVDataset('../data/alov/imagedata++/', '../data/alov/alov300++_rectangleAnnotation_full/', transform)
dataloader = DataLoader(alov, batch_size = 1)


# In[14]:


net = Re3Net().cuda()
loss_function = torch.nn.L1Loss(size_average=False).cuda()
optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.0005)


# In[15]:


if os.path.exists(save_directory):
    print('Directory %s already exists', save_directory)
else:
    os.makedirs(save_directory)
    print('Create directory: %s', save_directory)


# In[16]:


net = train_model(net, dataloader, optimizer, loss_function, num_epochs)


# In[31]:


torch.cuda.empty_cache()


# In[82]:


a = torch.empty(1, 4, dtype=torch.float)

b = torch.empty(4, dtype=torch.float)

print(a.size(), b.size())

c = a.view(4)
print(c.size())


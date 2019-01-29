import cv2
import glob
import os
import time
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
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
# Network Constants
from re3_utils.util import bb_util
from re3_utils.util import im_util
from constants import CROP_SIZE
from constants import CROP_PAD
from constants import LSTM_SIZE
from constants import LOG_DIR
from constants import GPU_ID
from constants import MAX_TRACK_LENGTH

SPEED_OUTPUT = True

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

    def forward(self, x, y):
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

        y_out1 = self.conv1(y)
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
        torch.cuda.empty_cache()
        self.conv_layers = alexnet_conv_layers()
        
        #2048 from conv_layers? maybe 1024?
        self.lstm1 =nn.LSTMCell(2048, LSTM_SIZE)
        self.lstm2 = nn.LSTMCell(2048 + LSTM_SIZE, LSTM_SIZE)

        self.fc_final = nn.Linear(LSTM_SIZE,4)
        
        self.h1 = Variable(torch.rand(1, LSTM_SIZE)).cuda()
        self.c1 = Variable(torch.rand(1, LSTM_SIZE)).cuda()
        
        self.h2 = Variable(torch.rand(1, LSTM_SIZE)).cuda()
        self.c2 = Variable(torch.rand(1, LSTM_SIZE)).cuda()

    def forward(self, x, y, lstm_state=None):
        out = self.conv_layers(x, y)
        
        if lstm_state is None:
            self.h1 = Variable(torch.rand(x.shape[0], LSTM_SIZE)).cuda()
            self.c1 = Variable(torch.rand(x.shape[0], LSTM_SIZE)).cuda()
            self.h2 = Variable(torch.rand(x.shape[0], LSTM_SIZE)).cuda()
            self.c2 = Variable(torch.rand(x.shape[0], LSTM_SIZE)).cuda()
        else:
            self.h1 = lstm_state[0]
            self.c1 = lstm_state[1]
            self.h2 = lstm_state[2]
            self.c2 = lstm_state[3]
        #print(self.h1.cpu().numpy()[0][0])
        #print(self.c1.cpu().numpy()[0][0])
        #print(self.h2.cpu().numpy()[0][0])
        #print(self.c2.cpu().numpy()[0][0])
        h1_, c1_ = self.lstm1(out, (self.h1, self.c1))

        lstm2_in = torch.cat((out, h1_), dim=1)

        h2_, c2_ = self.lstm2(lstm2_in, (self.h2, self.c2))

        out = self.fc_final(h2_)
        h1_ = h1_.detach()
        c1_ = c1_.detach()
        h2_ = h2_.detach()
        c2_ = c2_.detach()
        #print(h1_.cpu().numpy()[0][0])
        #print(c1_.cpu().numpy()[0][0])
        #print(h2_.cpu().numpy()[0][0])
        #print(c2_.cpu().numpy()[0][0])
        #print('======')
        return out, [h1_, c1_, h2_, c2_]


class Re3Tracker(object):
    def __init__(self):
        basedir = os.path.dirname(__file__)
        model_path = '../saved_models_lstmmm/_batch_90_loss_0.241.pth'
        #self.imagePlaceholder = tf.placeholder(tf.uint8, shape=(None, CROP_SIZE, CROP_SIZE, 3))
        #self.prevLstmState = tuple([tf.placeholder(tf.float32, shape=(None, LSTM_SIZE)) for _ in range(4)])
        #self.batch_size = tf.placeholder(tf.int32, shape=())
        '''self.outputs, self.state1, self.state2 = network.inference(
                self.imagePlaceholder, num_unrolls=1, batch_size=self.batch_size, train=False,
                prevLstmState=self.prevLstmState)'''

        self.net = Re3Net().cuda()
        self.net.load_state_dict(torch.load(model_path))
        print(self.net)
        self.outputs, self.state1, self.state2 = None, None, None
        self.tracked_data = {}
        self.lstmState = None
        self.time = 0
        self.total_forward_count = -1


    # unique_id{str}: A unique id for the object being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_box{None or 4x1 numpy array or list}: 4x1 bounding box in X1, Y1, X2, Y2 format.
    def track(self, unique_id, image, starting_box=None):
        start_time = time.time()

        #========== Get Image ==========
        if type(image) == str:
            image = cv2.imread(image)[:,:,::-1]
        else:
            image = image.copy()

        image_read_time = time.time() - start_time

        #========== If this is first time that we gave it a bbox ==========
        if starting_box is not None:
            lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)] # [X, X, X, X], initial lstm state with 4 value
            pastBBox = np.array(starting_box) # turns list into numpy array if not and copies for safety.
            prevImage = image
            originalFeatures = None
            forwardCount = 0

        #========== If same ID, EX:'webcam' ==========
        elif unique_id in self.tracked_data:
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
        else:
            raise Exception('Unique_id %s with no initial bounding box' % unique_id)

        croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
        croppedInput1, _ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
        
        croppedInput0 = torch.tensor(croppedInput0)
        croppedInput0 = croppedInput0.transpose(0,2)
        croppedInput0.unsqueeze_(0)
        croppedInput0 = croppedInput0.type('torch.FloatTensor')

        croppedInput1 = torch.tensor(croppedInput1)
        croppedInput1 = croppedInput1.transpose(0,2)
        croppedInput1 = croppedInput1.type('torch.FloatTensor')
        croppedInput1.unsqueeze_(0)

        '''feed_dict = {
                self.imagePlaceholder : [croppedInput0, croppedInput1],
                self.prevLstmState : lstmState,
                self.batch_size : 1,
                }'''
        #rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        rawOutput, self.lstmState = self.net(croppedInput0.cuda(), croppedInput1.cuda(), self.lstmState)
        rawOutput = rawOutput.cpu().detach().numpy()
        #s1 = s1.cpu().detach().numpy()
        #s2 = s2.cpu().detach().numpy()
        #s1 = np.squeeze(s1)
        #s2 = np.squeeze(s2)
        #lstmState = [s1[0], s1[1], s2[0], s2[1]]
        #print(s1[0])
        if forwardCount == 0:
            originalFeatures = self.lstmState

        prevImage = image

        # Shift output box to full image coordinate system.
        outputBox = bb_util.from_crop_coordinate_system(rawOutput.squeeze() / 10.0, pastBBoxPadded, 1, 1)
        #print(outputBox, rawOutput)

        # Reset LSTM state every 32 iterations to avoid drift
        if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
            croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
            # croppedInput[np.newaxis,...]   (227,227,3) --> (1,227,227,3)
            # np.tile(croppedInput[np.newaxis,...], (2,1,1,1))   (1,227,227,3) --> (2,227,227,3)
            input = np.tile(croppedInput[np.newaxis,...], (2,1,1,1))
            '''feed_dict = {
                    self.imagePlaceholder : input,
                    self.prevLstmState : originalFeatures,
                    self.batch_size : 1,
                    }'''
            #rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
            rawOutput, self.lstmState = self.net(croppedInput0.cuda(), croppedInput1.cuda(), None)
            rawOutput = rawOutput.cpu().detach().numpy()
            #s1 = s1.cpu().detach().numpy()
            #s2 = s2.cpu().detach().numpy()
            #s1 = np.squeeze(s1)
            #s2 = np.squeeze(s2)
            #lstmState = [s1[0], s1[1], s2[0], s2[1]]

        forwardCount += 1
        self.total_forward_count += 1

        if starting_box is not None:
            # Use label if it's given
            outputBox = np.array(starting_box)

        self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)
        end_time = time.time()
        if self.total_forward_count > 0:
            self.time += (end_time - start_time - image_read_time)
        if SPEED_OUTPUT and self.total_forward_count % 10 == 0:
            print('Current tracking speed:   %.3f FPS' % (1 / (end_time - start_time - image_read_time)))
            print('Current image read speed: %.3f FPS' % (1 / (image_read_time)))
            print('Mean tracking speed:      %.3f FPS\n' % (self.total_forward_count / max(.00001, self.time)))
        #time.sleep(0.1)
        return outputBox
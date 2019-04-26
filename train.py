import argparse

import torch.utils.data

from DataStorage import *
from fcn import *
from FCN_PP_VGG16 import *
from metrics.metrics import *
from trainer import *
from unet import UNet
import unet_Atrous
from PyrmaidUNet.model import PyrmaidUNet

parser = argparse.ArgumentParser(description='Training set')
parser.add_argument('--batch', default=2, help='set the batch size in training process', type=int)
parser.add_argument('--lr', default=1e-4, help='set the learning rate in training process', type=float)
parser.add_argument('--lr_step', default=4, help='the decay frequency of learning rate', type=int)
parser.add_argument('--gamma', default=0.1, help='the decay rate of learning rate', type=float)
parser.add_argument('--epoches', default=50, help='total training epoches', type=int)
parser.add_argument('--valfreq', default=50, help='validation frequency(iter)', type=int)
parser.add_argument('--comment', default='', help='comment', type=str)
parser.add_argument('--model', default='FCN-PP', help='choose a model', type=str, choices=('unet', 'fcn32s', 'fcn16s', 'fcn8s', 'fcnpp', 'unet-atrous', 'punet'))
parser.add_argument('--wdecay', default=0.0005, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--ckpt', default=400, type=int)
parser.add_argument('--N', default=1, type=int, help='how much silces will be put into the network')
parser.add_argument('--WC', default=0, type=int)
parser.add_argument('--WW', default=2048, type=int)
parser.add_argument('--gpu', default=1, help='choose a hardware to train', type=int, choices=(0, 1, 2))
parser.add_argument('--log_dir', default='logs', help='choose a dir to save tensorboardX log', type=str)
args = parser.parse_args()


BATCH_SIZE = args.batch
LEARNING_RATE = args.lr
LR_STEP = args.lr_step
GAMMA = args.gamma
EPOCHS = args.epoches
VAL_FREQ = args.valfreq
MODEL = args.model
WEIGHT_DECAY = args.wdecay
MOMENTUM = args.momentum
CKPT_FREQ = args.ckpt
N = args.N
WC = args.WC
WW = args.WW
LOG_DIR = args.log_dir
COMMENT = '_{}_liver_mixed_zfill_onlineAugmentation_batch_size_{}_lr_{}_lr_step_{}_gamma_{}_N_{}_WC_{}_WW_{}_'.format(str(MODEL), str(BATCH_SIZE), str(LEARNING_RATE), str(LR_STEP), str(GAMMA), str(N), str(WC), str(WW)) + str(args.comment)



torch.backends.cudnn.benchmark=True
if args.gpu == 1:
    device = torch.device('cuda:0')
elif args.gpu == 2:
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if MODEL == 'unet':
    print('Using U-net')
    model = UNet(N, 2)
    model = model.to(device)
elif MODEL == 'fcnpp':
    print('Using FCN-PP')
    model = MUNET(N, 2)
    model.fixPretrianedParams(False)
    model = model.to(device)
elif MODEL == 'fcn8s':
    model = FCN8s(N, 2)
    model = model.to(device)
elif MODEL == 'fcn16s':
    model = FCN16s(N, 2)
    model = model.to(device)
elif MODEL == 'fcn32s':
    model = FCN32s(N, 2)
    model = model.to(device)
elif MODEL == 'unet-atrous':
    model = unet_Atrous.UNet(N, 2)
    model = model.to(device)
elif MODEL == 'punet':
    model = PyrmaidUNet(N, 2)
    model = model.to(device)

# model = nn.DataParallel(model)



################# Dataset: liver_mixed_zfill ##################
raw_dir = './data/liver/liver_mixed_zfill/tr/raw'
target_dir = './data/liver/liver_mixed_zfill/tr/target'

val_raw_dir = './data/liver/liver_mixed_zfill/val/raw'
val_target_dir = './data/liver/liver_mixed_zfill/val/target'

# ################# Dataset: liver_mixed_zfill with superpixels ##################
# raw_dir = './data/liver/liver_mixed_zfill/superpixels/tr/raw'
# target_dir = './data/liver/liver_mixed_zfill/tr/target'
#
# val_raw_dir = './data/liver/liver_mixed_zfill/superpixels/val/raw'
# val_target_dir = './data/liver/liver_mixed_zfill/val/target'


if N == 1:
    data_store_tr = DICOM_DataStorage_training(raw_dir, target_dir, WC, WW)
    data_store_ts = DICOM_DataStorage_validation(val_raw_dir, val_target_dir, WC, WW)
else:
    data_store_tr = Multi_Silces_DataStorage_training(raw_dir, target_dir, WC, WW, N)
    data_store_ts = Multi_Silces_DataStorage_validation(val_raw_dir, val_target_dir, WC, WW, N)



loader_tr = torch.utils.data.DataLoader(dataset=data_store_tr, batch_size=BATCH_SIZE, shuffle=True,)
loader_ts = torch.utils.data.DataLoader(dataset=data_store_ts, batch_size=BATCH_SIZE, shuffle=False,)

dataset = {'train': data_store_tr, 'val': data_store_ts}
dataloader = {'train': loader_tr, 'val': loader_ts}
phase = ['train', 'val']


Loss_Recorder = {'train': torch.zeros(EPOCHS), 'val': torch.zeros(EPOCHS)}

# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1.2]))
criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=GAMMA)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 10], gamma=GAMMA, last_epoch=-1)
# lr_scheduler.step()

if __name__ == '__main__':
    model_trainer = trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                        loss_criterion=criterion, eval_criterion=IoU, device=device,
                        dataloaders=dataloader, max_epochs=EPOCHS, verbose_train = 1, verbose_val=VAL_FREQ,
                        ckpt_frequency=CKPT_FREQ, checkpoint_dir='checkpoints', max_iter=10000000,
                        comments=COMMENT)
    model_trainer.train()
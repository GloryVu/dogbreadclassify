import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from threading import Thread
from dogbreeds.classifier.trainlog import TrainLogger
from dogbreeds.classifier.test import (test_speed, obtain_num_parameters)
import shutil
class TrainThread(Thread):
    def __init__(self, train_root = 'classifier/data/train/',valid_root = 'classifier/data/val/'
    , sr = True, s = 0.0001, batch_size = 64,epochs = 40, start_epoch = 0
    , lr = 0.001,resume='', save = 'classifier/models',arch ='resnet50'
    , default_pretrain = True, pretrained_model_path ='', use_pretrain = True):
        Thread.__init__(self)
        self.setDaemon(True)
        self.train_root = train_root
        self.valid_root = valid_root
        self.sr = sr
        self.s = s
        self.batch_size = batch_size
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.lr = lr
        self.resume = resume
        self.save = save
        self.arch = arch
        self.default_pretrain =default_pretrain
        self.pretrained_model_path = pretrained_model_path
        self.use_pretrain = use_pretrain
        self.trainlogger = TrainLogger(create =(start_epoch==0),csv_path = 'classifier/checkpoints/train_log_1.csv')
    def train_process(self):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        print(device)
        # batch_size = 64
        if not os.path.exists(self.save):
            os.makedirs(self.save)

        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224,scale=(0.6,1.0),ratio=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
            torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224,scale=(1.0,1.0),ratio=(1.0,1.0)),
            # transforms.RandomResizedCrop(224,scale=(0.6,1.0),ratio=(0.8,1.0)),
            # transforms.RandomHorizontalFlip(),
            # torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
            # torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        train_data =  torchvision.datasets.ImageFolder(
                root=self.train_root,
                transform=train_transform
            )

        valid_data = torchvision.datasets.ImageFolder(
                root=self.valid_root,
                transform=test_transform
            )

        train_set = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        test_set = torch.utils.data.DataLoader(
            valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        def updateBN(model, s ,pruning_modules):
            for module in pruning_modules:
                module.weight.grad.data.add_(s * torch.sign(module.weight.data))

        criteration = nn.CrossEntropyLoss()
        def train(model,device,dataset,optimizer,epoch,pruning_modules):
            model.train()
            correct = 0
            for i,(x,y) in tqdm(enumerate(dataset)):
                x , y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                pred = output.max(1,keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
                loss =  criteration(output,y)     
                loss.backward()
                optimizer.step()

                if self.sr:
                    updateBN(model,self.s,pruning_modules)
                
            print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.3f}%)".format(epoch,loss,correct,len(dataset)*self.batch_size,100*correct/(len(dataset)*self.batch_size)))
            return loss.item(), (1.0*correct/(len(dataset)*self.batch_size))

        def valid(model,device,dataset):
            model.eval()
            correct = 0
            with torch.no_grad():
                for i,(x,y) in tqdm(enumerate(dataset)):
                    x,y = x.to(device) ,y.to(device)
                    output = model(x)
                    loss = criteration(output,y)
                    pred = output.max(1,keepdim=True)[1]
                    correct += pred.eq(y.view_as(pred)).sum().item()
            print("Test Loss {:.4f} Accuracy {}/{} ({:.3f}%)".format(loss,correct,len(dataset)*self.batch_size,100*correct/(len(dataset)*self.batch_size)))
            return loss.item(), (1.0*correct/(len(dataset)*self.batch_size))
        def get_pruning_modules(model):
            module_list = []
            for module in model.modules():
                if isinstance(module,torchvision.models.resnet.Bottleneck):
                    module_list.append(module.bn1)
                    module_list.append(module.bn2)
            return module_list
        resnet = getattr(torchvision.models, self.arch)
        model = resnet(pretrained=self.use_pretrain)
        if not self.default_pretrain:
            model.load_state_dict(torch.load(self.pretrained_model_path))
        num_classes = len(os.listdir(self.train_root))
        model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features,512),
                nn.Linear(512,num_classes),
            )
        model.to(device)

        pruning_modules = get_pruning_modules(model)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        def checkpoint(model, filename):
            torch.save({
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                }, filename)
    
        def resume(model, filename):
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.start_epoch > 0:
            resume_epoch = self.start_epoch - 1
            resume(model, f"epoch-{resume_epoch}.pth")        
        model.to(device)
        for epoch in range(self.start_epoch,self.epochs + 1):
            train_loss, train_accuracy = train(model,device,train_set,optimizer,epoch,pruning_modules)
            val_loss,val_accuracy = valid(model,device,test_set)
            checkpoint(model,f"classifier/checkpoints/train-epoch-{epoch}.pth")
            random_input = torch.rand((16, 3, 224, 224)).to(device)
            infer_time = test_speed(random_input, model, repeat=100)
            mode_size = obtain_num_parameters(model)
            self.trainlogger.insert(epoch,train_accuracy,val_accuracy,train_loss,val_loss,mode_size,infer_time)
            if epoch>0:
                os.remove(f"classifier/checkpoints/train-epoch-{epoch-1}.pth")
        torch.save(model.state_dict(), 'classifier/models/trained_model.pth')
    def run(self):
        self.train_process()
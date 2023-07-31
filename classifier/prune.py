import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch_pruning as tp
import shutil
from threading import Thread
from classifier.trainlog import TrainLogger
from classifier.test import (test_speed, obtain_num_parameters)
import shutil
class PruneThread(Thread):
    def __init__(self, train_root = 'classifier/data/train/',valid_root = 'classifier/data/val/'
        , sr = True, s = 0.0001, batch_size = 64,epochs = 40*3, start_epoch = 0
        , lr = 0.001,resume='', save = 'classifier/models',arch ='resnet50'
        , default_pretrain = True, pretrained_model_path ='', use_pretrain = True
        ,percent=0.1):
            Thread.__init__(self)
            self.train_root = train_root
            self.valid_root = valid_root
            self.sr = sr
            self.s = s
            self.setDaemon(True)
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
            self.trainlogger = TrainLogger(create =(start_epoch==0), csv_path = 'classifier/checkpoints/prune_log_1.csv')
            self.percent = percent

    def prune_process(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
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
                transform=train_transform
            )

        train_set = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True
        )

        test_set = torch.utils.data.DataLoader(
            valid_data,
            batch_size=self.batch_size,
            shuffle=False
        )

        def updateBN(model, s ,pruning_modules):
            for module in pruning_modules:
                module.weight.grad.data.add_(s * torch.sign(module.weight.data))

        criteration = nn.CrossEntropyLoss()
        def train(model,device,dataset,optimizer,epoch,pruning_modules):
            
            model.train().to(device)
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
            model.eval().to(device)
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

        def gather_bn_weights(model,pruning_modules):
            size_list = [module.weight.data.shape[0] for module in model.modules() if module in pruning_modules]
            bn_weights = torch.zeros(sum(size_list))
            index = 0
            for module, size in zip(pruning_modules, size_list):
                bn_weights[index:(index + size)] = module.weight.data.abs().clone()
                index += size

            return bn_weights

        def computer_eachlayer_pruned_number(bn_weights,thresh):
            num_list = []
            #print(bn_modules)
            for module in bn_modules:
                num = 0
                #print(module.weight.data.abs(),thresh)
                for data in module.weight.data.abs():
                    if thresh > data.float():
                        num +=1
                num_list.append(num)
            #print(thresh)
            return num_list

        def prune_model(model,num_list):
            model.to(device)
            sample = torch.randn(1, 3, 224, 224)
            sample.to(device)
            DG = tp.DependencyGraph().build_dependency(model, sample)
            def prune_bn(bn, num):
                L1_norm = bn.weight.detach().cpu().numpy()
                prune_index = np.argsort(L1_norm)[:num].tolist() # remove filters with small L1-Norm
                plan = DG.get_pruning_plan(bn, tp.prune_batchnorm, prune_index)
                plan.exec()
            
            blk_id = 0
            for m in model.modules():
                if isinstance( m, torchvision.models.resnet.Bottleneck ):
                    prune_bn( m.bn1, num_list[blk_id] )
                    prune_bn( m.bn2, num_list[blk_id+1] )
                    blk_id+=2
            return model  


        resnet = getattr(torchvision.models, self.arch)
        model = resnet(pretrained=self.use_pretrain)
        num_classes = len(os.listdir(self.train_root))
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features,512),
                                nn.Linear(512,num_classes),)
        if not self.default_pretrain:
            model.load_state_dict(torch.load(self.pretrained_model_path))
        
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
            resume(model, f"classifier/checkpoints/prune-epoch-{resume_epoch}.pth")
        def prune(model):
            bn_modules = get_pruning_modules(model)    

            bn_weights = gather_bn_weights(model,bn_modules)
            sorted_bn = torch.sort(bn_weights)[0]
            sorted_bn, sorted_index = torch.sort(bn_weights)
            thresh_index = int(len(bn_weights) * self.percent)
            thresh = sorted_bn[thresh_index].to(device)

            num_list = computer_eachlayer_pruned_number(bn_weights,thresh)

            return prune_model(model,num_list),bn_modules
            
        bn_modules = get_pruning_modules(model)
        #prec = valid(model,device,test_set)
        for epoch in range(self.epochs):
            if(epoch%8==0):
                model,bn_modules = prune(model) 
                model.to(device)
            train_loss, train_accuracy = train(model,device,train_set,optimizer,epoch,pruning_modules)
            val_loss,val_accuracy = valid(model,device,test_set)
            checkpoint(model,f"classifier/checkpoints/prune-epoch-{epoch}.pth")
            random_input = torch.rand((16, 3, 224, 224)).to(device)
            infer_time = test_speed(random_input, model, repeat=100)
            mode_size = obtain_num_parameters(model)
            self.trainlogger.insert(epoch,train_accuracy,val_accuracy,train_loss,val_loss,mode_size,infer_time)
            if epoch>0:
                os.remove(f"classifier/checkpoints/prune-epoch-{epoch-1}.pth")
            #torch.save(model.state_dict(), 'model_pruned.pth')
        torch.save(model, 'classifier/models/pruned_model.pth' )
    def run(self):
        self.prune_process()
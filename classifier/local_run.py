import time
from dogbreeds.classifier.prune import PruneThread
from dogbreeds.classifier.train import TrainThread
train = TrainThread(arch = 'resnet152',batch_size=32)
print('--------')
train.start()
while(train.is_alive()):
    time.sleep(1)
prune = PruneThread(arch = 'resnet152',batch_size=32)
prune.start()
while(prune.is_alive()):
    time.sleep(1)
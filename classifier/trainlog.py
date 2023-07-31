import pandas as pd
class TrainLogger:
    def __init__(self, create = True, csv_path = 'classifier/checkpoints/log.csv'):
        self.csv_path = csv_path
        if create:
            data = {'epoch':[], 'train_accuracy':[], 'val_accuracy':[],'train_loss':[],'val_loss':[],'mode_size':[],'infer_time':[]}
            df = pd.DataFrame(data)
            df.to_csv(csv_path)
        self.df = pd.read_csv(csv_path)    
    def insert(self,epoch,train_accuracy,val_accuracy,train_loss,val_loss,mode_size,infer_time):
        data = {'epoch':[epoch], 'train_accuracy':[train_accuracy], 'val_accuracy':[val_accuracy],'train_loss':[train_loss],'val_loss':[val_loss],'mode_size':[mode_size],'infer_time':[infer_time]}
        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True,axis = 0)
        self.df.to_csv(self.csv_path,index=False)
            
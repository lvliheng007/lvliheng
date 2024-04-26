from __future__ import print_function, division

import os
import time

from sklearn.model_selection import train_test_split
from data_processing import load_data

import torch.utils.data as tud
from torch.utils.data.dataset import TensorDataset

import torch.nn as nn

import torch

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter

from nilmtk.legacy.disaggregate import Disaggregator


import matplotlib
matplotlib.use('TkAgg')




cuda = True if torch.cuda.is_available() else False
USE_CUDA = torch.cuda.is_available()




cuda = True if torch.cuda.is_available() else False

# the structure of the Generator

class Generator(nn.Module):
    def __init__(self, mains_length, appliance_length, num_meters):
        super(Generator, self).__init__()
        self.seq_length = appliance_length
        self.mains_length = mains_length
        # Create a dynamic list of meters
        self.meters = nn.ModuleList([GeneratorB(self.mains_length, self.seq_length) for _ in range(num_meters)])
        # Recurrent or another form of collective processing unit
        self.rec = Generator_rec(self.mains_length, self.seq_length,num_meters)

    def forward(self, x):
        # Process input through each meter
        meter_outputs = [meter(x) for meter in self.meters]
        # Assuming Generator_rec expects inputs in a specific format, e.g., each as (-1,1,35)
        rec_input = [output.view(-1, 1, 35) for output in meter_outputs]
        rec_output = self.rec(*rec_input)

        return (*meter_outputs, rec_output)

# the structure of the individual Generator
class GeneratorB(nn.Module):

    def __init__(self, mains_length, appliance_length):
        # Refer to "ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring[C].The 32nd AAAI Conference on Artificial Intelligence"
        super(GeneratorB, self).__init__()
        self.seq_length = appliance_length
        self.mains_length = mains_length

        self.conv = nn.Sequential(
            nn.ConstantPad1d((4, 4), 0),
            nn.Conv1d(1, 24, 9, stride=1),

            nn.ReLU(True),
            nn.ConstantPad1d((3, 3), 0),
            nn.Conv1d(24, 48, 7, stride=1),

            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(48,72, 5, stride=1),

            nn.ReLU(True),
            nn.ConstantPad1d((1, 1), 0),
            nn.Conv1d(72, 96, 3, stride=1),

            nn.ReLU(True),

        )

        self.dense = nn.Sequential(
            nn.Linear(96 * self.seq_length, 6 *  self.seq_length),
            nn.ReLU(True),
            nn.Linear(6 *  self.seq_length, 35),

        )

    def forward(self, x):
        x1 = self.conv(x)
        x = self.dense(x1.view(-1, 96 * self.seq_length))

        return x.view(-1, 35)

# the structure of the Generator_rec

class Generator_rec(nn.Module):
    def __init__(self, mains_length, appliance_length,num_meters):
        # Refer to "ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring[C].The 32nd AAAI Conference on Artificial Intelligence"
        super(Generator_rec, self).__init__()
        self.seq_length = appliance_length
        self.mains = mains_length
        self.meter_num =35

        self.conv = nn.Sequential(
            nn.ConstantPad1d((4, 4), 0),
            nn.Conv1d(num_meters, 30, 9, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((3, 3), 0),
            nn.Conv1d(30, 40, 7, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(40, 50,5, stride=1),
            nn.ReLU(True),

        )

        self.dense = nn.Sequential(
            nn.Linear(50 *self.meter_num,     5*self.meter_num),
            nn.ReLU(True),
            nn.Linear( 5* self.meter_num, 1))

    def forward(self, *rec_input ):
        x = torch.cat((rec_input), dim=1)
        x = self.conv(x)
        x = self.dense(x.view(-1,50*   self.meter_num))

        return x.view(-1, 1)



directory = './closed_best' + './'
class closedDisaggregator(Disaggregator):

    def __init__(self,num_meters):
        '''Initialize disaggregator
        '''
        self.MODEL_NAME = "pytorch Gan"
        self.window_size = 129
        self.batchsize =64

        self.gen_model2 = Generator(self.batchsize, self.window_size,num_meters)
        self.writer = SummaryWriter(directory)
        self.n_critic=5
        self.rec_model = Generator_rec(self.batchsize, self.window_size,num_meters)

        self.num_update_iteration=0
    def initialize(self, layer):
        # Xavier_uniform will be applied to W_{ih}, Orthogonal will be applied to W_{hh}, to be consistent with Keras and Tensorflow
        if isinstance(layer, nn.GRU):
            torch.nn.init.xavier_uniform_(layer.weight_ih_l0.data)
            torch.nn.init.orthogonal_(layer.weight_hh_l0.data)
            torch.nn.init.constant_(layer.bias_ih_l0.data, val=0.0)
            torch.nn.init.constant_(layer.bias_hh_l0.data, val=0.0)
        # Xavier_uniform will be applied to conv1d and dense layer, to be consistent with Keras and Tensorflow
        if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight.data)
            torch.nn.init.constant_(layer.bias.data, val=0.0)

    def create_tensor_dataset(self,mains, *appliances,):

        # Convert mains and appliances to PyTorch tensors and adjust dimensions as required for your model
        mains_tensor = torch.from_numpy(mains).float()
        appliance_tensors = [torch.from_numpy(appliance).float() for appliance in appliances]
        #rec_app_tensor = torch.from_numpy(rec_app).float()

        # Create the dataset
        dataset = TensorDataset(mains_tensor, *appliance_tensors)
        return dataset

    def train_epoch(self,model, train_loader, optimizer, loss_func):
        model.train()  # Set the model to training mode
        total_loss = 0
        for batch in train_loader:
            # Unpack batch data
            batch_mains, *batch_appliances, batch_rec_mains = batch

            # Move data to GPU if CUDA is available
            if torch.cuda.is_available():
                batch_mains = batch_mains.cuda()
                batch_appliances = [appliance.cuda() for appliance in batch_appliances]
                batch_rec_mains = batch_rec_mains.cuda()

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_mains)
            loss = sum([loss_func(appliance, pred) for appliance, pred in zip(batch_appliances, predictions[:-1])])
            rec_loss = loss_func(batch_rec_mains, predictions[-1])
            total_loss += (loss + rec_loss).item()

            # Backward pass and optimize
            (loss + rec_loss).backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate_model(self,model, valid_loader, loss_func):
        model.eval()  # Set the model to evaluation mode
        total_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch_mains, *batch_appliances, batch_rec_mains = batch

                if torch.cuda.is_available():
                    batch_mains = batch_mains.cuda()
                    batch_appliances = [appliance.cuda() for appliance in batch_appliances]
                    batch_rec_mains = batch_rec_mains.cuda()

                predictions = model(batch_mains)
                loss = sum([loss_func(appliance, pred) for appliance, pred in zip(batch_appliances, predictions[:-1])])
                rec_loss = loss_func(batch_rec_mains, predictions[-1])
                total_loss += (loss + rec_loss).item()

        avg_loss = total_loss / len(valid_loader)
        return avg_loss
    def train_chunk(self, mains, train_appliances,epochs,learning_rate,batch_size ):
        # initialize the network
        self.gen_model2.apply(self.initialize)

        self.rec_model.apply(self.initialize)

        self.mains_mean=mains.mean()

        self.mains_std=mains.std()

        #calculate the mean and the std
        for i, appliance in enumerate(train_appliances, start=1):
            appliance_mean = appliance.mean()
            appliance_std = appliance.std()
            setattr(self, f'appliance{i}_mean', appliance_mean)
            setattr(self, f'appliance{i}_std', appliance_std)

        for i, appliance in enumerate(train_appliances, start=1):

            setattr(self, f'appliance{i}', appliance)
        # divide the training and testing data
        dataset_splits = train_test_split(mains, *train_appliances, test_size=0.1, random_state=10)
        train_mains, valid_mains, *train_valid_appliances = dataset_splits
        train_appliances = train_valid_appliances[0::2]
        valid_appliances = train_valid_appliances[1::2]
        # Creating datasets
        train_dataset = self.create_tensor_dataset(train_mains, *train_appliances)
        valid_dataset = self.create_tensor_dataset(valid_mains, *valid_appliances)

        train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                      drop_last=True)
        valid_loader = tud.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                      drop_last=True)

        optimizer_G = torch.optim.Adam(self.gen_model2.parameters(),lr=learning_rate)


        eval_loss = torch.nn.MSELoss(reduction = 'mean')
        patience, best_loss = 0, None



        train_time=[]
        self.gen_model2.cuda()
        for epoch in range(epochs):
            # Earlystopping
            # if (patience == train_patience):
            #     print("val_loss did not improve after {} Epochs, thus Earlystopping is calling".format(train_patience))
            #     break
            #     # Train the model


            st = time.time()

            train_loss= self.train_epoch(self.gen_model2, train_loader, optimizer_G, eval_loss)
            print(train_loss)

            ed = time.time()
            train_epoch_time=ed-st
            train_time.append( train_epoch_time)

            #Evaluate the model
            self.gen_model2.eval()
            with torch.no_grad():
                current_loss = self.validate_model(self.gen_model2, valid_loader, eval_loss)
                if best_loss is None or current_loss < best_loss:
                    best_loss = current_loss
                    patience = 0

                else:
                    patience += 1

                print(f"Epoch: {epoch}, Valid Loss: {current_loss}")


                G_net_state_dict = self.gen_model2.state_dict()
                Rec_net_state_dict = self.rec_model.state_dict()


                G_path_state_dict = "./" +"G_net.pt"
                Rec_path_state_dict = "./" +"Rec_net.pt"

                torch.save(G_net_state_dict, G_path_state_dict)
                torch.save(Rec_net_state_dict, Rec_path_state_dict)





    def disaggregate(self, mains,test_appliances,**load_kwargs):
        #load the model
        model_path = "./G_net.pt"
        state_dict = torch.load(model_path)
        self.gen_model2.load_state_dict(state_dict)

        batchsize = self.batchsize

        st = time.time()

        self.gen_model2.eval()

        # Create test dataset and dataloader
        batch_size = mains.shape[0] if batchsize > mains.shape[0] else batchsize
        test_dataset = TensorDataset(torch.from_numpy(mains).float())
        test_loader = tud.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        results = []

        with torch.no_grad():
            for i, batch_mains in enumerate(test_loader):
                batch_mains_cuda = batch_mains[0].cuda()
                batch_size_actual = batch_mains[0].shape[0]
                if batch_mains[0].shape[0] == batch_size:

                    predictions= self.gen_model2(
                        batch_mains_cuda)



                else:

                    padding = torch.zeros(batch_size - batch_size_actual, 1, batch_mains[0].shape[2])
                    padded_mains = torch.cat((batch_mains[0], padding), 0).cuda()
                    predictions = self.gen_model2(padded_mains)[:batch_size_actual]

                if not results:
                    results = [pred.clone() for pred in predictions]
                else:
                    for i, pred in enumerate(predictions):
                        results[i] = torch.cat((results[i], pred), dim=0)
                return results

        ed = time.time()
        print("Inference Time consumption: {}s.".format(ed - st))

    def postprocess(self, results, train_meter_means, train_meter_stds):
        data = {}
        # package the results, train_meter_means, train_meter_stds together.

        for i, (result, mean, std) in enumerate(zip(results, train_meter_means, train_meter_stds)):
            power = result.to('cpu').numpy()
            adjusted_power = self.adjust_power(power, mean, std)
            data[f'appliance{i + 1}_power'] = adjusted_power
        return data



    @staticmethod
    def adjust_power(power, mean, std):
        l2 = 35  # the length of the window of the output
        n = len(power) + l2 - 1
        sum_arr = np.zeros(n)
        counts_arr = np.zeros(n)

        # Sliding window summing and counting
        for i in range(len(power)):
            sum_arr[i:i + l2] += power[i].flatten()
            counts_arr[i:i + l2] += 1

        # 计算平均并根据均值和标准差调整
        adjusted = (sum_arr / counts_arr) * std + mean
        return np.where(adjusted > 0, adjusted, 0)

    def save_results(self, data, filename='appliance_power_output.csv'):
            results_df = pd.DataFrame(data)
            output_path = './output_files/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            file_path = os.path.join(output_path, filename)
            results_df.to_csv(file_path, index=False)
            return file_path





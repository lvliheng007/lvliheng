from __future__ import print_function, division

import copy
import time
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split
import random
import sys
import os
from utils import *

import torch.utils.data as tud
from torch.utils.data.dataset import TensorDataset

import torch.nn as nn
import torch.nn.functional as F
import torch


from matplotlib import rcParams
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import h5py


from tensorboardX import SummaryWriter
from matplotlib import rcParams

import matplotlib.pyplot as plt
from nilmtk.legacy.disaggregate import Disaggregator

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.elecmeter import ElecMeterID
import metrics
#from shortseq2pointdisaggregator import ShortSeq2PointDisaggregator


cuda = True if torch.cuda.is_available() else False
USE_CUDA = torch.cuda.is_available()




cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):



    def  __init__(self, mains_length,appliance_length):
        # Refer to "ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring[C].The 32nd AAAI Conference on Artificial Intelligence"
        super(Generator, self).__init__()
        self.seq_length = appliance_length
        self.mains_length = mains_length
        self.meter1 = GeneratorB(self.mains_length, self.seq_length)
        self.meter2 = GeneratorB(self.mains_length, self.seq_length)
        self.meter3 = GeneratorB(self.mains_length, self.seq_length)
        self.meter4 = GeneratorB(self.mains_length, self.seq_length)
        self.meter5 = GeneratorB(self.mains_length, self.seq_length)



        self.rec= Generator_rec(self.mains_length, self.seq_length)

    def forward(self, x):


        meter1 = self.meter1(x)
        meter2= self.meter2(x)
        meter3 = self.meter3(x)
        meter4 = self.meter4(x)
        meter5 = self.meter5(x)


        rec=self.rec( meter1.view(-1,1, 35), meter2.view(-1,1,  35), meter3.view(-1,1, 35), meter4.view(-1,1,35), meter5.view(-1,1, 35))




        #return outputs




        return  meter1,meter2,meter3,meter4,meter5,rec


class GeneratorB(nn.Module):

    def __init__(self, mains_length, appliance_length):
        # Refer to "ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring[C].The 32nd AAAI Conference on Artificial Intelligence"
        super(GeneratorB, self).__init__()
        self.seq_length = appliance_length
        self.mains_length = mains_length

        self.conv = nn.Sequential(
            nn.ConstantPad1d((4, 4), 0),
            nn.Conv1d(1, 24, 9, stride=1),
            #nn.MaxPool1d(2, stride=2),
            # nn.BatchNorm1d(30),
            nn.ReLU(True),
            nn.ConstantPad1d((3, 3), 0),
            nn.Conv1d(24, 48, 7, stride=1),
            # nn.BatchNorm1d(30),
            #nn.MaxPool1d(2, stride=2),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(48,72, 5, stride=1),
            # nn.BatchNorm1d(30),
           # nn.MaxPool1d(2, stride=2),
            nn.ReLU(True),
            nn.ConstantPad1d((1, 1), 0),
            nn.Conv1d(72, 96, 3, stride=1),
            # nn.BatchNorm1d(30),
            #nn.MaxPool1d(2, stride=2),
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
        # x = self.dense(x1.view(-1, 50*self.seq_length))
        # main_rec=self.rec(x.view(-1, 1),x2)

        return x.view(-1, 35)



class Generator_rec(nn.Module):
    def __init__(self, mains_length, appliance_length):
        # Refer to "ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring[C].The 32nd AAAI Conference on Artificial Intelligence"
        super(Generator_rec, self).__init__()
        self.seq_length = appliance_length
        self.mains = mains_length
        self.meter_num =35
        #



        self.conv = nn.Sequential(
            nn.ConstantPad1d((4, 4), 0),
            nn.Conv1d(5, 30, 9, stride=1),

            # nn.BatchNorm1d(30),
            nn.ReLU(True),
            nn.ConstantPad1d((3, 3), 0),
            nn.Conv1d(30, 40, 7, stride=1),
            # nn.BatchNorm1d(30),

            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(40, 50,5, stride=1),
            # nn.BatchNorm1d(30),

            nn.ReLU(True),

        )

        self.dense = nn.Sequential(
            nn.Linear(50 *    self.meter_num,     5*self.meter_num),
            # nn.LayerNorm(3*self.seq_length),
            nn.ReLU(True),
            nn.Linear( 5* self.meter_num, 1),

        )







    def forward(self, x1, x2, x3, x4,x5, ):
        x = torch.cat((x1, x2, x3, x4,x5), dim=1)
        #x = torch.unsqueeze(x, dim=1)#.permute(0, 2, 1)
        #x = torch.unsqueeze(x, dim=1)#.permute(0,2,1)#####gaile

        x = self.conv(x)

        x = self.dense(x.view(-1,50*   self.meter_num))
        #x = self.dense(x.view(-1,50*   self.meter_num))




        return x.view(-1, 1)



d_real_list = []
real_list = []
fake_list = []
d_fake_list = []
g_list = []
real_tensor_list=[]
fake_tensor_list=[]
directory = './closed_best' + './'
class ShortSeq2PointDisaggregator(Disaggregator):

    def __init__(self):
        '''Initialize disaggregator
        '''
        self.MODEL_NAME = "pytorch Gan"
        self.window_size = 129
        self.batchsize =64

        self.gen_model2 = Generator(self.batchsize, self.window_size)
        self.writer = SummaryWriter(directory)
        self.n_critic=5
        #self.rec_model = GeneratorB(self.batchsize, self.window_size)
        self.rec_model = Generator_rec(self.batchsize, self.window_size)

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
    def train_chunk(self,appliance_name, mains, appliance1,appliance2,appliance3,appliance4,appliance5,epochs,batch_size, pretrain=True,
                    checkpoint_interval=3, train_patience=3):
        if not pretrain:
            #self.gen_model1.apply(self.initialize)
            self.gen_model2.apply(self.initialize)

            #self.rec_model.apply(self.initialize)
            # dis_model.apply(initialize)
            shape = np.array(mains).shape
        self.mains_mean=mains.mean()

        self.mains_std=mains.std()
        self.appliance1_mean=appliance1.mean()
        self.appliance1_std=appliance1.std()
        self.appliance2_mean = appliance2.mean()
        self.appliance2_std = appliance2.std()
        self.appliance3_mean = appliance3.mean()
        self.appliance3_std = appliance3.std()
        self.appliance4_mean = appliance4.mean()
        self.appliance4_std = appliance4.std()
        self.appliance5_mean = appliance5.mean()
        self.appliance5_std = appliance5.std()

        mains, appliance1,appliance2,appliance3,appliance4,appliance5,rec_mains = self.call_preprocessing(mains, appliance1,appliance2,appliance3,appliance4,appliance5,method='train')
        random_seed = 10
        MODEL_FOLDER = './save_model'
        IMAGE_FOLDER = './save_image'
        INSTANCE_FOLDER = None

        train_main = pd.concat(mains, axis=0).values
        mains = train_main.reshape((-1, self.window_size, 1))
        app_df1 = pd.concat(appliance1, axis=0)
        appliance1 = app_df1.values.reshape((-1,35))

        app_df2 = pd.concat(appliance2, axis=0)
        appliance2 = app_df2.values.reshape((-1,35))

        app_df3 = pd.concat(appliance3, axis=0)
        appliance3 = app_df3.values.reshape((-1,35))

        app_df4 = pd.concat(appliance4, axis=0)
        appliance4 = app_df4.values.reshape((-1,35))

        app_df5 = pd.concat(appliance5, axis=0)
        appliance5 = app_df5.values.reshape((-1,35))




        app_rec = pd.concat(rec_mains, axis=0)
        rec_app = app_rec.values.reshape((-1, 1))

        # Split the train and validation set
        train_mains, valid_mains, train_appliance1, valid_appliance1, train_appliance2, valid_appliance2,train_appliance3, valid_appliance3,train_appliance4, valid_appliance4,train_appliance5, valid_appliance5,\
        train_rec,valid_rec= train_test_split(mains, appliance1,appliance2,appliance3,appliance4,appliance5, rec_app,test_size=.1,
                                                                                      random_state=random_seed,)

        train_dataset = TensorDataset(torch.from_numpy(train_mains).float().permute(0, 2, 1),
                                      torch.from_numpy(train_appliance1).float(),torch.from_numpy(train_appliance2).float(),torch.from_numpy(train_appliance3).float(),
                                      torch.from_numpy(train_appliance4).float(),torch.from_numpy(train_appliance5).float(),torch.from_numpy(train_rec).float())
        train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        valid_dataset = TensorDataset(torch.from_numpy(valid_mains).float().permute(0, 2, 1),
                                      torch.from_numpy(valid_appliance1).float(), torch.from_numpy(valid_appliance2).float(),torch.from_numpy(valid_appliance3).float(),torch.from_numpy(valid_appliance4).float(),torch.from_numpy(valid_appliance5).float(),
                                       torch.from_numpy(valid_rec).float())
        valid_loader = tud.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
        writer = SummaryWriter(comment='train_visual', log_dir='runs/embedding_example/uk/MSELoss_Valid')
        # Create optimizer, loss function, and dataload
        optimizer_G = torch.optim.Adam(self.gen_model2.parameters(),lr=2*1e-4)
        #optimizer_G2 = torch.optim.Adam(self.rec_model.parameters(),lr=2*1e-4)
        #optimizer_G = torch.optim.Adam(itertools.chain(self.gen_model2.parameters()),lr=1e-4)

        #self.gen_model2.load_state_dict("./" + appliance_name + "_ukhouse2UK输入窗口变化129.pt")
        # state_dict = torch.load("./" + appliance_name + "_recukhouse2UK输入窗口变化71.pt")
        # self.gen_model2.load_state_dict(state_dict)
        #self.gen_model2.load_state_dict(torch.load("best二维卷积.pt"))


        adversarial_loss = torch.nn.BCELoss()
        reconstruction_loss = torch.nn.L1Loss()
        eval_loss = torch.nn.MSELoss(reduction = 'mean')
        patience, best_loss = 0, None
        train_loss = []
        valid_loss = []
        #
        train_pix1, train_pix2, train_pix3, train_pix4, train_pix5, train_pix6, train_pix7, train_pix8, train_pix9, train_pix10, train_pix11, train_pix12,train_pix13, train_pix14, train_pix15, train_pix16, train_pix17 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        valid_pix1,valid_pix2,valid_pix3, valid_pix4, valid_pix5, valid_pix6, valid_pix7, valid_pix8, valid_pix9, valid_pix10, valid_pix11, valid_pix12,valid_pix13, valid_pix14, valid_pix15,valid_pix16, valid_pix17 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

        train_loss_valid = []
        cnt_train, loss_sum_train = 0, 0
        loss_sum_forward_train, loss_sum_feedback_train = 0, 0
        train_time=[]

        for epoch in range(epochs):
            # Earlystopping
            # if (patience == train_patience):
            #     print("val_loss did not improve after {} Epochs, thus Earlystopping is calling".format(train_patience))
            #     break
            #     # Train the model
            # self.gen_model2.train()


            st = time.time()
            loss_recorder = []
            train_loss_recorder = 0
            loss_valid_recorder = 0
            #
            train_pix1_recorder,train_pix2_recorder, train_pix3_recorder,train_pix4_recorder, train_pix5_recorder,train_pix6_recorder, train_pix7_recorder,train_pix8_recorder, train_pix9_recorder,train_pix10_recorder, train_pix11_recorder,train_pix12_recorder, train_pix13_recorder,train_pix14_recorder, train_pix15_recorder,train_pix16_recorder,train_pix17_recorder = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            valid_pix1_recorder, valid_pix2_recorder,  valid_pix3_recorder, valid_pix4_recorder,  valid_pix5_recorder, valid_pix6_recorder,  valid_pix7_recorder, valid_pix8_recorder,  valid_pix9_recorder, valid_pix10_recorder,  valid_pix11_recorder, valid_pix12_recorder, valid_pix13_recorder, valid_pix14_recorder,  valid_pix15_recorder, valid_pix16_recorder, valid_pix17_recorder = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            b1 = []

            for i, (batch_mains, batch_appliance1, batch_appliance2,batch_appliance3,batch_appliance4,batch_appliance5,batch_rec_mains) in enumerate(train_loader):


                if USE_CUDA:
                    batch_mains = batch_mains.cuda()
                    batch_appliance1 = batch_appliance1.cuda()
                    batch_appliance2 = batch_appliance2.cuda()
                    batch_appliance3 = batch_appliance3.cuda()
                    batch_appliance4 = batch_appliance4.cuda()
                    batch_appliance5 = batch_appliance5.cuda()


                    batch_rec_mains = batch_rec_mains.cuda()

                    self.gen_model2.cuda()


                self.gen_model2.train()
               # self.rec_model.train()

                optimizer_G.zero_grad()
                # batch_pred1, batch_pred2, batch_pred3, batch_pred4, batch_pred5, batch_pred6, batch_pred7, batch_pred8, batch_pred9, batch_pred10, batch_pred11, batch_pred12, batch_pred13, batch_pred14, batch_pred15, batch_pred16, batch_pred17,batch_rec_pred = self.gen_model2(
                #     batch_mains)
                #batch_pred1,batch_pred2,batch_pred3,batch_pred4,batch_pred5,batch_pred6,batch_pred7,batch_pred8,batch_pred9,batch_pred10, batch_pred11,batch_pred12,batch_pred13,batch_pred14,batch_pred15,batch_pred16,batch_pred17,G_batch_rec_pred = self.gen_model2(batch_mains)
                batch_pred1,batch_pred2,batch_pred3,batch_pred4,batch_pred5,G_batch_rec_pred= self.gen_model2(batch_mains   )


                pix_loss1 = eval_loss(batch_appliance1, batch_pred1)
                pix_loss2 = eval_loss(batch_appliance2, batch_pred2)
                pix_loss3 = eval_loss(batch_appliance3, batch_pred3)
                pix_loss4 = eval_loss(batch_appliance4, batch_pred4)
                pix_loss5 = eval_loss(batch_appliance5, batch_pred5)


                G_rec_loss= eval_loss(batch_rec_mains, G_batch_rec_pred)
                gen_loss1=(pix_loss1 + pix_loss2 + pix_loss3 + pix_loss4 + pix_loss5+G_rec_loss )





                gen_loss1.backward()
                optimizer_G.step()
                print("one loss", gen_loss1)





                self.writer.add_scalar('Loss/gen1_loss',   gen_loss1, global_step=self.num_update_iteration)
                self.num_update_iteration+=1
                loss_recorder.append(gen_loss1.item())
                print('\rEpoch: %d, Loss: %0.4f, ' % (i, loss_recorder[-1]), end='')
                #print('genloss', gen_loss1)

                print('batch_appliance2', batch_appliance2[0:1])
                print('batch pred_', batch_pred2[0:1])

                print('epoch{},i{},batch{}'.format(epoch, i, len(train_loader)))

                train_loss_recorder += (pix_loss1+pix_loss2+pix_loss3+pix_loss4+pix_loss5).item()
                ####

                train_pix1_recorder += pix_loss1 .item()
                train_pix2_recorder +=  pix_loss2.item()
                train_pix3_recorder += pix_loss3.item()
                train_pix4_recorder +=  pix_loss4.item()
                train_pix5_recorder +=  pix_loss5.item()

                cnt_train += 1

            train_loss.append( train_loss_recorder /(len(train_loader)))
            train_pix1.append(train_pix1_recorder /(len(train_loader)))
            train_pix2.append(train_pix2_recorder /(len(train_loader)))
            train_pix3.append(train_pix3_recorder /(len(train_loader)))
            train_pix4.append(train_pix4_recorder /(len(train_loader)))
            train_pix5.append(train_pix5_recorder / (len(train_loader) ))


            ed = time.time()
            train_epoch_time=ed-st
            train_time.append( train_epoch_time)
            #train_loss.append(loss_recorder)
        #
            #Evaluate the model
            self.gen_model2.eval()
            with torch.no_grad():
                cnt, loss_sum = 0, 0
                loss_sum_forward, loss_sum_feedback = 0, 0
                for i, (  batch_mains_valid, batch_appliance1_valid, batch_appliance2_valid,batch_appliance3_valid,batch_appliance4_valid,batch_appliance5_valid,batch_rec_mains_valid) in enumerate(valid_loader):
                    if USE_CUDA:
                        batch_mains_valid = batch_mains_valid.cuda()
                        batch_appliance1_valid = batch_appliance1_valid.cuda()
                        batch_appliance2_valid = batch_appliance2_valid.cuda()
                        batch_appliance3_valid = batch_appliance3_valid.cuda()
                        batch_appliance4_valid = batch_appliance4_valid.cuda()
                        batch_appliance5_valid = batch_appliance5_valid.cuda()



                        batch_rec_mains_valid = batch_rec_mains_valid.cuda()


                    batch_pred_valid1, batch_pred_valid2, batch_pred_valid3, batch_pred_valid4, batch_pred_valid5,batch_pred_rec = self.gen_model2(batch_mains_valid)


                    pix_loss1_valid = eval_loss(batch_appliance1_valid, batch_pred_valid1)
                    pix_loss2_valid = eval_loss(batch_appliance2_valid, batch_pred_valid2)
                    pix_loss3_valid = eval_loss(batch_appliance3_valid, batch_pred_valid3)
                    pix_loss4_valid = eval_loss(batch_appliance4_valid, batch_pred_valid4)
                    pix_loss5_valid = eval_loss(batch_appliance5_valid, batch_pred_valid5)



                    #rec_loss_main = eval_loss(batch_rec_mains_valid, batch_pred_rec)

                    #loss = (pix_loss2+rec_loss_main)/2
                    loss_valid = pix_loss1_valid + pix_loss2_valid + pix_loss3_valid + pix_loss4_valid + pix_loss5_valid #+ pix_loss6_valid + pix_loss7_valid + pix_loss8_valid + pix_loss9_valid + pix_loss10_valid + pix_loss11_valid + pix_loss12_valid + pix_loss13_valid + pix_loss14_valid + pix_loss15_valid + pix_loss16_valid + pix_loss17_valid #+rec_loss_main
                    # loss_sum += loss
                    # loss_sum_forward += (loss -rec_loss_main)
                    # loss_sum_feedback += rec_loss_main

                    ###########乘以真实
                    real_m1_valid, test_m1_valid = batch_appliance1_valid * self.appliance1_std + self.appliance1_mean,batch_pred_valid1 * self.appliance1_std + self.appliance1_mean

                    real_m2_valid, test_m2_valid =batch_appliance2_valid * self.appliance2_std + self.appliance2_mean, batch_pred_valid2 * self.appliance2_std + self.appliance2_mean

                    real_m3_valid, test_m3_valid = batch_appliance3_valid * self.appliance3_std + self.appliance3_mean, batch_pred_valid3 * self.appliance3_std + self.appliance3_mean

                    real_m4_valid, test_m4_valid = batch_appliance4_valid * self.appliance4_std + self.appliance4_mean, batch_pred_valid4 * self.appliance4_std + self.appliance4_mean
                    real_m5_valid, test_m5_valid = batch_appliance5_valid * self.appliance5_std + self.appliance5_mean, batch_pred_valid5 * self.appliance5_std + self.appliance5_mean


                    mut_loss1_valid = eval_loss(  real_m1_valid, test_m1_valid)
                    mut_loss2_valid = eval_loss(  real_m2_valid, test_m2_valid)
                    mut_loss3_valid = eval_loss(  real_m3_valid, test_m3_valid)
                    mut_loss4_valid = eval_loss(  real_m4_valid, test_m4_valid)
                    mut_loss5_valid = eval_loss(  real_m5_valid, test_m5_valid)

                    loss_valid_recorder += loss_valid.item()

                    valid_pix1_recorder +=  mut_loss1_valid .item()
                    valid_pix2_recorder += mut_loss2_valid.item()
                    valid_pix3_recorder += mut_loss3_valid.item()
                    valid_pix4_recorder += mut_loss4_valid.item()
                    valid_pix5_recorder += mut_loss5_valid.item()

                    cnt += 1

            valid_loss.append(loss_valid_recorder/ (len(valid_loader) ))

            valid_pix1.append( valid_pix1_recorder / (len(valid_loader)))
            valid_pix2.append(valid_pix2_recorder / (len(valid_loader)))
            valid_pix3.append(valid_pix3_recorder / (len(valid_loader)))
            valid_pix4.append(valid_pix4_recorder / (len(valid_loader)))
            valid_pix5.append(valid_pix5_recorder / (len(valid_loader)))

            final_loss =   loss_valid_recorder / cnt
            final_loss_forward = loss_sum_forward / cnt
            final_loss_feedback = loss_sum_feedback / cnt

            if best_loss is None or final_loss < best_loss:
                best_loss = final_loss
                patience = 0
                net_state_dict = self.gen_model2.state_dict()
                #Dnet_state_dict = self.rec_model.state_dict()

                #path_state_dict = "./" + appliance_name + "_recukhouse2UK输入窗口变化129.pt"
                path_state_dict = "./" +"best二维卷积.pt"

                torch.save(net_state_dict, path_state_dict)
                #torch.save(Dnet_state_dict, Dpath_state_dict)
            else:
                patience = patience + 1
            print("Epoch: {}, Valid_Loss: {}, Time consumption: {}s.".format(epoch, final_loss, ed - st))

            # writer.add_scalars("MSELoss", {"Valid": final_loss}, epoch)

            # # For the visualization of training process
            # for name, param in self.gen_model2.named_parameters():
            #     writer.add_histogram(name + '_grad', param.grad, epoch)
            #     writer.add_histogram(name + '_data', param, epoch)
            # writer.add_scalars("MSELoss", {"Valid_alone": final_loss, "feedforward_alone": final_loss_forward,
            #                                "feedback_alone": final_loss_feedback}, epoch)
            # img = torch.rand([self.batchsize, 1, self.window_size], dtype=torch.float32).cuda()
            # writer.add_graph(self.gen_model2, input_to_model=img)
            # # Save checkpoint
            # if (checkpoint_interval != None) and ((epoch + 1) % checkpoint_interval == 0):
            #     checkpoint = {"model_state_dict": self.gen_model2.state_dict(),"epoch": epoch}
            #     path_checkpoint = "./" + appliance_name + "_REDD_cyc_{}_epoch.pkl".format(epoch)
            #     torch.save(checkpoint, path_checkpoint)
        # plt.plot(np.arange(epochs), (train_loss), label="train loss")




    def disaggregate(self, mains, output_data1store, output_data2store,output_data3store, output_data4store,output_data5store, output_data6store,output_data7store, output_data8store,output_data9store, output_data10store,output_data11store, output_data12store,output_data13store, output_data14store,output_data15store, output_data16store, output_data17store, meter1_metadata, meter2_metadata, meter3_metadata, meter4_metadata, meter5_metadata,appliance_name,**load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : a nilmtk.ElecMeter of aggregate data
        meter_metadata: a nilmtk.ElecMeter of the observed meter used for storing the metadata
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''



        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        # self.gen_model1.to('cpu')
        # self.gen_model2.to('cpu')
        # self.rec_model.to('cpu')
        self.gen_model2.cuda()
        self.rec_model.cuda()
        #self.gen_model2.load_state_dict(torch.load("./" + appliance_name + "_zuigood_UKjiange5ci_dict.pt"))

        data_is_available = False

        meter1_power_series =meter1_metadata.power_series(sample_period = sample_period)
        test_meter1chunk = next(meter1_power_series)
        test_meter1chunk.fillna(0, inplace=True)




        for chunk in mains.power_series(**load_kwargs):
            batchsize = self.batchsize
            # chunk.fillna(0, inplace=True)

            chunk.fillna(0, inplace=True)

            #test_meter1chunk = test_meter1chunk[:len(chunk)]
            timeframes.append(chunk.timeframe)

            measurement = chunk.name
            # chunk2 = normalize(chunk, mmax_main,min_main,mean_main,std_main)
            main = np.array(chunk)
            #test_main_list = self.call_preprocessing(main, submeters_lst1=None, submeters_lst2=None, method='test')
            test_main_list= self.call_preprocessing(main, submeters_lst1=None, submeters_lst2=None,submeters_lst3=None,submeters_lst4=None,submeters_lst5=None, method='test')

            test_main = pd.concat(test_main_list, axis=0).values
            test_mains = test_main.reshape((-1, self.window_size, 1))

            # test_appliances1  = pd.concat(test_app , axis=0).values
            # test_appliances1  = test_appliances1 .reshape((-1, 1))

            st = time.time()
            # self.gen_model1.eval()
            self.gen_model2.eval()

            # Create test dataset and dataloader
            batch_size = test_mains.shape[0] if batchsize > test_mains.shape[0] else batchsize
            test_dataset = TensorDataset(torch.from_numpy(test_mains).float().permute(0, 2, 1))
            test_loader = tud.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            with torch.no_grad():
                for i, batch_mains in enumerate(test_loader):
                    batch_mains_cuda=batch_mains[0].cuda()
                    #batch_pred2 = self.gen_model2(batch_mains[0])
                    #a2=batch_mains[0][:,:,0].detach()

                    if batch_mains[0].shape[0] == batch_size:
                        #batch_pred1, batch_pred2, batch_pred3, batch_pred4, batch_pred5, batch_pred6, batch_pred7, batch_pred8, batch_pred9, batch_pred10,batch_pred11, batch_pred12, batch_pred13, batch_pred14, batch_pred15, batch_pred16, batch_pred17, batch_pred18, batch_rec= self.gen_model2(batch_mains[0])
                        #batch_pred1, batch_pred2, batch_pred3, batch_pred4, batch_pred5, batch_pred6, batch_pred7, batch_pred8, batch_pred9, batch_pred10,batch_pred11, batch_pred12, batch_pred13, batch_pred14, batch_pred15, batch_pred16, batch_pred17,  batch_rec= self.gen_model2(batch_mains_cuda)
                        batch_pred1, batch_pred2, batch_pred3, batch_pred4, batch_pred5,batch_rec= self.gen_model2(batch_mains_cuda)



                    else:

                        a = torch.zeros(batch_size - batch_mains[0].shape[0], 1, batch_mains[0].shape[2])
                        b = torch.cat((batch_mains[0], a), 0).cuda()
                        #batch_pred1, batch_pred2, batch_pred3, batch_pred4, batch_pred5, batch_pred6, batch_pred7, batch_pred8, batch_pred9, batch_pred10,batch_pred11, batch_pred12, batch_pred13, batch_pred14, batch_pred15, batch_pred16, batch_pred17,  batch_rec = self.gen_model2(b)
                        batch_pred1, batch_pred2, batch_pred3, batch_pred4, batch_pred5,batch_rec= self.gen_model2(b)
                        # batch_rec = self.rec_model(batch_pred1, batch_pred2, batch_pred3, batch_pred4,
                        #                            batch_pred5, batch_pred6, batch_pred7, batch_pred8,
                        #                            batch_pred9, batch_pred10, batch_pred11, batch_pred12,
                        #                            batch_pred13, batch_pred14, batch_pred15, batch_pred16,
                        #                            batch_pred17)
                        batch_pred1 = batch_pred1[:batch_mains[0].shape[0], :]

                        batch_pred2 =  batch_pred2[:batch_mains[0].shape[0], :]



                        batch_pred3 = batch_pred3[:batch_mains[0].shape[0], :]
                        batch_pred4 = batch_pred4[:batch_mains[0].shape[0], :]
                        batch_pred5 = batch_pred5[:batch_mains[0].shape[0], :]


                    if i == 0:
                        res1 = batch_pred1

                        res2 =  batch_pred2

                        res3 = batch_pred3
                        res4 = batch_pred4
                        res5 = batch_pred5



                        #res_rec=batch_rec

                    else:
                        print('test_batch main', batch_mains)
                        # print('test_batch pred1',batch_pred1)
                        print('test_batch pred2', batch_pred1)
                        # print('test_batch pre3',batch_pred3)
                        res1 = torch.cat((res1, batch_pred1), dim=0)
                        res2=torch.cat((res2, batch_pred2), dim=0)
                        res3=torch.cat((res3, batch_pred3), dim=0)
                        res4=torch.cat((res4, batch_pred4), dim=0)
                        res5=torch.cat((res5, batch_pred5), dim=0)
                        # res6=torch.cat((res6, batch_pred6), dim=0)


            ed = time.time()
            print("Inference Time consumption: {}s.".format(ed - st))

            # appliance1_power = test(train_main)
            # appliance1_power[appliance1_power < 0] = 0
            # appliance1_power=res1.numpy()
            appliance1_power = res1.to('cpu').numpy()
            appliance2_power = res2.to('cpu').numpy()
            appliance3_power = res3.to('cpu').numpy()
            appliance4_power = res4.to('cpu').numpy()
            appliance5_power = res5.to('cpu').numpy()


            l2 =35
            n = len( appliance1_power) + l2 - 1

            sum_arr1 = np.zeros((n))
            sum_arr2 = np.zeros((n))
            sum_arr3 = np.zeros((n))
            sum_arr4 = np.zeros((n))
            sum_arr5 = np.zeros((n))
            counts_arr = np.zeros((n))

            for i in range( appliance1_power.shape[0]):
                sum_arr1[i:i + l2] +=  appliance1_power[i].flatten()
                sum_arr2[i:i + l2] +=  appliance2_power[i].flatten()
                sum_arr3[i:i + l2] +=  appliance3_power[i].flatten()
                sum_arr4[i:i + l2] +=  appliance4_power[i].flatten()
                sum_arr5[i:i + l2] +=  appliance5_power[i].flatten()
                counts_arr[i:i + l2] += 1
            for i in range(len(  sum_arr1)):
                sum_arr1[i] = sum_arr1[i] / counts_arr[i]
                sum_arr2[i] = sum_arr2[i] / counts_arr[i]
                sum_arr3[i] = sum_arr3[i] / counts_arr[i]
                sum_arr4[i] = sum_arr4[i] / counts_arr[i]
                sum_arr5[i] = sum_arr5[i] / counts_arr[i]


            #rec_agg=res_rec.numpy()
            appliance1_power =  sum_arr1 * self.appliance1_std + self.appliance1_mean

            #appliance2_power = appliance2_power * self.mains_std + self.mains_mean
            appliance2_power =  sum_arr2* self.appliance2_std + self.appliance2_mean

            appliance3_power =  sum_arr3 * self.appliance3_std + self.appliance3_mean
            appliance4_power =  sum_arr4 * self.appliance4_std + self.appliance4_mean
            appliance5_power =  sum_arr5 * self.appliance5_std + self.appliance5_mean







            valid_predictions1 = appliance1_power.flatten()
            valid_predictions2 = appliance2_power.flatten()
            #valid_predictions2 =  appliance2_power.flatten()[0:14366]
            valid_predictions3 = appliance3_power.flatten()
            valid_predictions4 = appliance4_power.flatten()
            valid_predictions5 = appliance5_power.flatten()



            # valid_predictions=valid_predictions.detach().numpy()
            appliance1_power = np.where(valid_predictions1 > 0, valid_predictions1, 0)
            appliance2_power = np.where(valid_predictions2 > 0, valid_predictions2, 0)
            appliance3_power = np.where(valid_predictions3 > 0, valid_predictions3, 0)
            appliance4_power = np.where(valid_predictions4 > 0, valid_predictions4, 0)
            appliance5_power = np.where(valid_predictions5 > 0, valid_predictions5, 0)
            plt.plot(appliance1_power[8850:9600])  # ，linewidth=5
            plt.plot( test_meter1chunk.values[8850:9600])
            plt.show()



            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter1_instance = meter1_metadata.instance()
            meter2_instance = meter2_metadata.instance()
            meter3_instance = meter3_metadata.instance()
            meter4_instance = meter4_metadata.instance()
            meter5_instance = meter5_metadata.instance()

            df1 = pd.DataFrame(appliance1_power, index=chunk.index, columns=cols, dtype="float32")
            df2 = pd.DataFrame(appliance2_power, index=chunk.index,columns=cols, dtype="float32")
            df3 = pd.DataFrame(appliance3_power, index=chunk.index, columns=cols, dtype="float32")
            df4 = pd.DataFrame(appliance4_power, index=chunk.index, columns=cols, dtype="float32")
            df5 = pd.DataFrame(appliance5_power, index=chunk.index, columns=cols, dtype="float32")


            key1 = '{}/elec/meter{}'.format(building_path, meter1_instance)
            key2 = '{}/elec/meter{}'.format(building_path, meter2_instance)
            key3 = '{}/elec/meter{}'.format(building_path, meter3_instance)
            key4 = '{}/elec/meter{}'.format(building_path, meter4_instance)
            key5 = '{}/elec/meter{}'.format(building_path, meter5_instance)


            output_data1store.append(key1, df1)
            output_data2store.append(key2, df2)
            output_data3store.append(key3, df3)
            output_data4store.append(key4, df4)
            output_data5store.append(key5, df5)



            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_data1store.append(key=mains_data_location, value=mains_df)
            output_data2store.append(key=mains_data_location, value=mains_df)
            output_data3store.append(key=mains_data_location, value=mains_df)
            output_data4store.append(key=mains_data_location, value=mains_df)
            output_data5store.append(key=mains_data_location, value=mains_df)
            output_data6store.append(key=mains_data_location, value=mains_df)
            output_data1store.append(key=mains_data_location, value=mains_df)
            output_data2store.append(key=mains_data_location, value=mains_df)
            output_data1store.append(key=mains_data_location, value=mains_df)
            output_data2store.append(key=mains_data_location, value=mains_df)
        if data_is_available:
            # self._save_metadata_for_disaggregation(
            #     output_datastore=output_data1store,
            #     sample_period=load_kwargs['sample_period'],
            #     measurement=measurement,
            #     timeframes=timeframes,
            #     building=mains.building(),
            #     meters=[meter1_metadata]
            # )
            self._save_metadata_for_disaggregation(
                output_datastore=output_data1store,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter1_metadata]
            )
            self._save_metadata_for_disaggregation(
                output_datastore=output_data2store,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter2_metadata]
            )
            self._save_metadata_for_disaggregation(
                output_datastore=output_data3store,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter3_metadata]
            )
            self._save_metadata_for_disaggregation(
                output_datastore=output_data4store,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter4_metadata]
            )
            self._save_metadata_for_disaggregation(
                output_datastore=output_data5store,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter5_metadata]
            )

    def call_preprocessing(self, mains_lst, submeters_lst1, submeters_lst2,submeters_lst3,submeters_lst4,submeters_lst5, method):

        sequence_length = self.window_size
        batch_size_length = (self.batchsize // 2)
        if method == 'train':
            # Seq2Seq Version

            # Preprocess the main and appliance data, the parameter 'overlapping' will be set 'True'
            bianyuan = []
            mains_df_list = []
            # for mains in mains_lst:
            #     new_mains = mains.values.flatten()
            mains_mean, mains_std = mains_lst.mean(), mains_lst.std()
            #n = sequence_length

            n =  sequence_length-35
            units_to_pad = n // 2
            new_mains = np.pad(mains_lst, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
            new_mains = np.array([new_mains[i:i + sequence_length] for i in range(len(new_mains) - sequence_length+ 1)])
            new_mains = (new_mains - self.mains_mean) / self.mains_std
            # new_mains = (new_mains - self.mains_mean) / (self.mains_max-self.mains_min)
            mains_df_list.append(pd.DataFrame(new_mains))

            n_load = 35
            units_to_pad_load = n_load // 2
            tuples_of_appliances1 = []
            new_meters1 = np.array(submeters_lst1)
            #new_meters1= np.pad(submeters_lst1, (  units_to_pad_load ,   units_to_pad_load ), 'constant', constant_values=(0, 0))
            new_meters1 = np.array([ new_meters1[i:i +n_load] for i in range(len(new_meters1) - n_load + 1)])
            new_meters1 = (new_meters1 - self.appliance1_mean) / self.appliance1_std
            tuples_of_appliances1.append(pd.DataFrame(new_meters1))



            tuples_of_appliances2 = []
            new_meters2 = np.array(submeters_lst2)
            # new_meters2 = np.pad(submeters_lst2, (units_to_pad_load, units_to_pad_load), 'constant',
            #                      constant_values=(0, 0))
            new_meters2 = np.array([new_meters2[i:i + n_load] for i in range(len(new_meters2) - n_load + 1)])
            new_meters2= (new_meters2 - self.appliance2_mean) / self.appliance2_std
            tuples_of_appliances2.append(pd.DataFrame(new_meters2))
            # submeters_lst2 = np.array(submeters_lst2)
            # new_meters2 = (submeters_lst2 - self.appliance2_mean) / self.appliance2_std
            # tuples_of_appliances2.append(pd.DataFrame(new_meters2))

            tuples_of_appliances3 = []
            new_meters3 = np.array(submeters_lst3)
            # new_meters3= np.pad(submeters_lst3, (units_to_pad_load, units_to_pad_load), 'constant',
            #                      constant_values=(0, 0))
            new_meters3 = np.array([new_meters3[i:i + n_load] for i in range(len(new_meters3) - n_load + 1)])
            new_meters3 = (new_meters3 - self.appliance3_mean) / self.appliance3_std
            tuples_of_appliances3.append(pd.DataFrame(new_meters3))
            # submeters_lst3 = np.array(submeters_lst3)
            # new_meters3 = (submeters_lst3 - self.appliance3_mean) / self.appliance3_std
            # tuples_of_appliances3.append(pd.DataFrame(new_meters3))

            tuples_of_appliances4= []
            new_meters4= np.array(submeters_lst4)
            # new_meters4 = np.pad(submeters_lst4, (units_to_pad_load, units_to_pad_load), 'constant',
            #                      constant_values=(0, 0))
            new_meters4 = np.array([new_meters4[i:i + n_load] for i in range(len(new_meters4) - n_load + 1)])
            new_meters4 = (new_meters4 - self.appliance4_mean) / self.appliance4_std
            tuples_of_appliances4.append(pd.DataFrame(new_meters4))
            # submeters_lst4 = np.array(submeters_lst4)
            # new_meters4 = (submeters_lst4 - self.appliance4_mean) / self.appliance4_std
            # tuples_of_appliances4.append(pd.DataFrame(new_meters4))

            tuples_of_appliances5 = []
            new_meters5 = np.array(submeters_lst5)
            # new_meters5= np.pad(submeters_lst5, (units_to_pad_load, units_to_pad_load), 'constant',
            #                      constant_values=(0, 0))
            new_meters5 = np.array([new_meters5[i:i + n_load] for i in range(len(new_meters5) - n_load + 1)])
            new_meters5 = (new_meters5 - self.appliance5_mean) / self.appliance5_std
            tuples_of_appliances5.append(pd.DataFrame(new_meters5))


            rec_mains_df_list = []

            rec_mains=np.array(mains_lst)[units_to_pad_load :len(mains_lst)-units_to_pad_load ]
            rec_mains= (rec_mains - self.mains_mean) / self.mains_std
            rec_mains_df_list.append(pd.DataFrame(rec_mains))


            return mains_df_list, tuples_of_appliances1, tuples_of_appliances2,tuples_of_appliances3,tuples_of_appliances4,tuples_of_appliances5,rec_mains_df_list


        else:
            # Preprocess the main data only, the parameter 'overlapping' will be set 'False'
            mains_df_list = []

            # for mains in mains_lst:
            new_mains = mains_lst
            n = sequence_length - 35
            units_to_pad = n // 2
            new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
            new_mains = np.array([new_mains[i:i + sequence_length] for i in range(len(new_mains) - sequence_length + 1)])

            new_mains = (new_mains - self.mains_mean) / self.mains_std



            mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list


print("========== OPEN DATASETS ============")

train = DataSet('D:/Git code/daima/zijixiede/ukdale.h5')
test = DataSet('D:/Git code/daima/zijixiede/ukdale.h5')
# #

####下面是房间3 ,4uk
# train.set_window(start="2013-02-28", end="2013-04-01")
# test.set_window(start="2013-04-01", end="2013-04-07")
# ####下面是房间5uk
# train.set_window(start="2014-08-01", end="2014-09-01")
# test.set_window(start="2014-09-01", end="2014-09-07")

# train.set_window(start="2013-06-01", end="2013-07-01")
# test.set_window(start="2013-07-01", end="2013-07-07")
#train.set_window(start="2013-6-29", end="2013-6-30")###画图
train.set_window(start="2013-6-15", end="2013-7-01")###房间2
#test.set_window(start="2013-7-01", end="2013-7-04")
test.set_window(start="2013-7-06", end="2013-7-07")
# train.set_window( end="2014-9-5")
#
# test.set_window(start="2014-9-1", end="2014-9-5 ")###房间5的日期
# train = DataSet('D:/Git code/zhen refit/refit.h5')
# test = DataSet('D:/Git code/zhen refit/refit.h5')
# train.set_window(start="2014-06-01", end="2014-07-01")
# test.set_window(start="2014-07-01", end="2014-07-07")##房间1
# train = DataSet('D:/Git code/daima/redd/low_freq/redd.h5')
# test = DataSet('D:/Git code/daima/redd/low_freq/redd.h5')
# test.set_window(start="2011-5-01",end="2011-5-02")###red房间1
# train.set_window(end="2011-4-30")
# train.set_window(end="2011-4-30")
# test.set_window(start="2011-4-30",end="2011-5-2")###red房间2
train_building = 2
test_building = 2
sample_period = 6
meter_key1 = 'washing machine'####[8850:9600]

meter_key2 = 'microwave'##[7800:8400]

meter_key3= 'dish washer'##(start="2013-7-03", end="2013-7-04")[11400:12200]
meter_key4 = 'fridge'#[8800:9300]
meter_key5 = 'kettle'

meter_key=['fridge']
train_elec = train.buildings[train_building].elec
test_elec = test.buildings[test_building].elec

train_elec_socker_main = train_elec.select_using_appliances(type=['washing machine','microwave','dish washer','fridge',
                                                              'kettle'])

test_elec_socker_main = test_elec.select_using_appliances(type=['washing machine','microwave','dish washer','fridge',
                                                              'kettle'])


train_mains = train_elec.mains()#.all_meters()[0]
test_mains = test_elec.mains()#.all_meters()[0]

train_meter_main = train_elec_socker_main
test_meter_main = test_elec_socker_main

train_meter1 = train_elec[meter_key1]
train_meter2 = train_elec[meter_key2]
train_meter3 = train_elec[meter_key3]
train_meter4 = train_elec[meter_key4]
train_meter5 = train_elec[meter_key5]

#train_meter18 = train_elec[meter_key18]
hecheng_power_series = train_meter_main.power_series(sample_period = sample_period)
test_hecheng_power_series = test_meter_main.power_series(sample_period = sample_period)




#
# test_meter1 = test_elec.select_using_appliances(type=['sockets','electric oven','unknown','microwave','dish washer','light','electric space heater','washer dryer','electric stove'])
test_meter1 = test_elec.submeters()[meter_key1]
test_meter2 = test_elec.submeters()[meter_key2]
test_meter3 = test_elec.submeters()[meter_key3]
test_meter4 = test_elec.submeters()[meter_key4]
test_meter5 = test_elec.submeters()[meter_key5]



main_power_series = train_mains.power_series(sample_period = sample_period)
meter1_power_series = train_meter1.power_series(sample_period = sample_period)
meter2_power_series = train_meter2.power_series(sample_period = sample_period)
meter3_power_series = train_meter3.power_series(sample_period = sample_period)
meter4_power_series = train_meter4.power_series(sample_period = sample_period)
meter5_power_series = train_meter5.power_series(sample_period = sample_period)


test_meter1_power_series = test_meter1.power_series(sample_period = sample_period)




mainchunk = next(main_power_series)
meter1chunk = next(meter1_power_series)
meter2chunk = next(meter2_power_series)
meter3chunk = next(meter3_power_series)
meter4chunk = next(meter4_power_series)
meter5chunk = next(meter5_power_series)
test_meter1_chunk = next(test_meter1_power_series)
# meter5chunk.plot()

hechengchunk = next(hecheng_power_series)
test_hechengchunk = next(test_hecheng_power_series)


mainchunk.fillna(0, inplace=True)
meter1chunk.fillna(0, inplace=True)

meter2chunk.fillna(0, inplace=True)
meter3chunk.fillna(0, inplace=True)
meter4chunk.fillna(0, inplace=True)
meter5chunk.fillna(0, inplace=True)
test_meter1_chunk.fillna(0, inplace=True)



hechengchunk.fillna(0, inplace=True)
test_hechengchunk.fillna(0, inplace=True)

mainchunk=mainchunk[:len(meter2chunk)]
meter1chunk=meter1chunk[:len(meter2chunk)]

meter3chunk=meter3chunk[:len(meter2chunk)]
meter4chunk=meter4chunk[:len(meter2chunk)]
meter5chunk=meter5chunk[:len(meter2chunk)]


hechengchunk=hechengchunk[:len(meter2chunk)]


disaggregator = ShortSeq2PointDisaggregator()
epochs =350


disaggregator.train_chunk(meter_key1,hechengchunk,meter1chunk,meter2chunk,meter3chunk,meter4chunk,meter5chunk, epochs,batch_size=64)
#disaggregator.train_chunk(meter_key1,test_hechengchunk[9030:9158],meter1chunk[9030:9158],meter2chunk[9030:9158],meter3chunk[9030:9158],meter4chunk[9030:9158],meter5chunk[9030:9158], epochs,batch_size=1)
#
print("========== DISAGGREGATE ============")
disag_filename1 = 'disag1-out.h5'
disag_filename2 = 'disag2-out.h5'
disag_filename3 = 'disag3-out.h5'
disag_filename4 = 'disag4-out.h5'
disag_filename5 = 'disag5-out.h5'
disag_filename6 = 'disag6-out.h5'
disag_filename7 = 'disag7-out.h5'
disag_filename8 = 'disag8-out.h5'
disag_filename9 = 'disag9-out.h5'
disag_filename10 = 'disag10-out.h5'
disag_filename11 = 'disag11-out.h5'
disag_filename12 = 'disag12-out.h5'
disag_filename13 = 'disag13-out.h5'
disag_filename14 = 'disag14-out.h5'
disag_filename15 = 'disag15-out.h5'
disag_filename16 = 'disag16-out.h5'
disag_filename17 = 'disag17-out.h5'


output1 = HDFDataStore(disag_filename1, 'w')
output2 = HDFDataStore(disag_filename2, 'w')
output3 = HDFDataStore(disag_filename3, 'w')
output4 = HDFDataStore(disag_filename4, 'w')
output5 = HDFDataStore(disag_filename5, 'w')
output6 = HDFDataStore(disag_filename6, 'w')
output7 = HDFDataStore(disag_filename7, 'w')
output8 = HDFDataStore(disag_filename8, 'w')
output9 = HDFDataStore(disag_filename9, 'w')
output10 = HDFDataStore(disag_filename10, 'w')
output11 = HDFDataStore(disag_filename11, 'w')
output12 = HDFDataStore(disag_filename12, 'w')
output13 = HDFDataStore(disag_filename13, 'w')
output14 = HDFDataStore(disag_filename14, 'w')
output15 = HDFDataStore(disag_filename15, 'w')
output16 = HDFDataStore(disag_filename16, 'w')
output17 = HDFDataStore(disag_filename17, 'w')

#disaggregator.disaggregate(test_mains, output, train_meter, sample_period=sample_period1)


disaggregator.disaggregate(test_elec_socker_main, output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11,output12,output13,output14,output15,output16,output17, test_meter1,test_meter2, test_meter3,test_meter4, test_meter5,meter_key1,sample_period=sample_period)
output1.close()
output2.close()
output3.close()
output4.close()
output5.close()
output6.close()
output7.close()
output8.close()
output9.close()
output10.close()
output11.close()
output12.close()
output13.close()
output14.close()
output15.close()
output16.close()
output17.close()





print("========== RESULTS ============")
disag_filename1= 'disag1-out.h5'
disag_filename2= 'disag2-out.h5'
disag_filename3= 'disag3-out.h5'
disag_filename4= 'disag4-out.h5'
disag_filename5= 'disag5-out.h5'
disag_filename6= 'disag6-out.h5'
disag_filename7= 'disag7-out.h5'
disag_filename8= 'disag8-out.h5'
disag_filename9= 'disag9-out.h5'
disag_filename10= 'disag10-out.h5'
disag_filename11= 'disag11-out.h5'
disag_filename12= 'disag12-out.h5'
disag_filename13= 'disag13-out.h5'
disag_filename14= 'disag14-out.h5'
disag_filename15= 'disag15-out.h5'
disag_filename16= 'disag16-out.h5'
disag_filename17= 'disag17-out.h5'




result1 = DataSet(disag_filename1)
res_elec1 = result1.buildings[test_building].elec
rpaf1 = metrics.recall_precision_accuracy_f1(res_elec1[meter_key1], test_elec[meter_key1])
print("============ Recall: {}".format(rpaf1[0]))
print("============ Precision: {}".format(rpaf1[1]))
print("============ Accuracy: {}".format(rpaf1[2]))
print("============ F1 Score: {}".format(rpaf1[3]))
#
print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec1[meter_key1], test_elec[meter_key1])))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec1[meter_key1], test_elec[meter_key1])))
print("============ Mean square error(in Watts): {}".format(metrics.mean_squared_error(res_elec1[meter_key1], test_elec[meter_key1])))
print("============ SAE(in Watts): {}".format(metrics.SAE(res_elec1[meter_key1], test_elec[meter_key1])))

result2 = DataSet(disag_filename2)
res_elec2 = result2.buildings[test_building].elec
rpaf2 = metrics.recall_precision_accuracy_f1(res_elec2[meter_key2],test_elec[meter_key2])
print("============ Recall: {}".format(rpaf2[0]))
print("============ Precision: {}".format(rpaf2[1]))
print("============ Accuracy: {}".format(rpaf2[2]))
print("============ F1 Score: {}".format(rpaf2[3]))

print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec2[meter_key2],test_elec[meter_key2])))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec2[meter_key2],test_elec[meter_key2])))
print("============ Mean square error(in Watts): {}".format(metrics.mean_squared_error(res_elec2[meter_key2],test_elec[meter_key2])))
print("============ SAE(in Watts): {}".format(metrics.SAE(res_elec2[meter_key2],test_elec[meter_key2])))
#

result4 = DataSet(disag_filename4)
res_elec4 = result4.buildings[test_building].elec
rpaf4 = metrics.recall_precision_accuracy_f1(res_elec4[meter_key4], test_elec[meter_key4])
print("============ Recall: {}".format(rpaf4[0]))
print("============ Precision: {}".format(rpaf4[1]))
print("============ Accuracy: {}".format(rpaf4[2]))
print("============ F1 Score: {}".format(rpaf4[3]))

print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec4[meter_key4], test_elec[meter_key4])))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec4[meter_key4], test_elec[meter_key4])))
print("============ Mean square error(in Watts): {}".format(metrics.mean_squared_error(res_elec4[meter_key4], test_elec[meter_key4])))
print("============ SAE(in Watts): {}".format(metrics.SAE(res_elec4[meter_key4], test_elec[meter_key4])))

result5 = DataSet(disag_filename5)
res_elec5 = result5.buildings[test_building].elec
rpaf5 = metrics.recall_precision_accuracy_f1(res_elec5[meter_key5], test_elec[meter_key5])
print("============ Recall: {}".format(rpaf5[0]))
print("============ Precision: {}".format(rpaf5[1]))
print("============ Accuracy: {}".format(rpaf5[2]))
print("============ F1 Score: {}".format(rpaf5[3]))

print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec5[meter_key5], test_elec[meter_key5])))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec5[meter_key5], test_elec[meter_key5])))
print("============ Mean square error(in Watts): {}".format(metrics.mean_squared_error(res_elec5[meter_key5], test_elec[meter_key5])))
print("============ SAE(in Watts): {}".format(metrics.SAE(res_elec5[meter_key5], test_elec[meter_key5])))


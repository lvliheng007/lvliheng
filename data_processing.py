from __future__ import print_function, division

import pandas as pd
import numpy as np


from nilmtk import DataSet
#
def dataset(train_start,train_end,test_start,test_end,building,meter_key1,meter_key2,meter_key3,meter_key4,meter_key5,window_size,batch_size,method):
    print("========== OPEN DATASETS ============")

    train = DataSet('D:/Git code/daima/zijixiede/ukdale.h5')
    test = DataSet('D:/Git code/daima/zijixiede/ukdale.h5')


    train.set_window(start=train_start, end=train_end)  ###房间2
    # test.set_window(start="2013-7-01", end="2013-7-04")
    test.set_window(start=test_start, end=test_end)

    train_building = building
    test_building = building
    sample_period = 6


    train_elec = train.buildings[train_building].elec
    test_elec = test.buildings[test_building].elec

    train_elec_socker_main = train_elec.select_using_appliances(
        type=[meter_key1, meter_key2, meter_key3, meter_key4,
              meter_key5])

    test_elec_socker_main = test_elec.select_using_appliances(
        type=[meter_key1, meter_key2, meter_key3, meter_key4,
              meter_key5])



    train_meter_main = train_elec_socker_main
    test_meter_main = test_elec_socker_main
    #
    train_meter1 = train_elec[meter_key1]
    train_meter2 = train_elec[meter_key2]
    train_meter3 = train_elec[meter_key3]
    train_meter4 = train_elec[meter_key4]
    train_meter5 = train_elec[meter_key5]


    hecheng_power_series = train_meter_main.power_series(sample_period=sample_period)
    test_hecheng_power_series = test_meter_main.power_series(sample_period=sample_period)



    test_meter1 = test_elec.submeters()[meter_key1]
    test_meter2 = test_elec.submeters()[meter_key2]
    test_meter3 = test_elec.submeters()[meter_key3]
    test_meter4 = test_elec.submeters()[meter_key4]
    test_meter5 = test_elec.submeters()[meter_key5]


    meter1_power_series = train_meter1.power_series(sample_period=sample_period)
    meter2_power_series = train_meter2.power_series(sample_period=sample_period)
    meter3_power_series = train_meter3.power_series(sample_period=sample_period)
    meter4_power_series = train_meter4.power_series(sample_period=sample_period)
    meter5_power_series = train_meter5.power_series(sample_period=sample_period)

    test_meter1_power_series = test_meter1.power_series(sample_period=sample_period)
    test_meter2_power_series = test_meter2.power_series(sample_period=sample_period)
    test_meter3_power_series = test_meter3.power_series(sample_period=sample_period)
    test_meter4_power_series = test_meter4.power_series(sample_period=sample_period)
    test_meter5_power_series = test_meter5.power_series(sample_period=sample_period)


    meter1chunk = next(meter1_power_series)
    meter2chunk = next(meter2_power_series)
    meter3chunk = next(meter3_power_series)
    meter4chunk = next(meter4_power_series)
    meter5chunk = next(meter5_power_series)
    test_meter1_chunk = next(test_meter1_power_series)
    test_meter2_chunk = next(test_meter2_power_series)
    test_meter3_chunk = next(test_meter3_power_series)
    test_meter4_chunk = next(test_meter4_power_series)
    test_meter5_chunk = next(test_meter5_power_series)
    # meter5chunk.plot()

    hechengchunk = next(hecheng_power_series)
    test_hechengchunk = next(test_hecheng_power_series)

    #
    meter1chunk.fillna(0, inplace=True)

    meter2chunk.fillna(0, inplace=True)
    meter3chunk.fillna(0, inplace=True)
    meter4chunk.fillna(0, inplace=True)
    meter5chunk.fillna(0, inplace=True)
    test_meter1_chunk.fillna(0, inplace=True)
    test_meter2_chunk.fillna(0, inplace=True)
    test_meter3_chunk.fillna(0, inplace=True)
    test_meter4_chunk.fillna(0, inplace=True)
    test_meter5_chunk.fillna(0, inplace=True)

    hechengchunk.fillna(0, inplace=True)
    test_hechengchunk.fillna(0, inplace=True)

    mains_mean = hechengchunk.mean()

    mains_std = hechengchunk.std()
    appliance1_mean = meter1chunk.mean()
    appliance1_std = meter1chunk.std()
    appliance2_mean = meter2chunk.mean()
    appliance2_std = meter2chunk.std()
    appliance3_mean = meter3chunk.mean()
    appliance3_std = meter3chunk.std()
    appliance4_mean = meter4chunk.mean()
    appliance4_std = meter4chunk.std()
    appliance5_mean = meter5chunk.mean()
    appliance5_std = meter5chunk.std()

    mains_mean = hechengchunk.mean()

    mains_std = hechengchunk.std()
    # appliance1_mean = meter1chunk.mean()
    # appliance1_std = meter1chunk.std()

    def closed_call_preprocessing( mains_lst, submeters_lst1, submeters_lst2, submeters_lst3, submeters_lst4,
                           submeters_lst5, mains_mean,mains_std,method):

        sequence_length = window_size
        batch_size_length = (batch_size // 2)
        if method == 'train':
            n = sequence_length - 35
            n_load = 35
            # Preprocess the main and appliance data, the parameter 'overlapping' will be set 'True'
            bianyuan = []
            mains_df_list = []
            # for mains in mains_lst:
            #     new_mains = mains.values.flatten()
            mains_mean, mains_std = mains_lst.mean(), mains_lst.std()
            # n = sequence_length

            units_to_pad = n // 2
            new_mains = np.pad(mains_lst, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
            new_mains = np.array(
                [new_mains[i:i + sequence_length] for i in range(len(new_mains) - sequence_length + 1)])
            new_mains = (new_mains - mains_mean) / mains_std
            mains_df_list.append(pd.DataFrame(new_mains))

            units_to_pad_load = n_load // 2
            tuples_of_appliances1 = []
            new_meters1 = np.array(submeters_lst1)
            new_meters1 = np.array([new_meters1[i:i + n_load] for i in range(len(new_meters1) - n_load + 1)])
            new_meters1 = (new_meters1 - appliance1_mean) / appliance1_std
            tuples_of_appliances1.append(pd.DataFrame(new_meters1))

            tuples_of_appliances2 = []
            new_meters2 = np.array(submeters_lst2)
            new_meters2 = np.array([new_meters2[i:i + n_load] for i in range(len(new_meters2) - n_load + 1)])
            new_meters2 = (new_meters2 - appliance2_mean) / appliance2_std
            tuples_of_appliances2.append(pd.DataFrame(new_meters2))

            tuples_of_appliances3 = []
            new_meters3 = np.array(submeters_lst3)

            new_meters3 = np.array([new_meters3[i:i + n_load] for i in range(len(new_meters3) - n_load + 1)])
            new_meters3 = (new_meters3 - appliance3_mean) / appliance3_std
            tuples_of_appliances3.append(pd.DataFrame(new_meters3))

            tuples_of_appliances4 = []
            new_meters4 = np.array(submeters_lst4)

            new_meters4 = np.array([new_meters4[i:i + n_load] for i in range(len(new_meters4) - n_load + 1)])
            new_meters4 = (new_meters4 - appliance4_mean) / appliance4_std
            tuples_of_appliances4.append(pd.DataFrame(new_meters4))

            tuples_of_appliances5 = []
            new_meters5 = np.array(submeters_lst5)

            new_meters5 = np.array([new_meters5[i:i + n_load] for i in range(len(new_meters5) - n_load + 1)])
            new_meters5 = (new_meters5 - appliance5_mean) / appliance5_std
            tuples_of_appliances5.append(pd.DataFrame(new_meters5))

            rec_mains_df_list = []

            rec_mains = np.array(mains_lst)[units_to_pad_load:len(mains_lst) - units_to_pad_load]
            rec_mains = (rec_mains - mains_mean) / mains_std
            rec_mains_df_list.append(pd.DataFrame(rec_mains))

            return mains_df_list, tuples_of_appliances1, tuples_of_appliances2, tuples_of_appliances3, tuples_of_appliances4, tuples_of_appliances5, rec_mains_df_list

        else:
            # Preprocess the main data only, the parameter 'overlapping' will be set 'False'
            mains_df_list = []

            # for mains in mains_lst:
            new_mains = mains_lst
            n = sequence_length - 35
            units_to_pad = n // 2
            new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
            new_mains = np.array(
                [new_mains[i:i + sequence_length] for i in range(len(new_mains) - sequence_length + 1)])

            new_mains = (new_mains - mains_mean) / mains_std

            mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list

    def ddpg_call_preprocessing( mains_lst, submeters_lst1, mains_mean, mains_std, method):

            sequence_length = window_size
            batch_size_length = (batch_size // 2)
        # if method == 'train':
            # Seq2Seq Version

            # Preprocess the main and appliance data, the parameter 'overlapping' will be set 'True'
            bianyuan = []
            mains_df_list = []
            # for mains in mains_lst:
            #     new_mains = mains.values.flatten()
            mains_mean, mains_std = mains_lst.mean(), mains_lst.std()
            # n = sequence_length

            n = sequence_length
            units_to_pad = n // 2
            new_mains = np.pad(mains_lst, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
            new_mains = np.array(
                [new_mains[i:i + sequence_length] for i in range(len(new_mains) - sequence_length + 1)])
            new_mains = (new_mains - mains_mean) / mains_std
            # new_mains = (new_mains - self.mains_mean) / (self.mains_max-self.mains_min)
            mains_df_list.append(pd.DataFrame(new_mains))

            n_load = 1
            units_to_pad_load = n_load // 2
            tuples_of_appliances1 = []
            new_meters1 = np.array(submeters_lst1)
            # new_meters1= np.pad(submeters_lst1, (  units_to_pad_load ,   units_to_pad_load ), 'constant', constant_values=(0, 0))
            new_meters1 = np.array([new_meters1[i:i + n_load] for i in range(len(new_meters1) - n_load + 1)])
            new_meters1 = (new_meters1 - appliance1_mean) / appliance1_std
            tuples_of_appliances1.append(pd.DataFrame(new_meters1))

            return mains_df_list, tuples_of_appliances1




    if method == 'closed_train':
        mains, appliance1, appliance2, appliance3, appliance4, appliance5, rec_mains = closed_call_preprocessing(hechengchunk,
                                                                                                           meter1chunk,
                                                                                                           meter2chunk,
                                                                                                           meter3chunk,
                                                                                                           meter4chunk,
                                                                                                           meter5chunk,mains_mean,mains_std,
                                                                                                           method='train')
        train_main = pd.concat(mains, axis=0).values
        mains = train_main.reshape((-1, window_size, 1))
        app_df1 = pd.concat(appliance1, axis=0)
        appliance1 = app_df1.values.reshape((-1, 35))

        app_df2 = pd.concat(appliance2, axis=0)
        appliance2 = app_df2.values.reshape((-1, 35))

        app_df3 = pd.concat(appliance3, axis=0)
        appliance3 = app_df3.values.reshape((-1, 35))

        app_df4 = pd.concat(appliance4, axis=0)
        appliance4 = app_df4.values.reshape((-1, 35))

        app_df5 = pd.concat(appliance5, axis=0)
        appliance5 = app_df5.values.reshape((-1, 35))

        app_rec = pd.concat(rec_mains, axis=0)
        rec_app = app_rec.values.reshape((-1, 1))

        return mains, appliance1,appliance2, appliance3,appliance4,appliance5,rec_app,test_elec_socker_main ,test_meter1_chunk,test_meter2_chunk, test_meter3_chunk,    test_meter4_chunk,    test_meter5_chunk
    elif method == 'ddpg_train':
        mains, appliance1 = ddpg_call_preprocessing(hechengchunk, meter1chunk, mains_mean, mains_std, method='train')

        train_main = pd.concat(mains, axis=0).values
        mains = train_main.reshape((-1, 1, window_size))
        app_df1 = pd.concat(appliance1, axis=0)
        appliance1 = app_df1.values.reshape((-1, 1))


        return mains, appliance1,appliance1_mean,appliance1_std
    elif method == 'ddpg_test':
        mains, appliance1 = ddpg_call_preprocessing(test_hechengchunk,   test_meter1_chunk, mains_mean, mains_std, method='test')

        train_main = pd.concat(mains, axis=0).values
        mains = train_main.reshape((-1, 1, window_size))
        app_df1 = pd.concat(appliance1, axis=0)
        appliance1 = app_df1.values.reshape((-1, 1))


        return mains, appliance1,appliance1_mean,appliance1_std


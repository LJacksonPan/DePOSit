import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
from torch.utils.data import Dataset
from utils import ang2joint
import pickle as pkl
from os import walk


class ATM(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, split=0, miss_rate=0.2, all_data=False, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 train_ratio=0.7, test_ratio=0.2, inc_quaternion = True,
                 excl_qua_out = False):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir, 'AMASS') + '/'
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.miss_rate = miss_rate
        # self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)  # start from 4 for 17 joints, removing the non moving ones
        seq_len = self.in_n + self.out_n


        if size == None:
            self.seq_len = input_n
            self.label_len = 0
            self.pred_len = output_n
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]


        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.inc_quaternion = inc_quaternion
        self.quaternion_columns = ['Quaternion_x', 'Quaternion_y', 'Quaternion_z', 'Quaternion_w']
        self.excl_qua_out = excl_qua_out
        # print(self.test_ratio)

        self.root_path = data_dir
        self.__read_data__()

    
    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]
        data_list = []
        stamp_list = []
        self.data = list()

        for file in files:
            df_raw = pd.read_csv(os.path.join(self.root_path, file))
            df_output_raw = df_raw.copy()

            if not self.inc_quaternion or self.excl_qua_out:
                df_output_raw = df_output_raw.drop(columns=[col for col in df_output_raw if "Quaternion" in col])

            if not self.inc_quaternion:
                df_raw = df_raw.drop(columns=[col for col in df_raw if "Quaternion" in col])

            cols = list(df_raw.columns)
            # cols.remove(self.target)
            # cols.remove('date')
            # df_raw = df_raw[['date'] + cols + [self.target]]
            # print(cols)
            num_train = int(len(df_raw) * self.train_ratio)
            num_test = int(len(df_raw) * self.test_ratio)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
                cols_out = df_output_raw.columns[1:]
                df_out = df_output_raw[cols_out]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                # print(self.scaler.mean_)
                # exit()
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values
            
            if self.scale:
                train_output_data = df_out[border1s[0]:border2s[0]]
                self.scaler.fit(train_output_data.values)
                output_data = self.scaler.transform(df_out.values)
            else:
                output_data = df_out.values

            # df_stamp = df_raw[['date']][border1:border2]
            # df_stamp['date'] = pd.to_datetime(df_stamp.date)
            # if self.timeenc == 0:
            #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            #     data_stamp = df_stamp.drop(['date'], axis=1).values
            # elif self.timeenc == 1:
            #     data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            #     data_stamp = data_stamp.transpose(1, 0)

            self.data_x = data[border1:border2]
            self.data_y = output_data[border1:border2]  # Output data
            # self.data_stamp = data_stamp
            self.data_stamp = df_raw[['DeltaTime']].values[border1:border2]

            for i in range(len(self.data_x) - self.seq_len - self.pred_len + 1):
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x = self.data_x[s_begin:s_end]
                seq_y = self.data_y[r_begin:r_end]
                seq_x_mark = self.data_stamp[s_begin:s_end]
                seq_y_mark = self.data_stamp[r_begin:r_end]

                self.data.append((seq_x, seq_y, seq_x_mark, seq_y_mark))


    def __getitem__(self, index):
        # s_begin = index
        # s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        pose = torch.cat([torch.tensor(self.data[index][0]), torch.tensor(self.data[index][1])])

        mask = np.zeros((pose.shape[0], pose.shape[1]))
        mask[0:self.in_n, :] = 1
        mask[self.in_n:self.in_n + self.out_n, :] = 0

        data = {
            "pose": pose,
            "mask": mask,
            "timepoints": np.arange(self.in_n + self.out_n)
        }

        return data

        # return self.data[index][0], self.data[index][1], self.data[index][2], self.data[index][3]

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.data)

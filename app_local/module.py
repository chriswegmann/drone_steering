from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from os import listdir
import re
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.interpolate import interp1d
import math


class XCentralizer(BaseEstimator, TransformerMixin):

    def __init__(self, x_columns):
        self.x_columns = x_columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):  # x is df
        shift = x[["rightShoulder_x", "leftShoulder_x"]].sum(axis=1)/2
        for col in self.x_columns:
            x[col] = x[col] - shift
        return x


class YCentralizer(BaseEstimator, TransformerMixin):

    def __init__(self, y_columns):
        self.y_columns = y_columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):  # x is df
        shift = x[["rightShoulder_y", "leftShoulder_y",
                   "leftHip_y", "rightHip_y"]].sum(axis=1)/4
        for col in list(set(self.y_columns)-set(["label"])):
            x[col] = x[col] - shift
        return x


class YScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):  # x is df
        shoulder_y = x[["rightShoulder_y", "leftShoulder_y"]].sum(axis=1)/2
        hip_y = x[["leftHip_y", "rightHip_y"]].sum(axis=1)/2
        y_dist = hip_y - shoulder_y

        for col in list(set(x.columns)-set(["label"])):
            x[col] /= y_dist
        return x


class LabelGeneratorFramebased():
    
    def __init__(self, data_df, labels_df, ms_per_frame):
        
        # stores the original data and the used framerate.
        self.data_df = data_df
        self.labels_df = labels_df
        self.ms_per_frame = ms_per_frame

        time_cols = ['ms_since_last_frame','ms_since_start']
        for col in time_cols:
            if col in self.data_df.columns:
                self.data_df.drop(col, axis = 1, inplace = True)

        steps = math.ceil(2000/self.ms_per_frame) + 1
        count_cols = self.data_df.columns.shape[0]
        count_rows = self.data_df.shape[0] - steps + 1

        print('steps: ' + str(steps))
        print('self.data_df.shape[0]: ' + str(self.data_df.shape[0]))
        
        # transforms labels from start/end format to a label per row
        self.__y = np.zeros(count_rows)
        for index, row in self.labels_df.iterrows():
            for i in range(row['real_start'], row['real_end'] + 1):
                self.__y[i] = row['label']

        # transforms frames into frame sets containing multiple frames
        self.__X = np.zeros((count_rows,steps,count_cols))
        data_np = self.data_df.loc[:,:].values
        for i in range(count_rows):
            self.__X[i] = data_np[i:i+steps,:]

        self.__feature_names = self.data_df.columns


    @property
    def X(self):
        return self.__X
        
    @property
    def y(self):
        return self.__y

    @property
    def feature_names(self):
        return self.__feature_names    

    # this propery is not labelled despite its name. It's named like this due to compatibility with LabelGenerator. $
    # Needs to be re-worked after we decided if take a frame- or time-based labelling approach.
    @property
    def labeled_data(self): 
        return self.data_df    


        
class LabelGenerator():
    
    def __init__(self, data, raw_labels, ms_per_frame):
        
        # stores the original data and the used framerate.
        self.data = data
        self.raw_labels = raw_labels 
        self.ms_per_frame = ms_per_frame
        
        # creates label-Dataframe whose "from"/"to" columns will be used for labelling.
        self.__label_df = pd.DataFrame(
            columns = ["label","real_start","real_end"]
        )

        self.__label_df[["label","real_start","real_end"]] =\
            self.raw_labels[["label","real_start","real_end"]]

        self.__label_df["real_start"] = self.__label_df["real_start"].apply(lambda x: x*1000)
        self.__label_df["real_end"] = self.__label_df["real_end"].apply(lambda x: x*1000)
        
        self.__label_df["real_start"] = np.round(self.__label_df["real_start"],0).astype("int32")
        self.__label_df["real_end"] = np.round(self.__label_df["real_end"],0).astype("int32")
        
        
        # default variables
        self.__is_fitted = False
        self.__is_labeled = False
        self.__is_finalized = False
        
        
        
        
    # PUBLIC method that creates two Dataframe, __label_df and __label_info
    # __label_df --> includes the acceptable range with the columns "from" and "to".
    #              any value between "from" and "to" is an acceptable movement endpoint and can be labeled accordingly.
    #              USAGE: this data frame will be used to create the final training data set
    # __label_info --> provides additional information about how the slacks were calculated
    #              USAGE: this data frame is for debugging mainly
    def fit_range(self, tolerance_range, max_error):
        
        self.__is_fitted = False
        self.__is_labeled = False
        self.__is_finalized = False
        
        tolerance_range = self.__check_variable("tolerance_range", tolerance_range)
        max_error = self.__check_variable("max_error", max_error)
        
        diff = self.__label_df["real_end"] - self.__label_df["real_start"] - 2000
        
        lower_slack, upper_slack, tolerance_range_ind = self.__calc_range(diff)
    
        self.__label_df["from"] = (self.__label_df["real_end"] + lower_slack).astype("int32")
        self.__label_df["to"] = (self.__label_df["real_end"] + upper_slack).astype("int32")
        self.__label_df["ignore"] = (abs(diff) >= self.max_error)

    
        # creates a Dataframe to store the used slacks for each labeled sample
        self.__label_info = pd.DataFrame(
            columns=["diff","l_slack","u_slack","tol_range_indicator"]
        )
        
        self.__label_info["diff"] = diff
        self.__label_info["l_slack"] = lower_slack 
        self.__label_info["u_slack"] = upper_slack
        self.__label_info["tol_range_indicator"] = tolerance_range_ind
        
        
        self.__set_error_df()

        self.__is_fitted = True


    # PRIVATE METHOD
    # calculates the acceptance interval for each sample
    def __calc_range(self, diff):
        indicator = (diff >= 0)
       
        lower_slack = - indicator * diff
        upper_slack = - (~indicator) * diff

        current_range = upper_slack - lower_slack
        range_delta = self.tolerance_range - current_range
        tolerance_range_ind = (range_delta > 0)

        lower_slack = lower_slack - range_delta//2 * tolerance_range_ind
        upper_slack = upper_slack + range_delta//2 * tolerance_range_ind
        
        return lower_slack.astype("int32"), upper_slack.astype("int32"), tolerance_range_ind
    
  
    # PRIVATE method that returns default variable values if no value is provided 
    #   and sets instance variables otherwise:
    #   symmetric_slack, tolerance_range, max_error
    def __check_variable(self, identifier, value):
        
        if identifier == "tolerance_range":
            if not value:
                value = self.tolerance_range
            else:
                self.tolerance_range = value
        
        elif identifier == "max_error":
            if not value:
                value = self.max_error
            else:
                self.max_error = value
            
        return value
    
    
    # creates the cutoff Dataframe with additional information about all movements that exceeded the max_error
    #   ... specified on initialization
    # USAGE: any movement in the error_df will not yield any labeled data. In the future it might even be removed 
    #        ... completely from the data (not implemented yet)
    def __set_error_df(self):
          
        self.__error_df = self.__label_df[abs(self.__label_info["diff"])>= self.max_error]\
            [["real_start","real_end"]]
        self.__error_df["start_idx"] =\
            (self.__error_df["real_start"]//self.ms_per_frame).apply(int)
        self.__error_df["start_calc"] =\
            self.__error_df["start_idx"] * self.ms_per_frame
        self.__error_df["end_idx"] =\
            np.ceil(self.__error_df["real_end"]/self.ms_per_frame).apply(int)
        self.__error_df["end_calc"] =\
            self.__error_df["end_idx"] * self.ms_per_frame

        
    
    # calls the Error-Dataframe with additional information about all movements that exceeded the max_error
    # this method can only be called after the Error-Dataframe has been created by calling the get_error_df method
    @property
    def error_df(self):
        
        if not self.__is_fitted:
            raise ValueError("You have to fit the range with the fit_range method")
            
        else:
            return self.__error_df
        
 
    # PUBLIC method that creates the PRIVATE labeled-Data Dataframe. 
    # This dataframe can be called by the get_labeled_data method
    # this is the the data frame that provides a label for each wire frame from posenet
    def set_labels(self):
        
        if not self.__is_fitted:
            raise ValueError("You have to fit the range with the fit_range method")
            
        self.__is_labeled = False
        self.__is_finalized = False
            
        _T = pd.DataFrame(columns=["time"])
        _T["time"] = self.data["ms_since_start"]
        _T["_key_"] = 0
        _l = self.__label_df[["from","to","label","ignore"]]
        _l["_key_"] = 0
        _m = _T.reset_index().merge(_l, how="left").set_index("index")
        _l = _m[(_m["time"] >= _m["from"]) & (_m["time"] <= _m["to"])].loc[:,["time","label","ignore"]]
        
        self.__labeled_data = self.data.copy()


        self.__labeled_data["label"] = _l["label"][~_l["ignore"]]
        self.__labeled_data.fillna(value={'label': 0}, inplace = True)
        self.__labeled_data["label"] = self.__labeled_data["label"].astype("int32")
        self.__labeled_data["ms_since_start"] = np.round(_T["time"],0).astype("int32")
 
        self.__is_labeled = True
    
    
    # PUBLIC get-Method for the private dataset that stores the labeled data
    @property
    def labeled_data(self):
        
        if not self.__is_labeled:
            raise ValueError("You have to set the labels with the set_labels-method")
        else:
            return self.__labeled_data
        
    
    # provides 3D labeled data and labels for training. The instance can call X, y, feature_names and final_time
    # X --> Array with dimensions [sample size] x [timesteps per sample] x [number of features]
    # y --> vector of labels with length [sample size]
    # feature_names --> list of the names of the assiciated columns in X
    # final_time --> vector with the number of milliseconds associated with the first dimension of X ([sample size])
    def extract_training_data(self, framelength_strategy = 'PoseNet'):
        
        if not self.__is_labeled:
            raise ValueError("You have to set the labels with the set_labels-method")
            
        self.__is_finalized = False

        n = self.__labeled_data.shape[0]
        self.avg_framelength = self.__labeled_data["ms_since_start"].values[n-1]/n
        
        if framelength_strategy == 'PoseNet':
            steps = math.ceil(2000/self.ms_per_frame) + 1
        elif framelength_strategy == 'Avg':
            steps = math.ceil(2000/self.avg_framelength) + 1
        else:
            steps = math.ceil(2000/framelength_strategy) + 1


        self.__feature_names = list(self.__labeled_data.filter(regex = '_(x|y)$', axis = 1).columns)
        
        _fn = self.__labeled_data.shape[0] - steps + 1
        _ln = self.__labeled_data.shape[0]
        self.__seq_end_time = self.__labeled_data.loc[(_ln-_fn):_ln,"ms_since_start"].values
        
        self.__X = np.zeros((
            _fn,
            steps,
            len(self.__feature_names)
        ))
        self.__y = np.zeros(self.__labeled_data.shape[0] - steps + 1)
        _F = self.__labeled_data.loc[:,self.__feature_names].values

        for i in range(steps,_F.shape[0]+1):
            self.__X[i-steps] = _F[i-steps:i,:]
            self.__y[i-steps] = self.__labeled_data['label'][i-1] 
    
       
        self.__is_finalized = True
        
      
    @property
    def label_df(self):
        
        if not self.__is_fitted:
            raise ValueError("You have to set label_df by calling the fit_range method")
    
        else:
            return self.__label_df
        
    
    @property
    def label_info(self):

        if not self.__is_fitted:
            raise ValueError("You have to set label_info by calling the fit_range method")
    
        else:
            return self.__label_info
        
     
    # PUBLIC get-Methods for the private finalized data
    @property
    def X(self):
        
        if not self.__is_finalized:
            raise ValueError("You have to set X by calling the extract_training_data method")
            
        else:
            return self.__X
        
    @property
    def y(self):
        
        if not self.__is_finalized:
            raise ValueError("You have to set y by calling the extract_training_data method")
            
        else:
            return self.__y
        
      
    @property
    def feature_names(self):
        
        if not self.__is_finalized:
            raise ValueError("You have to set feature_names by calling the extract_training_data method")
            
        else:
            return self.__feature_names
        
    
    @property
    def sequence_end_time(self):
        
        if not self.__is_finalized:
            raise ValueError("You have to set sequence_end_time by calling the extract_training_data method")
            
        else:
            return self.__seq_end_time
        





class DataEnsembler():

    pattern = '(?P<filename>(?P<filetype>[a-z]*)_(?P<movement>[a-z]*)_(?P<person>[a-z]*)_(?P<filenum>\d*)(_(?P<frame_length>\d*))?\.csv)'
    video_stats_pattern = '(?P<movement>[a-z]*)_(?P<person>[a-z]*)_(?P<filenum>\d*)'
    
    def __init__(self, ms_per_frame = 120):
        self.ms_per_frame = ms_per_frame
        
    
    def investigate_available_datafiles(self, data_dir, is_frame_based = False):
        self.data_directory = data_dir
        self.filenames = listdir(data_dir)
        self.is_frame_based = is_frame_based

        filenames_features = listdir(self.data_directory + 'features/')
        if self.is_frame_based:
            filenames_labels = listdir(self.data_directory + 'labels_framebased/')
        else:
            filenames_labels = listdir(self.data_directory + 'labels_timebased/')

        
        ds = self.__get_data_source_df(filenames_features, filenames_labels)
        vs = self.__get_video_stats()

        ds = pd.merge(
            ds, 
            vs, 
            on = ['movement','person','filenum'],
            how = 'left'
        )

        self.data_source_df = ds
        self.combined_data_files_df = self.__get_combined_datafiles_df(ds)



    def __get_data_source_df(self, filenames_features, filenames_labels):
        ds = pd.DataFrame(columns = ['filename','filetype','movement','person','filenum','frame_length'])
 
        reg = re.compile(DataEnsembler.pattern)        

        matches = []

        for file_name in filenames_features:
            match = reg.search(file_name)
            if match:
                matches.append(match)

        for file_name in filenames_labels:
            match = reg.search(file_name)
            if match:
                matches.append(match)
          
        for i, match in enumerate(matches):
            ds.loc[i] = match.groupdict()

        return ds


    def __get_video_stats(self):
        self.video_stats = pd.read_csv('docs/video_stats.csv', sep="\t")
        rev = re.compile(DataEnsembler.video_stats_pattern)

        group_dicts = []

        for file, actual_length in self.video_stats[["file","actual_length"]].itertuples(index = False):
            match = rev.search(file)
            if match:
                gdict = match.groupdict()
                gdict["actual_length"] = actual_length
                group_dicts.append(gdict)

        vs = pd.DataFrame(columns = ["movement","person","filenum","actual_length"])

        for i, gdict in enumerate(group_dicts):
            vs.loc[i] = gdict

        
        return vs


    def __get_combined_datafiles_df(self,ds):
        
        ds_features = ds[(ds.filetype == 'features') & (ds.frame_length == '000{0}'.format(str(self.ms_per_frame))[-3:])]
        ds_labels = ds[ds.filetype == 'labels']

        comb_ds = pd.merge(ds_features,
                           ds_labels,
                           on = ['movement','person','filenum'],
                           how = 'left',
                           suffixes = ['_features','_labels'])[['movement','filename_features','filename_labels']]
        comb_ds = comb_ds.drop(comb_ds[(comb_ds.movement != 'none') & (pd.isnull(comb_ds.filename_labels))].index)
        comb_ds = comb_ds.fillna({'filename_labels': 'labels_none.csv'})
        comb_ds = comb_ds.reset_index(drop = True)
        comb_ds = comb_ds[['filename_features','filename_labels']]

        return comb_ds



    def load_data(self):
        self.data = []
        self.labels = []

        if self.is_frame_based:
            label_folder = 'labels_framebased/'
        else:
            label_folder = 'labels_timebased/'

        for file_name_feat, file_name_label in self.combined_data_files_df.itertuples(index = False):
            new_data = pd.read_csv(self.data_directory + 'features/' + file_name_feat)
            
            if 'label' in list(new_data):
                new_data = new_data.drop('label', axis = 1)
            
            self.data.append(new_data)
            self.labels.append(pd.read_csv(self.data_directory + label_folder + file_name_label))
            
    
    def rescale_data_frames(self, time_of_first_frame = None, verbose = False):
        act_df = self.data_source_df[(self.data_source_df["filetype"]=="features") & (self.data_source_df["actual_length"])]
        
        act_lens = pd.merge(
            self.combined_data_files_df,
            act_df,
            how = 'inner',
            left_on = ['filename_features'],
            right_on = ['filename'],
            left_index = True
        )[['filename_features','actual_length']]

        di = DataFrameInterpolator()
        
        for idx in act_lens.index:
            if verbose:
                print("Current Index:", idx)

            self.data[idx] = di.scaleDataFrame(
                df = self.data[idx],
                actual_length_in_ms = act_lens['actual_length'].loc[idx], 
                time_of_first_frame = time_of_first_frame,
                verbose = verbose
            )


        self.actual_lengths_df = act_lens


    def interpolate_and_convert_framebased_labels(self, new_frmlen, verbose=False):

        self.ms_per_frame = new_frmlen
        if not self.is_frame_based:
            raise ValueError("This instance of DataEnsembler is not framebased. You can not convert the current labels to framebased labels.")

        self.rescaled_labels = []

        if verbose:
            print("")
            print("Rescaling Feature Timesteps & Framebased Labels:")
        for i in range(len(self.data)):
            
            if verbose:
                print(str(i)+":",self.combined_data_files_df.iloc[i]["filename_features"],
                    '\t', self.combined_data_files_df.iloc[i]["filename_labels"])
                print("\tBefore --> \tData Shape:", self.data[i].shape, "\tLabeled Samples:",
                    (self.labels[i]["real_end"] - self.labels[i]["real_start"] + 1).values.sum())
                
            self.data[i], res_labels = self.__interpolate_and_generate_new_framebased_labels(
                data_orig = self.data[i],
                labels_orig = self.labels[i],
                new_frmlen = 50
            )
            self.rescaled_labels.append(res_labels)
            
            if verbose:
                print("\tAfter ---> \tData Shape:", self.data[i].shape, "\tLabeled Samples:", 
                    (self.rescaled_labels[i]["new_idx_end"] - self.rescaled_labels[i]["new_idx_start"] + 1).values.sum())
                print("")

            self.labels[i][["real_start","real_end"]] = self.rescaled_labels[i][["new_idx_start","new_idx_end"]]
                



    def __interpolate_and_generate_new_framebased_labels(self, data_orig, labels_orig, new_frmlen):
    
        temp_labels = labels_orig.copy()
        
        start_times = []
        end_times = []

        for j in range(labels_orig.shape[0]):
            start_times.append(data_orig.iloc[int(labels_orig.iloc[j]["real_start"])]["ms_since_start"])
            end_times.append(data_orig.iloc[int(labels_orig.iloc[j]["real_end"])]["ms_since_start"])
        
        temp_labels["start_time"] = start_times
        temp_labels["end_time"] = end_times
        
        di = DataFrameInterpolator()
        
        new_data = di.get_new_df(
            df = data_orig,
            frmlen = new_frmlen
        )
        
        
        j = 0
        has_start = False
        new_start_idx = []
        new_end_idx = []

        for i in range(new_data.shape[0]):
            if j >= temp_labels.shape[0]:
                break

            t = new_data.iloc[i]["ms_since_start"]
            current_end = temp_labels.iloc[j]["end_time"]

            if has_start:
                if t > current_end - new_frmlen/2:
                    new_end_idx.append(i)
                    j = j+1
                    has_start = False
                    if j >= temp_labels.shape[0]:
                        break

            current_start = temp_labels.iloc[j]["start_time"]


            if not has_start:
                if t >= current_start - new_frmlen/2:
                    new_start_idx.append(i)
                    has_start = True
                

        temp_labels["new_idx_start"] = new_start_idx
        temp_labels["new_idx_end"] = new_end_idx
        
        temp_labels["new_start_time"] = list(new_data.loc[new_start_idx,"ms_since_start"])
        temp_labels["new_end_time"] = list(new_data.loc[new_end_idx,"ms_since_start"])
        
        return new_data, temp_labels


    
    def interpolate_data_frames(self, frmlen, verbose = False):
        self.Interpolators = []

        for i, df in enumerate(self.data):

            di = DataFrameInterpolator()
            self.data[i] = di.get_new_df(df, frmlen)
            self.Interpolators.append(di)

            if verbose:
                print(str(i)+":", df.shape, self.data[i].shape)



    def assemble_data(self, tolerance_range = None, max_error = None, framelength_strategy = 'PoseNet'):
        
        n = len(self.data)
        self.LabelGenerators = []
        self.X = None
        self.y = None
        
        for i in range(n):
            if self.is_frame_based:
                print('-------------------------------------')
                print('File ID: ' + str(i))
                lg = LabelGeneratorFramebased(
                    data_df = self.data[i],
                    labels_df = self.labels[i],
                    ms_per_frame = self.ms_per_frame)
            else:
                if tolerance_range is None or max_error is None:
                    raise ValueError("You are currently working with a time based LabelGenerator. Hence, you must provide a tolerance range and a max error."\
                        + "Alternatively you can use a frame-based LabelGenerator by calling the DataEnsembler.investigate_available_datafiles method with "\
                        + "Argument \'is_frame_based\' set to \'True\'."
                    )

                lg = LabelGenerator(
                    data = self.data[i],
                    raw_labels = self.labels[i],
                    ms_per_frame = self.ms_per_frame
                )
                lg.fit_range(
                    tolerance_range = tolerance_range,
                    max_error = max_error
                )
                lg.set_labels()
                lg.extract_training_data(framelength_strategy=framelength_strategy)
            
            self.LabelGenerators.append(lg)
            
            self.X = np.concatenate([lg.X for lg in self.LabelGenerators], axis = 0)
            self.y = np.concatenate([lg.y for lg in self.LabelGenerators], axis = 0)
            self.feature_names = self.LabelGenerators[0].feature_names

            
    def display_information(self):
        
        for i,lg in enumerate(self.LabelGenerators):
            print('i:', i, "\tshape X:", lg.X.shape, "\tshape y:", lg.y.shape, "\tcount:", 
                    len(lg.y[ lg.y > 0 ])
            )

        print("-----------------------------------------------------------------------------")
        print("shape final X:",self.X.shape)
        print("number of labeled samples:",len(self.y[self.y > 0]))




class GestureTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feature_names, byrow = True):
        self.byrow = byrow
        self.feature_names = feature_names
        self.feat_dict = {feature:feature_names.index(feature) for feature in self.feature_names}
        self.idx_x = [self.feat_dict[key] for key in self.feat_dict.keys() if key.endswith('_x')]
        self.idx_y = [self.feat_dict[key] for key in self.feat_dict.keys() if key.endswith('_y')]
        self.idx_hip_shoulder_x = [self.feat_dict[key] for key in self.feat_dict.keys()\
                      if key.endswith('_x') and ('Hip' in key or 'Shoulder' in key) ]
        self.idx_hip_y = [self.feat_dict[key] for key in self.feat_dict.keys()\
                      if key.endswith('_y') and ('Hip' in key) ]
        self.idx_shoulder_y = [self.feat_dict[key] for key in self.feat_dict.keys()\
                      if key.endswith('_y') and ('Shoulder' in key) ]
        self.idx_hip_shoulder_y = sorted(self.idx_hip_y + self.idx_shoulder_y)
        
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        
        Z = X.copy()
        
        if self.byrow:
            ax = (2)
        else:
            ax = (1,2)
         
        
        Z = self.perform_x_shift(Z,ax)
        Z = self.perform_y_shift(Z,ax)
        Z = self.perform_scaling(Z,ax)     
                  
        return Z


    def perform_x_shift(self,X, ax):
        Z = X.copy()
        self.x_shift = Z[:,:,self.idx_hip_shoulder_x].mean(axis = ax)
        Z[:,:,self.idx_x] = (Z[:,:,self.idx_x].transpose() - self.x_shift.transpose()).transpose()
        return Z

    def perform_y_shift(self,X, ax):
        Z = X.copy()
        self.y_shift = Z[:,:,self.idx_hip_shoulder_y].mean(axis = ax)
        Z[:,:,self.idx_y] = (Z[:,:,self.idx_y].transpose() - self.y_shift.transpose()).transpose()
        return Z

    def perform_scaling(self,X, ax):
        Z = X.copy()
        self.scale = (Z[:,:,self.idx_shoulder_y].mean(axis = ax) - Z[:,:,self.idx_hip_y].mean(axis = ax))
        Z = (Z.transpose() / self.scale.transpose()).transpose() 
        return Z



class DataFrameInterpolator():
    
    def __init__(self):
        self.__can_display = False
    
    def get_new_df(self, df, frmlen):
        
        self.__can_display = False
        self.__get_new_t(df, frmlen)
        self.__get_interpolation_functions(df)
        self.new_df = self.__create_new_df()    
        self.__can_display = True
        
        return self.new_df
    
    
    def __get_new_t(self, df, frmlen):
        
        self.frmlen = frmlen
        self.__orig_shape = df.shape
        self.t = df["ms_since_start"].values
        
        t_min, t_max  = int(self.t[0]), int(self.t[-1])
        diff = t_max - t_min
        
        if diff % frmlen == 0:
            num_frames = diff // frmlen 
        else:
            num_frames = diff // frmlen + 1
            
        self.t_new = np.arange(num_frames+1) * frmlen + t_min
        

        
    def __get_interpolation_functions(self, df):
        
        self.features = list(df.filter(regex = '_(x|y)$', axis = 1).columns)
        self.cubic_interpolation_functions = {}

        for feat in self.features:
            f = df[feat].values
    
            cub_f = interp1d(
                self.t, f, 
                kind = 'cubic', 
                fill_value = (f[0],f[-1]), 
                bounds_error = False,
                assume_sorted=True
            )
        
            self.cubic_interpolation_functions[feat] = cub_f
            
            
     
    def __create_new_df(self):
        new_df = pd.DataFrame(columns=self.features)
        new_df["ms_since_start"] = self.t_new

        for feat in self.features:
            new_df[feat] = self.cubic_interpolation_functions[feat](self.t_new)
            
        return new_df

    

    def scaleDataFrame(self, df, actual_length_in_ms, time_of_first_frame = None, verbose = False):
        if verbose:
            print("Calling DataFrameInterpolator.scaleDataFrame method with current length {0} and actual length {1}".format(df["ms_since_start"].values[-1],actual_length_in_ms))
        
        n = df.shape[0]
        t = df["ms_since_start"].values
        

        if time_of_first_frame is None:
            if verbose:
                print("No information provided. Time of first frame is still {0}".format(t[0]))
            
            time_of_first_frame = int(t[0])

        elif time_of_first_frame == 'avg':
            time_of_first_frame = int(t[n-1]/n)
            if verbose:
                print("Using \'avg\' to calculate new time of first frame: {0} ==> {1}".format(t[0],time_of_first_frame))
        
        elif not isinstance(time_of_first_frame, int):
            raise ValueError("The value you provided for time_of_first_frame is invalid. Please provide \'avg\', \'None\' or an integer value.")

        else:
            if verbose:
                print("Using provided value as time of first frame: {0}".format(time_of_first_frame))

        
        t = t + time_of_first_frame
        
        s = actual_length_in_ms/t[n-1]
        
        t = np.around(t * s)
        d = np.diff(t)
        d = np.insert(arr = d, obj = 0, values = 0)
        
        new_df = df.copy()
        new_df["ms_since_start"] = t
        new_df["ms_since_last_frame"] = d
        
        return new_df

    
    def display_information(self):
        if self.__can_display:
            print("FrameLength in Milliseconds:", self.frmlen,'\n')
            print("Original Time:")
            print("\tNumber of Frames:", len(self.t))
            print("\t",self.t[0:5],"...",self.t[-5:], '\n')
            print("New Time:")
            print("\tNumber of Frames:", len(self.t_new))
            print("\t",self.t_new[0:5],"...",self.t_new[-5:],'\n')
            print("Original DataFrame Shape:", self.__orig_shape)
            print("New DataFrame Shape:", self.new_df.shape)
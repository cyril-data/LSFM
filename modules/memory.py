
import random
import csv
import numpy as np
from numpy.linalg import norm
import pandas as pd
from modules.lsfm import discretize
from modules.params import PARAM_ENV_LSFM,PARAM_AGENT_LSFM
from collections import Counter
from imblearn.over_sampling import SMOTE



def zeros_columns(buffer_df, column, col_value_condition) : 
    
    state_null = list(np.zeros(len(buffer_df[column][0])))

    col_condition = col_value_condition[0]
    value_condition = col_value_condition[1]


    Ends_train = buffer_df.loc[buffer_df[col_condition]==value_condition]

    list_null = []

    for index, row in buffer_df.iterrows():
        if row[col_condition] == value_condition : 
            list_null.append(state_null)
        else : 
            list_null.append(row[column])

    buffer_df[column] = list_null

    return buffer_df

def expand_list_for_SMOTE(buffer_df, reward_parser) : 

    colX = list(buffer_df.columns)
    del colX[colX.index("reward")]
    X = buffer_df[colX]

    reward_S = buffer_df["reward"]
    y_float =  pd.Series( np.array(reward_S.values.tolist())) 
    y_class = y_float.apply(lambda x: discretize(x, reward_parser) )

    # data_transform for SMOTE

    col_obs = [ "cur_obs_"+str(k) for k in range(len(X["observation"][0]))]
    col_mask = [ "mask_"+str(k) for k in range(len(X["action_mask"][0]))]
    col_new_obs = [ "new_obs_"+str(k) for k in range(len(X["new_observation"][0]))]

    X_for_SMOTE_obs = pd.DataFrame(X["observation"].to_list(), columns = col_obs)
    X_for_SMOTE_mask = pd.DataFrame(X["action_mask"].to_list(), columns = col_mask)
    X_for_SMOTE_new_obs = pd.DataFrame(X["new_observation"].to_list(), columns = col_new_obs)

    X_for_SMOTE = pd.concat([X_for_SMOTE_obs, X_for_SMOTE_mask, X_for_SMOTE_new_obs], axis=1)
    X_for_SMOTE["action"] = X["action"]
    X_for_SMOTE["finished"] = X["finished"]
    
    

    reward_S = buffer_df["reward"]
    y_float =  pd.Series( np.array(reward_S.values.tolist())) 
    y_class = y_float.apply(lambda x: discretize(x, reward_parser) )
    
    return X_for_SMOTE, y_class


def SMOTE_tranformation(X_for_SMOTE, y, SMOTE_ratio) : 


    minor_class = [key  for key, value in dict(Counter(y)).items() if value/len(y) < SMOTE_ratio   ]
    major_class_len = max(val for k, val in dict(Counter(y)).items())
    major_class = [key for key,value in dict(Counter(y)).items() if value == major_class_len][0]

    sampling_list = [value if key not in minor_class else  int(len(y)* SMOTE_ratio ) for key, value in dict(Counter(y)).items() ]

    re_sampling = dict(zip(list(dict(Counter(y)).keys()), sampling_list))
    re_sampling


    counter = Counter(y)
    print(counter)
    oversample=SMOTE(sampling_strategy=re_sampling,random_state=10)
    X_SMOTE, y_SMOTE = oversample.fit_resample(X_for_SMOTE, y)


    counter = Counter(y_SMOTE)
    print(counter)
    
    print("SMOTE action", X_SMOTE["action"].unique())
    print("SMOTE finished", X_SMOTE["finished"].unique())

    return X_SMOTE, y_SMOTE


def mean_reward_reconstruction(class_r, reward_parser) : 
    r_parser_sort = sorted(reward_parser)

    if class_r == 0 : 
        inter = r_parser_sort[1]- r_parser_sort[0]
        r = r_parser_sort[0] - inter/2
    else : 
        if class_r == len(r_parser_sort) : 
            inter = r_parser_sort[-1]- r_parser_sort[-2]
            r = r_parser_sort[-1] + inter/2
        else : 
            r = (r_parser_sort[class_r-1] + r_parser_sort[class_r])/2.
    return r

def buffer_SMOTE(X_SMOTE,y_SMOTE,reward_parser ) : 
    list_X_SMOTE_obs =  X_SMOTE.loc[:,X_SMOTE.columns.str.contains("cur_obs_")].values.tolist()
    list_X_SMOTE_mask =  X_SMOTE.loc[:,X_SMOTE.columns.str.contains("mask")].values.tolist()
    list_X_SMOTE_new_obs =  X_SMOTE.loc[:,X_SMOTE.columns.str.contains("new_obs")].values.tolist()
    y_float_SMOTE = y_SMOTE.apply(lambda x: mean_reward_reconstruction(x, reward_parser) )
    d = {'observation': list_X_SMOTE_obs, 
         'action_mask': list_X_SMOTE_mask, 
         'action': X_SMOTE["action"], 
         "reward": y_float_SMOTE, 
         "new_observation": list_X_SMOTE_new_obs, 
         "finished": X_SMOTE["finished"], 
        }
    buffer_SMOTE_df = pd.DataFrame(data=d)
    buffer_SMOTE_df = zeros_columns(buffer_SMOTE_df, "new_observation", ["finished", True])
    # norm verification of next state norm at the end of episode


    # norm(buffer_train_df.loc[10,"new_observation"])
    Ends_SMOTE = buffer_SMOTE_df.loc[buffer_SMOTE_df["finished"]==True]["new_observation"]
    
    print("sum of new_observation norm for finished episode : ", sum(norm(Ends_SMOTE.iloc[k]) for k in range(len(Ends_SMOTE)) ))
    # Ends_SMOTE.iloc[-1]

    
    buffer_SMOTE = list(buffer_SMOTE_df.values)
    
    
    return buffer_SMOTE_df, buffer_SMOTE


class Memory:
        
    def __init__(self, max_memory, buffer_input=None):
        self._max_memory = max_memory
        
        if buffer_input==None : self._samples = []
        else : self._samples = buffer_input
            
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)
            
    def reset(self):
        self._samples = []
        
    def write(self, file) : 
        
        with open(file, 'w', newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["observation", "action_mask", "action", "reward","new_observation", "finished"],
            )
            writer.writeheader()
            
            for idrow,row in enumerate(self._samples) : 
                
                observation = f"[{','.join(str(float(x)) for x in  row[0])}]"
                action_mask = f"[{','.join(str(int(x)) for x in  row[1])}]"
                action = row[2]
                reward = row[3]
                new_observation = f"[{','.join(str(float(x)) for x in  row[4])}]"
                finished = row[5]
                
                writer.writerow(
                {
                    "observation": observation,
                    "action_mask": action_mask,
                    "action": action,
                    "reward": reward,
                    "new_observation": new_observation,
                    "finished": finished,
                })

    @property
    def num_samples(self):
        return len(self._samples)

    


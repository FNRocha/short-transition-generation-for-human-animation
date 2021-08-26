"""# BVH Preprocess"""

import pandas as pd
import numpy as np
import glob

#!pip install bvhtoolbox

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/data_CMU/IndianDanceBvh_valid
#%cd /content/drive/MyDrive/data_CMU/MartialArtsBvh_valid

def read_bvhpos(filename):

  df = pd.read_csv(filename,
                          usecols = ["time",
                                  "hip.x", "hip.y", "hip.z",
                                  "abdomen.x", "abdomen.y", "abdomen.z", 
                                  "chest.x", "chest.y", "chest.z",                                                 
                                  "head.x", "head.y", "head.z",
                                  "rCollar.x", "rCollar.y", "rCollar.z",
                                  "rShldr.x", "rShldr.y", "rShldr.z",
                                  "rForeArm.x", "rForeArm.y",	"rForeArm.z",
                                  "rHand.x",	"rHand.y",	"rHand.z",
                                  "lCollar.x", "lCollar.y", "lCollar.z",
                                  "lShldr.x", "lShldr.y", "lShldr.z",
                                  "lForeArm.x", "lForeArm.y",	"lForeArm.z",
                                  "lHand.x",	"lHand.y",	"lHand.z",                                                 
                                  "rButtock.x",	"rButtock.y",	"rButtock.z",
                                  "rThigh.x",	"rThigh.y",	"rThigh.z",
                                  "rShin.x",	"rShin.y",	"rShin.z",
                                  "rFoot.x",	"rFoot.y",	"rFoot.z",
                                  "rFoot_End.x",	"rFoot_End.y",	"rFoot_End.z",
                                  "lButtock.x",	"lButtock.y",	"lButtock.z",	
                                  "lThigh.x",	"lThigh.y",	"lThigh.z",
                                  "lShin.x",	"lShin.y",	"lShin.z",
                                  "lFoot.x",	"lFoot.y",	"lFoot.z",
                                  "lFoot_End.x",	"lFoot_End.y",	"lFoot_End.z"
                                  ])
  
  return df

def split_anim (anim, anim_len):

  split_df = anim.iloc[::anim_len, :]

  return split_df

def norm_mean_std(df):

  df_norm = (df - df.mean()) / (df.std())

  return df_norm

def preprocess(filename, anim_len):

  df_anim = read_bvhpos(filename)
  df_norm = norm_mean_std(df_anim)
  df_split = split_anim(df_norm,anim_len)
  
  return df_norm, df_split

def anim_interp(anim_len,big_frame):
  
  df_norm_val = norm_mean_std(big_frame)

  num_obs=int(df_norm_val.iloc[0:df_norm_val.shape[0]-1].shape[0]/anim_len)

  state_frame = np.reshape(np.array(df_norm_val.iloc[0:num_obs*anim_len,1:67]),(num_obs,anim_len,66))
  state_resp = np.reshape(np.array(df_norm_val.iloc[1:num_obs*anim_len+1,1:67]),(num_obs,anim_len,66))
  target_frame = np.reshape(state_resp[:,state_resp.shape[1]-1,:],(num_obs,1,66))*np.ones(state_frame.shape)    
  offset_frame = target_frame - state_frame

  return state_frame, state_resp, num_obs

"""# Concatena Tot"""

#Concatena todos arquivos _pos.csv em pasta

path =r'/content/drive/MyDrive/data_CMU/MartialArtsBvh_valid'

filenames = glob.glob(path + "/135*_pos.csv")

dfs = []

for filename in filenames:

    dfs.append(pd.read_csv(filename, usecols = ["time",
                                                "hip.x", "hip.y", "hip.z",
                                                 "abdomen.x", "abdomen.y", "abdomen.z", 
                                                 "chest.x", "chest.y", "chest.z",                                                 
                                                 "head.x", "head.y", "head.z",
                                                 "rCollar.x", "rCollar.y", "rCollar.z",
                                                 "rShldr.x", "rShldr.y", "rShldr.z",
                                                 "rForeArm.x", "rForeArm.y",	"rForeArm.z",
                                                 "rHand.x",	"rHand.y",	"rHand.z",
                                                 "lCollar.x", "lCollar.y", "lCollar.z",
                                                 "lShldr.x", "lShldr.y", "lShldr.z",
                                                 "lForeArm.x", "lForeArm.y",	"lForeArm.z",
                                                 "lHand.x",	"lHand.y",	"lHand.z",                                                 
                                                 "rButtock.x",	"rButtock.y",	"rButtock.z",
                                                 "rThigh.x",	"rThigh.y",	"rThigh.z",
                                                 "rShin.x",	"rShin.y",	"rShin.z",
                                                 "rFoot.x",	"rFoot.y",	"rFoot.z",
                                                 "rFoot_End.x",	"rFoot_End.y",	"rFoot_End.z",
                                                 "lButtock.x",	"lButtock.y",	"lButtock.z",	
                                                 "lThigh.x",	"lThigh.y",	"lThigh.z",
                                                 "lShin.x",	"lShin.y",	"lShin.z",
                                                 "lFoot.x",	"lFoot.y",	"lFoot.z",
                                                 "lFoot_End.x",	"lFoot_End.y",	"lFoot_End.z"
                                                 ]))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)

len(big_frame)


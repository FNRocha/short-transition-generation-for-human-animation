"""# Inferencia"""

#model_RTN.load_weights("/content/drive/MyDrive/data_CMU/IndianDanceBvh_valid/model_weights_indian.h5")

anim_len = 60

df_norm_val = norm_mean_std(big_frame)

num_obs=int(df_norm_val.iloc[0:df_norm_val.shape[0]-1].shape[0]/anim_len)
num_obs

state_frame = np.reshape(np.array(df_norm_val.iloc[0:num_obs*anim_len,1:67]),(num_obs,anim_len,66))
state_resp = np.reshape(np.array(df_norm_val.iloc[1:num_obs*anim_len+1,1:67]),(num_obs,anim_len,66))
target_frame = np.reshape(state_resp[:,state_resp.shape[1]-1,:],(num_obs,1,66))*np.ones(state_frame.shape)    
offset_frame = target_frame - state_frame

#y_ = model_RTN([target_frame, offset_frame, state_frame])
#tamanho de valid
#y_np = y_.numpy()

"""# Validação"""

import matplotlib.pyplot as plt
import random
from scipy import interpolate
from sklearn.metrics import mean_squared_error

valid_obs = int(0.1*state_frame.shape[0])
print("valid_obs: " + str(valid_obs))

valid_idx = np.random.seed(42)np.random.randint(0,state_frame.shape[0],size=valid_obs)

"""#Interpolação"""

from scipy import interpolate
from scipy import stats
from random import random

def interp_MSE(anim_len, state_frame, state_resp):

  #Cria dados para Interpolação Linear
  frame_interp = state_frame[:,0::anim_len-1,:]
  list_points = [0,anim_len]
  list_interp = np.arange(0,anim_len)

  #Interpolação Linear
  interp_func = interpolate.interp1d(list_points, frame_interp, axis = 1, kind="linear")
  interp_anim = interp_func(list_interp)

  valid_obs = int(0.1*state_frame.shape[0])
  print("valid_obs: " + str(valid_obs))
  
  np.random.seed(42)
  valid_idx = np.random.randint(0,state_frame.shape[0],size=valid_obs)

  dist_euclid = []  

  MSE_interp_temp = []
  
  for i in valid_idx:     
    #for j in range(0,state_frame.shape[1]):
    dist_euclid.append(np.sqrt(np.sum((np.square(interp_anim[i]-state_resp[i])), axis=1)))      
    
    MSE_interp_temp.append(mean_squared_error(interp_anim[i],state_resp[i]))    
    #equivalente explicito MSE
    #MSE_interp_temp.append(np.mean(np.mean((np.square(interp_anim[i]-state_resp[i])), axis=1), axis=0))    

  dist_euclid = np.array(dist_euclid)
  MSE_interp_temp = np.array(MSE_interp_temp) 

  return MSE_interp_temp, dist_euclid

state_frame30, state_resp30, num_obs = anim_interp(30, big_frame) 

state_frame60, state_resp60, num_obs = anim_interp(60, big_frame) 

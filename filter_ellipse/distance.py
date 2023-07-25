import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
  
d_index = []
for d in range(1,20,3):
  d_index.append(d)
  
sigmaColor_index = []
for sigmaColor in range(10,70,5):
  sigmaColor_index.append(sigmaColor)

zeros = np.zeros((len(sigmaColor_index), len(d_index)))
distance_nbubbles_avg = pd.DataFrame(data = zeros,
                                     index = sigmaColor_index, columns = d_index)
distance_nb_weighted_avg = pd.DataFrame(data = zeros,
                                        index = sigmaColor_index, columns = d_index)


k = 0
for i in range(2,81):
  k += 1
  
  path_manual = f'./modified_manual/m_X{i}.csv'
  df_manual = pd.read_csv(path_manual)
  
  #Area distributions dataframes
  lendf = len(df_manual)
  manual_1q = df_manual.iloc[0:int(lendf*0.25)]
  manual_2q = df_manual.iloc[int(lendf*0.25+1):int(lendf*0.5)]
  manual_3q = df_manual.iloc[int(lendf*0.5+1):int(lendf*0.75)]
  manual_4q = df_manual.iloc[int(lendf*0.75):lendf]
  
  distance_nbubbles = pd.DataFrame(index = sigmaColor_index)
  distance_nb_weighted = pd.DataFrame(index = sigmaColor_index)
  
  for d in range(1,20,3):
    #Distance with nbubbles
    nb_list = []
    nb_1q_list = np.array([])
    nb_2q_list = np.array([])
    nb_3q_list = np.array([])
    nb_4q_list = np.array([])
    
    for sigmaColor in range(10,70,5):
      path_automated = f'./auto/m_X{i}/m_X{i}_d_{d}_sigmaColor_{sigmaColor}.csv'
      df_auto = pd.read_csv(path_automated)
      
      #Area distribution dataframes
      lendf = len(df_auto)
      auto_1q = df_auto.iloc[0:int(lendf*0.25)]
      auto_2q = df_auto.iloc[int(lendf*0.25+1):int(lendf*0.5)]
      auto_3q = df_auto.iloc[int(lendf*0.5+1):int(lendf*0.75)]
      auto_4q = df_auto.iloc[int(lendf*0.75+1):lendf]
      
      #Distance with nbubbles
      nb_auto = len(df_auto)
      nb_manual = len(df_manual)
      nb_list.append(abs(nb_auto-nb_manual))
      
      #Distance with nbubbles for each area distribution
      nb_1q_list = np.append(nb_1q_list, abs(len(manual_1q) - len(auto_1q)))
      nb_2q_list = np.append(nb_2q_list, abs(len(manual_2q) - len(auto_2q)))
      nb_3q_list = np.append(nb_3q_list, abs(len(manual_3q) - len(auto_3q)))
      nb_4q_list = np.append(nb_4q_list, abs(len(manual_4q) - len(auto_4q)))
      
      
    distance_nbubbles[d] = nb_list
    
    #Distance with weighted average of nbubbles for each area distribution
    w1, w2, w3, w4 = 1, 1, 1, 1 #Weights
    wT = w1 + w2 + w3 + w4  #Total weight
    #print(nb_4q_list)
    distance_nb_weighted[d] = (w1*nb_1q_list + w2*nb_2q_list + w3*nb_3q_list + w4*nb_4q_list)/wT
  
  distance_nbubbles_avg =  distance_nbubbles_avg.add(distance_nbubbles)
  distance_nb_weighted_avg = distance_nb_weighted_avg.add(distance_nb_weighted)
  

distance_nbubbles_avg = distance_nbubbles_avg.divide(k)
distance_nb_weighted_avg = distance_nb_weighted_avg.divide(k)

print('Distance using number of bubbles:')
print(distance_nbubbles_avg)

print('Distance using weighted number of bubbles in different area distributions:')
print(distance_nb_weighted_avg)


array_img = distance_nb_weighted_avg.to_numpy()
plt.imshow(array_img)
plt.colorbar()
plt.show()
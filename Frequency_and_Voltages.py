#%% Author and time info: 
"""
Created on Tue Jan  3 18:26:35 2023

@author: PC
"""
#%% Libraries:
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


#%% Data:
    
freq = np.arange(150, 1151, 100)

one_kilo_ohm_resitance_voltage = [800*10**-3, 1.28, 1.76, 2.16, 2.48, 2.80, 3.04, 3.20, 3.28, 3.36, 3.36]
five_hundred_ohm_resitance_voltage = [480*10**-3, 800*10**-3, 1.04, 1.28, 1.60, 1.92, 2.24, 2.56, 2.80, 2.80, 2.88]
two_kilo_ohm_resitance_voltage = [1.44, 2.24, 2.64, 3.04, 3.20, 3.36, 3.44, 3.52, 3.52, 3.60, 3.60]

Voltages_df = pd.DataFrame(data=[one_kilo_ohm_resitance_voltage, five_hundred_ohm_resitance_voltage, two_kilo_ohm_resitance_voltage],
                           columns= freq, index= ["1 kohm resistance", "500 ohm resistance", "2 kohm resistance"])

df_columns = Voltages_df.columns
df_index = Voltages_df.index

#Voltages_df.to_csv("Voltages.csv")

regression_squared_coeffs = [0.9997, 0.9882, 0.9707]

#%% Frequency vs. Output Voltages: This is for Q2
    
#print(Voltages_df[df_columns[0]][0]) # Use this to get the output voltage values.

sns.set_context("notebook")
z_list = []

for i in range(len(Voltages_df)):
    
    z = np.polyfit(freq, Voltages_df.iloc[i], 2)    
    p = np.poly1d(z)
    
    z_list.append(z)
    
    plt.figure("%s's Output Voltage" % df_index[i])
    
    plt.title("Frequency (Hz) vs. Voltage (V) of %s \n %.8fx^2 + %.6fx + %.6f \n R_squared = %.6f" % (df_index[i], z[0], z[1], z[2], regression_squared_coeffs[i]))
    
    plt.scatter(freq, Voltages_df.iloc[i], c="purple", label="Experimental Values")
    plt.plot(freq, p(freq), "r--", label= "Polynomial Trendline")
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.ylim(0,4)
    plt.xlim(0,1200)
    plt.xticks(np.arange(150,1151,100), np.arange(150,1151,100))
    plt.legend(loc= "lower right")

    #plt.savefig("Frequency_(Hz)_vs_Voltage_(V)_of_%s.png" % df_index[i], dpi = 800, figsize = (10,10), bbox_inches="tight")

#%% RMSE Analysis:
    
resultant_sum = []

for i in range(len(Voltages_df)):
    
    original_values = Voltages_df.iloc[i]
    
    P_m_A = []
    
    for j in range(len(freq)):
        
        predicted = z_list[i][0]*freq[j]**2 + z_list[i][1]*freq[j] + z_list[i][0]
        
        res = (predicted - original_values[freq[j]])**2
        
        P_m_A.append(res)
        
    resultant_sum.append(np.sum(P_m_A))

avg_resultant_sum = []

for summmation in resultant_sum:
    
    avg_resultant_sum.append(summmation/len(freq))

sqrt_avg_resultant_sum = np.sqrt(avg_resultant_sum)















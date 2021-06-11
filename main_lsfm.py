import numpy as np
import pandas as pd


from modules.lsfm import Agent
from modules.environnement import custom_env
from modules.params import PARAM_ENV,PARAM_AGENT
from modules.experiments import experience_offline_LSFM
import csv

import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


# Environnement Mae 2D : SimpleGrid
# env_name = "SimpleGrid"
# env = custom_env(env_name, PARAM_ENV)
# state = env.reset()
# env.render()

environment = custom_env("custom", action_dim = 88,  state_dim= 324)


# read buffer
data = csv.DictReader(open("memory_Ubi.csv"))
buffer = []
for row in data:
    episode = []
    for key, value in row.items():
#         print("key,value", key, type(eval(value)))
        episode.append(eval(value) )
    buffer.append(np.array(episode))

# initialisation data
data = pd.DataFrame()
        
# launch experience of simulations
data, agent_LSFM = experience_offline_LSFM(environment,PARAM_AGENT, buffer)

# plot
df = data

fig, axs = plt.subplots(2, 3,figsize=(15, 10))
plt.figure(figsize=(5, 5))
sns.lineplot(x="cum_step", y="Avg_loss", data=df, ax = axs[0,0] )
sns.lineplot(x="cum_step", y="Avg_loss_r", data=df, ax = axs[0,1] )
sns.lineplot(x="cum_step", y="Avg_loss_N", data=df, ax = axs[0,2] )
sns.lineplot(x="cum_step", y="Avg_loss_psi", data=df, ax = axs[1,0] )
plt.show()

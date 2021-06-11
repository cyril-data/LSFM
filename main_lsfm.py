import numpy as np
import pandas as pd


from modules.lsfm import Agent
from modules.environnement import custom_env
from modules.params import PARAM_ENV,PARAM_AGENT
from modules.experiments import experience_buffer_LSFM
import csv

import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt



env_name = "SimpleGrid"
env = custom_env(env_name, PARAM_ENV)

state = env.reset()
env.render()


# read buffer
data = csv.DictReader(open("memory_simple.csv"))
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
data, agent_LSFM = experience_offline_LSFM(env,PARAM_AGENT, buffer)

# plot
df = data

fig, axs = plt.subplots(2, 3,figsize=(15, 10))
plt.figure(figsize=(5, 5))
sns.lineplot(x="cum_step", y="Avg_loss", data=df, ax = axs[0,0] )
sns.lineplot(x="cum_step", y="Avg_loss_r", data=df, ax = axs[0,1] )
sns.lineplot(x="cum_step", y="Avg_loss_N", data=df, ax = axs[0,2] )
sns.lineplot(x="cum_step", y="Avg_loss_psi", data=df, ax = axs[1,0] )
plt.show()

eval_step = int(PARAM_AGENT["num_episodes"] - PARAM_AGENT["num_episodes"] /10)

avg_loss = df.loc[eval_step:,"Avg_loss"].mean()
avg_loss_r = df.loc[eval_step:,"Avg_loss_r"].mean()
avg_loss_N = df.loc[eval_step:,"Avg_loss_N"].mean()
avg_loss_psi = df.loc[eval_step:,"Avg_loss_psi"].mean()

print("avg_loss", avg_loss)
print("avg_loss_r", avg_loss_r)
print("avg_loss_N", avg_loss_N)
print("avg_loss_psi", avg_loss_psi)


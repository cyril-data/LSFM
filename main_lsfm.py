import numpy as np
import pandas as pd


from modules.lsfm import Agent
from modules.environnement import custom_env
from modules.params import PARAM_ENV,PARAM_AGENT
from modules.experiments import experience_offline_LSFM, experience_online_LSFM, test_error_Ma_offline_LSFM
from modules.post import global_loss,losses_on_rewards_global
import csv

import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from datetime import datetime


import sys, getopt


# Online training on simple environnement :   
# python main_lsfm.py -e SimpleGrid -o True train

# Offline training on simple environnement on buffer :   
# python main_lsfm.py -e SimpleGrid -b memory_maze_word.csv -o False train
# python main_lsfm.py -e custom -b memory_Ubi.csv -o False train

# Test on buffer with model :   
# python main_lsfm.py -e SimpleGrid -b memory_maze_word.csv -m LSFM_offline_simplemaze test
# python main_lsfm.py -e custom -b memory_Ubi.csv -m LSFM_offline_Ubi test



def main(argv):

    try:
        opts, args = getopt.getopt(argv,"e:b:m:o:",["environnement=","buffer=", "model=", "online="])
    except getopt.GetoptError:
      print ('main.py -e <environnement> -b <buffer> -m <model> -o <online> ')


    print("args", args)
    print("opts", opts)



    buffer = None 
    save_model_path = None
    online = False
    train = True


    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")

    for opt, arg in opts:
        if opt == '-b':
            buffer = []
                # read buffer
            data = csv.DictReader(open(arg))
            for row in data:
                episode = []
                for key, value in row.items():
                    episode.append(eval(value) )
                buffer.append(np.array(episode))

            buff_df = pd.DataFrame(buffer, columns = data.fieldnames)


            Ends = buff_df.loc[buff_df["finished"]==True]
            nb_ep = len(Ends)
            nb_Ep_train = int(nb_ep*0.8)
            nb_Ep_test = nb_ep - int(nb_ep*0.8)

            id_train = buff_df.loc[buff_df["finished"]==True].iloc[nb_Ep_train].name

            buffer_train = list(buff_df.loc[0:id_train].values)
            buffer_test = list(buff_df.loc[id_train+1:].values)


            fig, axs = plt.subplots(1,2, figsize=(13, 5))

            sns.distplot(buff_df['reward'], ax = axs[0]).set(title='Reward distribution')
            sns.distplot(Ends['reward'], ax = axs[1]).set(title='Reward distribution on at the end of episode')
            plt.savefig(dt_string+"reward_dist.jpg")


        if opt == '-e':

            ## Environnement Mae 2D : SimpleGrid
            if arg == "SimpleGrid" : 
                env_name = "SimpleGrid"
                environment = custom_env(env_name, PARAM_ENV)
                state = environment.reset()
                environment.render()
            if arg == "custom" : 
                env_name = "custom"
                environment = custom_env(
                    "custom", 
                    action_dim = PARAM_ENV["action_space"],  
                    state_dim  = PARAM_ENV["state_dim"])


        if opt == '-m':
            save_model_path = arg+"/agent_LSFM_model"


        if opt == '-o':
            if arg == "True" : online = True

    if args[0] == 'test' : 
        train = False


    folder_WORK = "LSFM"+str(PARAM_AGENT["latent_space"])+"_"+dt_string


    if train : 
        if online : 
            data_train_df, agent_LSFM = experience_online_LSFM(environment,PARAM_AGENT)

            agent_LSFM.model_LSFM.save_weights(folder_WORK+"/agent_LSFM_model")

        else : 

            PARAM_AGENT["num_steps"]    = id_train
            print("id_train", id_train)

            data_train_df, agent_LSFM = experience_offline_LSFM(environment,PARAM_AGENT,buffer_train)

            agent_LSFM.model_LSFM.save_weights(folder_WORK+"/agent_LSFM_model")

        # plot
        df = data_train_df

        fig, axs = plt.subplots(2, 3,figsize=(15, 10))
        plt.figure(figsize=(5, 5))
        sns.lineplot(x="cum_step", y="Avg_loss", data=df, ax = axs[0,0] )
        sns.lineplot(x="cum_step", y="Avg_loss_r", data=df, ax = axs[0,1] )
        sns.lineplot(x="cum_step", y="Avg_loss_N", data=df, ax = axs[0,2] )
        sns.lineplot(x="cum_step", y="Avg_loss_psi", data=df, ax = axs[1,0] )
        plt.savefig(folder_WORK+"/"+dt_string+"Loss_LSFM.jpg")
        
    else : 
        nb_actions = environment._action_dim
        windows = [[ -4.1, -2.9], [ -2.9, -1.9], [ -1.9, -0.9], [-0.9, 0.9], [ 0.9, 1.9]]
        PARAM_AGENT["num_episodes"] = nb_Ep_test-1

        y_pred_df, y_true_df= test_error_Ma_offline_LSFM(environment,PARAM_AGENT, buffer_test, save_model_path)

        global_losses = global_loss(y_pred_df.loc[:, "reward_one-step":], y_true_df.loc[:, "reward_one-step":])

        rewards_window = []

        for window in windows : 
            print("window", window)
            rewards_window.append(losses_on_rewards_global(
                y_pred_df.loc[:, "action":], y_true_df.loc[:, "action":], window =window, nb_actions=nb_actions))


        fig, axs = plt.subplots((len(rewards_window) +1)//2,2, figsize=(15, 25))

        sns.barplot(x="error", y="MSE", data=global_losses, ax = axs[0,0]).set(
            title='Test global errors', xlabel="")
        plt.setp(axs[0,0].get_xticklabels(), rotation=30, horizontalalignment='right')

        for i, window in enumerate(windows) : 
            
            sns.boxplot(x="error", y="MSE",  data=rewards_window[i], ax = axs[(i+1)//2, (i+1)%2]).set(
                xlabel="", title='Test errors on [{}< reward < {}] by actions'.format(window[0], window[1]))
            plt.setp(axs[(i+1)//2, (i+1)%2].get_xticklabels(), rotation=30, horizontalalignment='right')


    
        plt.savefig(dt_string+"errors_test.jpg")

if __name__ == "__main__":
   main(sys.argv[1:])
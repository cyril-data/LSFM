import numpy as np
import pandas as pd


from modules.lsfm import AgentLSFM
from modules.environnement import custom_env
from modules.params import PARAM_ENV_LSFM,PARAM_AGENT_LSFM
from modules.experiments import experience_offline_LSFM, experience_online_LSFM, test_error_Ma_rewClassif_offline_LSFM
from modules.post import global_loss_reg,losses_on_rewards_global, classif, plot_2d_embelled
from modules.memory import zeros_columns, expand_list_for_SMOTE, SMOTE_tranformation, buffer_SMOTE
from modules.clustering import SF_clustering, 

import csv

import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from datetime import datetime


import sys, getopt
import os

# Online training on simple environnement :   
# python main_lsfm.py -e SimpleGrid -o True train

# Offline training on simple environnement on buffer :   
# python main_lsfm.py -e SimpleGrid -b memory_maze_word.csv -o False train
# python main_lsfm.py -e custom -b memory_Ubi_0.5.csv -o False train

# Test on buffer with model :   
# python main_lsfm.py -e SimpleGrid -b memory_maze_word.csv -m LSFM_Ma_classif_offline test
# python main_lsfm.py -e custom -b memory_Ubi_0.5.csv -m LSFM250_2021-07-15_09h-12m-53s test



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
    folder_WORK =  "LSFM"+"/" +"LSFM"+str(PARAM_AGENT_LSFM["latent_space"])+"_"+dt_string

    os.mkdir(folder_WORK)


    for opt, arg in opts:
        if opt == '-b':
            # buffer = []
            #     # read buffer
            # data = csv.DictReader(open(arg))
            # for row in data:
            #     episode = []
            #     for key, value in row.items():
            #         episode.append(eval(value) )
            #     buffer.append(np.array(episode))

            # buff_df = pd.DataFrame(buffer, columns = data.fieldnames)

            buff_df = pd.read_csv(open(arg))
            print("read 1",  datetime.now() - now)
            buff_df["observation"] = buff_df["observation"].apply(eval)
            print("eval 1",  datetime.now() - now)
            buff_df["action_mask"] = buff_df["action_mask"].apply(eval)
            print("eval 2",  datetime.now() - now)
            buff_df["new_observation"] = buff_df["new_observation"].apply(eval)
            print("eval 3",  datetime.now() - now)


            # zeros all new_observation when episodes are finished
            buff_df = zeros_columns(buff_df, "new_observation", ["finished", True])

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
            plt.savefig(folder_WORK+"/"+dt_string+"reward_dist.jpg")
            plt.close()


            # SMOTE data tranformation
            buffer_train_df = pd.DataFrame(buffer_train, columns = buff_df.columns)
            if PARAM_AGENT_LSFM["SMOTE_ratio"]>0 : 
                # data preprocessing for SMOTE
                X_for_SMOTE, y_class = expand_list_for_SMOTE(buffer_train_df, PARAM_AGENT_LSFM["reward_parser"])
                # plot 2d embelled original data 
                plot_2d_embelled(X_for_SMOTE, y_class, folder_WORK+"/"+dt_string+"state_2d_embelled.jpg")
                # SMOTE transformation
                X_SMOTE, y_SMOTE = SMOTE_tranformation(X_for_SMOTE, y_class, PARAM_AGENT_LSFM["SMOTE_ratio"]) 
                # plot 2d embelled SMOTE data 
                plot_2d_embelled(X_SMOTE, y_SMOTE, folder_WORK+"/"+dt_string+"state_2d_embelled_SMOTE.jpg")
                # buffer reconstitution
                buffer_train_df , buffer_train= buffer_SMOTE(X_SMOTE,y_SMOTE,  PARAM_AGENT_LSFM["reward_parser"])
                 
        if opt == '-e':

            ## Environnement Mae 2D : SimpleGrid
            if arg == "SimpleGrid" : 
                env_name = "SimpleGrid"
                environment = custom_env(env_name, PARAM_ENV_LSFM)
                state = environment.reset()
                environment.render()
            if arg == "custom" : 
                env_name = "custom"
                environment = custom_env(
                    "custom", 
                    action_dim = PARAM_ENV_LSFM["action_space"],  
                    state_dim  = PARAM_ENV_LSFM["state_dim"])


        if opt == '-m':
            save_model_path = arg+"/agent_LSFM_model"


        if opt == '-o':
            if arg == "True" : online = True

    if args[0] == 'test' : 
        train = False




    if train : 
        if online : 
            data_train_df, agent_LSFM = experience_online_LSFM(environment,PARAM_AGENT_LSFM)

            agent_LSFM.model_LSFM.save_weights(folder_WORK+"/agent_LSFM_model")

        else : 

            PARAM_AGENT_LSFM["num_steps"]    = id_train
            print("id_train", id_train)

            data_train_df, agent_LSFM = experience_offline_LSFM(environment,PARAM_AGENT_LSFM,buffer_train)

            agent_LSFM.model_LSFM.save_weights(folder_WORK+"/agent_LSFM_model")

        #*******************************************
        #**************  clustering  **************
        SF_clusters_df = SF_clustering(
            buffer_train_df,agent_LSFM.model_LSFM, 
            nb_r_class = 10 , 
            nb_action = 4 , 
            r_discretizer = None)

        # --------- plot spider

        filtre_action = []
        filtre_comp_int = [] 

        spider_cls_mean(
            SF_clusters_df, 
            filtre_action, 
            filtre_comp_int, 
            savefig = folder_WORK+"/"+dt_string+"spider.jpg" )

        for a in range(nb_actions) : 
            spider_cls_mean(
                SF_clusters_df, 
                [a], 
                filtre_comp_int, 
                savefig = folder_WORK+"/"+dt_string+"spider_a" +a+".jpg" )

        split_analysis = [
            list(range(0,12)),
            list(range(12,24)),
            list(range(24,36))
        ]
        for filtre_comp_int in split_analysis : 
            spider_cls_mean(
                SF_clusters_df, 
                filtre_action, 
                filtre_comp_int, 
                savefig = folder_WORK+"/"+dt_string+"spider_comp" + str(filtre_comp_int) +".jpg" )
        #*******************************************



        
        # plot
        df = data_train_df
        print("data_train_df", data_train_df)
        fig, axs = plt.subplots(2, 3,figsize=(15, 10))
        sns.lineplot(x="cum_step", y="Avg_loss", data=df, ax = axs[0,0] )
        sns.lineplot(x="cum_step", y="Avg_loss_r", data=df, ax = axs[0,1] )
        sns.lineplot(x="cum_step", y="Avg_loss_N", data=df, ax = axs[0,2] )
        sns.lineplot(x="cum_step", y="Avg_loss_psi", data=df, ax = axs[1,0] )
        plt.savefig(folder_WORK+"/"+dt_string+"Loss_LSFM.jpg")
        plt.close()

        
    else : 
        
        nb_actions = environment._action_dim

        #*******************************************
        #**************  clustering  **************
        agent_LSFM = Agent(env, param_agent, save_model_path)
        model_LSFM = agent_LSFM.model_LSFM
        df = buffer_train_df
        nb_r_class = 10
        init_r_discretizer(df["reward"].values.reshape(-1,1), nb_r_class)
        df["reward_discrete"] = tranform_r_discretizer(df["reward" ], r_discretizer)
        SF_clusters_df = SF_clustering(
            buffer_train_df,model_LSFM, nb_action = 4 , r_discretizer = r_discretizer)

        # --------- plot spider

        filtre_action = []
        filtre_comp_int = [] 

        spider_cls_mean(
            SF_clusters_df, 
            filtre_action, 
            filtre_comp_int, 
            savefig = folder_WORK+"/"+dt_string+"spider.jpg" )

        for a in range(nb_actions) : 
            spider_cls_mean(
                SF_clusters_df, 
                [a], 
                filtre_comp_int, 
                savefig = folder_WORK+"/"+dt_string+"spider_a" +a+".jpg" )

        split_analysis = [
            list(range(0,12)),
            list(range(12,24)),
            list(range(24,36))
        ]
        for filtre_comp_int in split_analysis : 
            spider_cls_mean(
                SF_clusters_df, 
                filtre_action, 
                filtre_comp_int, 
                savefig = folder_WORK+"/"+dt_string+"spider_comp" + str(filtre_comp_int) +".jpg" )
        #*******************************************


        PARAM_AGENT_LSFM["num_episodes"] = nb_Ep_test-1

        y_pred_df, y_true_df= test_error_Ma_rewClassif_offline_LSFM(
            environment,PARAM_AGENT_LSFM, buffer_test, save_model_path,  PARAM_AGENT_LSFM["reward_parser" ])

        parser =  PARAM_AGENT_LSFM["reward_parser" ]
        reward_classes = [[-np.inf,parser[0]]] + [[parser[k],parser[k+1]]  for k in range(len(parser)-1)] + [[parser[-1], np.inf]]

        y_pred = y_pred_df["reward_one-step"].apply(np.argmax)
        y_true = y_true_df["reward_one-step"]

        error_classif_df = classif(y_true, y_pred, folder_WORK+"/"+dt_string, label = reward_classes)
        print("error_classif_df", error_classif_df)



        fig , ax = plt.subplots()
        sns.barplot(x="reward_classes", y="error", hue = "type", data=error_classif_df).set(
            title='Test classif reward errors', xlabel="")
        plt.savefig(folder_WORK+"/"+dt_string+"classif_reward_error.jpg")
        plt.close()

        cols = ["Norm_phi"   , "target_psi"    , "phi_sp1"  ] 
        metric = "MSE"

        global_losses = global_loss_reg(y_pred_df.loc[:, cols], y_true_df.loc[:,cols])


        fig , ax = plt.subplots()
        sns.barplot(x="error", y="MSE", data=global_losses).set(
            title='Test global errors', xlabel="")
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.savefig(folder_WORK+"/"+dt_string+"global_losses_MSE.jpg")
        plt.close()

        rewards_window = []
        for i in range(len(reward_classes)) : 
            rewards_window.append(losses_on_rewards_global(cols,
                y_pred_df.loc[:, "action":], y_true_df.loc[:, "action":], 
                reward_class =i, nb_actions=nb_actions))
    
        for i in range(len(reward_classes)) : 
            if rewards_window[i].loc[:,"MSE":"MAE"].applymap(lambda x: x is None).all().all() : 
                print("No value for class", i)
            else :       
                fig , ax = plt.subplots()
                dataplot_i = rewards_window[i].fillna(value=np.nan)
                sns.boxplot(x="error", y=metric,  data=dataplot_i).set(
                    xlabel="", title='Test errors on reward class {} by actions'.format(reward_classes[i]))
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                plt.savefig(folder_WORK+"/"+dt_string+"MSE_reward_class_"+str(i)+".jpg")
                plt.close()


if __name__ == "__main__":
   main(sys.argv[1:])
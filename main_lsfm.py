import getopt
import os
import sys
from datetime import datetime
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from modules.clustering import (
    SF_clustering,
    init_r_discretizer,
    plot_spiders,
    tranform_r_discretizer,
)
from modules.environnement import custom_env
from modules.experiments import (
    experience_offline_LSFM,
    experience_oneline_eigenoption,
    test_error_Ma_rewClassif_offline_LSFM,
)
from modules.lsfm import AgentLSFM
from modules.memory import (
    SMOTE_tranformation,
    buffer_SMOTE,
    expand_list_for_SMOTE,
    zeros_columns,
)

# from modules.params import PARAM_AGENT_LSFM, PARAM_ENV_LSFM
from modules.post import (
    classif,
    global_loss_reg,
    losses_on_rewards_global,
    plot_classif_reward_error,
    plot_data,
    plot_error,
)


from pathlib import Path
import glob

sys.path.append("/".join(os.getcwd().split("/")[:-1]))

import argparse

sns.set()


# Online training on simple environnement :
# python main_lsfm.py --o True --t train

# Offline training on simple environnement on buffer :
# python main_lsfm.py --e SimpleGrid --b memory_maze_word.csv --o False --t train
# python main_lsfm.py -e custom -b memory_Ubi_0.5.csv -o False train

# Test on buffer with model :
# python main_lsfm.py -e SimpleGrid -b memory_maze_word.csv -m LSFM_Ma_classif_offline test
# python main_lsfm.py -e custom -b memory_Ubi_0.5.csv -m LSFM250_2021-07-15_09h-12m-53s test


def check_file(file):
    # Search/download file (if necessary) and return path
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == "":  # exists
        return file
    else:
        return None

    # else:  # search
    #     files = glob.glob("./**/" + file, recursive=True)  # find file
    #     assert len(files), f"File not found: {file}"  # assert file was found
    #     assert (
    #         len(files) == 1
    #     ), f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
    #     return files[0]  # return file


def main(opt):

    print("opt", opt)

    # try:
    #     opts, args = getopt.getopt(
    #         argv,
    #         "e:b:m:o:l:n:s:",
    #         [
    #             "environnement=",
    #             "buffer=",
    #             "model=",
    #             "online=",
    #             "latentdim=",
    #             "namefolder=",
    #             "smoteratio=",
    #         ],
    #     )
    # except getopt.GetoptError:
    #     print(
    #         "main.py -e <environnement> -b <buffer> -m <model> -o <online> -l <latentdim> -n <namefolder> -r <smoteratio>"
    #     )

    buffer = None
    save_model_path = None
    online = False
    train = True

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    fold_dir = "RESULTS"

    if opt.env:
        PARAM_ENV_LSFM = opt.env["PARAM_ENV_LSFM"]
        ENV_NAME = opt.env["ENV_NAME"]
    # from modules.params import PARAM_AGENT_LSFM, PARAM_ENV_LSFM

    if opt.agent:
        PARAM_AGENT_LSFM = opt.agent["PARAM_AGENT_LSFM"]

    fold_result = "/" + str(PARAM_AGENT_LSFM["latent_space"]) + "latDim_" + dt_string

    if opt.n:
        fold_result = "/" + opt.n

    if opt.s:
        print("SMOTE_ratio", type(opt.s), float(opt.s))
        PARAM_AGENT_LSFM["SMOTE_ratio"] = opt.s

    if opt.l:
        print("latent_space", type(opt.l), int(opt.l))
        PARAM_AGENT_LSFM["latent_space"] = opt.l

    if os.path.exists("LSFM"):
        fold_dir = "LSFM/" + fold_dir

    folder_WORK = fold_dir + fold_result
    os.mkdir(folder_WORK)

    if opt.t == "test":
        train = False

    if opt.b:
        buffer = []
        # read buffer
        print("read 0")

        buff_df = pd.read_csv(open(opt.b))
        print("read 1", datetime.now() - now)
        buff_df["observation"] = buff_df["observation"].apply(eval)
        print("eval 1", datetime.now() - now)
        buff_df["action_mask"] = buff_df["action_mask"].apply(eval)
        print("eval 2", datetime.now() - now)
        buff_df["new_observation"] = buff_df["new_observation"].apply(eval)
        print("eval 3", datetime.now() - now)

        # buff_df = pd.DataFrame(buffer, columns = data.fieldnames)
        # zeros all new_observation when episodes are finished
        buff_df = zeros_columns(buff_df, "new_observation", ["finished", True])

        Ends = buff_df.loc[buff_df["finished"] == True]
        nb_ep = len(Ends)
        nb_Ep_train = int(nb_ep * 0.8)
        nb_Ep_test = nb_ep - int(nb_ep * 0.8)

        id_train = buff_df.loc[buff_df["finished"] == True].iloc[nb_Ep_train].name

        buffer_train = list(buff_df.loc[0:id_train].values)
        buffer_test = list(buff_df.loc[id_train + 1 :].values)

        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        sns.distplot(buff_df["reward"], ax=axs[0]).set(title="Reward distribution")
        sns.distplot(Ends["reward"], ax=axs[1]).set(
            title="Reward distribution on at the end of episode"
        )
        plt.savefig(folder_WORK + "/" + dt_string + "reward_dist.jpg")
        plt.close()

        buffer_train_df = pd.DataFrame(buffer_train, columns=buff_df.columns)
        buffer_test_df = pd.DataFrame(buffer_train, columns=buff_df.columns)

        # SMOTE data tranformation
        if PARAM_AGENT_LSFM["SMOTE_ratio"] > 0 and train:
            print("SMOTE 0", datetime.now() - now)
            # data preprocessing for SMOTE
            print("SMOTE 1", datetime.now() - now)
            X_for_SMOTE, y_class = expand_list_for_SMOTE(
                buffer_train_df, PARAM_AGENT_LSFM["reward_parser"]
            )
            # plot 2d embelled original data
            print("SMOTE 2", datetime.now() - now)
            # plot_2d_embelled(X_for_SMOTE, y_class, folder_WORK+"/"+dt_string+"state_2d_embelled.jpg")
            # SMOTE transformation
            print("SMOTE 3", datetime.now() - now)
            X_SMOTE, y_SMOTE = SMOTE_tranformation(
                X_for_SMOTE, y_class, PARAM_AGENT_LSFM["SMOTE_ratio"]
            )
            # plot 2d embelled SMOTE data
            print("SMOTE 4", datetime.now() - now)
            # plot_2d_embelled(X_SMOTE, y_SMOTE, folder_WORK+"/"+dt_string+"state_2d_embelled_SMOTE.jpg")
            # buffer reconstitution
            print("SMOTE 5", datetime.now() - now)
            buffer_train_df, buffer_train = buffer_SMOTE(
                X_SMOTE, y_SMOTE, PARAM_AGENT_LSFM["reward_parser"]
            )
            id_train = len(buffer_train_df)
            print("SMOTE 6", datetime.now() - now)

    # Environnement Mae 2D : SimpleGrid
    if ENV_NAME == "SimpleGrid":
        env_name = "SimpleGrid"
        environment = custom_env(env_name, PARAM_ENV_LSFM)
        state = environment.reset()
        environment.render()
    if ENV_NAME == "custom":
        env_name = "custom"
        environment = custom_env(
            "custom",
            action_dim=PARAM_ENV_LSFM["action_space"],
            state_dim=PARAM_ENV_LSFM["state_dim"],
        )

    if opt.m:
        save_model_path = fold_dir + "/" + opt.m + "/agent_LSFM_model"

    if opt.o:
        if opt.o == "True":
            online = True

    print("train 1", now - datetime.now())

    if train:
        if online:

            data_eigen = pd.DataFrame()
            for latent_dim in [300]:
                PARAM_AGENT_LSFM["latent_space"] = latent_dim
                for eigen in [5]:
                    PARAM_AGENT_LSFM["eigenoption_number"] = eigen

                    dt_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")

                    data, agent_LSFM, memory = experience_oneline_eigenoption(
                        environment,
                        PARAM_AGENT_LSFM,
                        eigenoption=True,
                        file_save=folder_WORK
                        + "/"
                        + dt_string
                        + "lat"
                        + str(latent_dim),
                    )
                    data["eigen_opt"] = eigen
                    data["latent_space"] = latent_dim

                    data_eigen = pd.concat([data_eigen, data])

                    data_plot = data_eigen
                    data_plot["eigen"] = True
                    data_plot.loc[data_plot["eigen_opt"] == 0, "eigen"] = False
                    plot_data(
                        data_eigen,
                        datafile=folder_WORK
                        + "/"
                        + dt_string
                        + "online_eigendiscovery.jpg",
                    )

                    data_plot = data_eigen
                    data_plot["eigen"] = True
                    data_plot.loc[data_plot["eigen_opt"] == 0, "eigen"] = False
                    data_plot.to_csv(folder_WORK + "/" + dt_string + "_data.csv")
                    dt_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
                    plot_data(
                        data_plot,
                        datafile=folder_WORK
                        + "/"
                        + dt_string
                        + "online_eigendiscovery.jpg",
                    )

            data_plot = data_eigen
            data_plot["eigen"] = True
            data_plot.loc[data_plot["eigen_opt"] == 0, "eigen"] = False

            # for eps_min in [0.2, 0.5, 0.7, 1.]:
            #     PARAM_AGENT_LSFM["policy"]["eps-greedy"]["exponantial"]["eps_min"] = eps_min
            #     for eigen in [8, 16]:
            #         PARAM_AGENT_LSFM["eigenoption_number"] = eigen
            #         for _lambda in [0.0005, 0.001, 0.002]:
            #             PARAM_AGENT_LSFM["policy"]["eps-greedy"]["exponantial"]["lambda"] = _lambda

            #             data, agent_LSFM, memory = experience_oneline_eigenoption(
            #                 environment, PARAM_AGENT_LSFM,  eigenoption=True, file_save=folder_WORK)

            #             data["lambda_exp"] = _lambda
            #             data["eigen_opt"] = eigen
            #             data["eps_min"] = eps_min

            #             data_eigen = pd.concat([data_eigen, data])

            # data_plot = data_eigen
            # data_plot["eigen"] = True
            # data_plot.loc[data_plot["eigen_opt"] == 0, "eigen"] = False
            # data_plot.loc[data_plot["lambda_exp"] == 0, "eigen"] = False
            # data_plot.loc[data_plot["eps_min"] == 1, "eigen"] = False

            print("data", data_plot)
            dt_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
            plot_data(
                data_plot,
                datafile=folder_WORK + "/" + dt_string + "online_eigendiscovery.jpg",
            )

            # fig, axs = plt.subplots(2, 3, figsize=(25, 15))

            # # for option in options :
            # hue = "eigen_opt"
            # x = "cum_step"
            # style = "eigen"

            # sns.lineplot(x=x, y="Avg_loss",  hue=hue,
            #              style=style, data=data_plot, ax=axs[0, 0])
            # sns.lineplot(x=x, y="Avg_loss_r",  hue=hue,
            #              style=style, data=data_plot, ax=axs[0, 1])
            # sns.lineplot(x=x, y="Avg_loss_psi",  hue=hue,
            #              style=style, data=data_plot, ax=axs[0, 2])
            # sns.lineplot(x=x, y="Avg_loss_phip1",  hue=hue,
            #              style=style, data=data_plot, ax=axs[1, 0])
            # sns.lineplot(x=x, y="exploration_ratio",  hue=hue,
            #              style=style, data=data_plot, ax=axs[1, 1])
            # sns.lineplot(x=x, y="eigen_exploration",  hue=hue,
            #              style=style, data=data_plot, ax=axs[1, 2])

            # plt.savefig(folder_WORK+"/"+dt_string+"online_eigendiscovery.jpg")
            # plt.close()

            buffer_train_df = pd.DataFrame(
                memory._samples,
                columns=[
                    "observation",
                    "action_mask",
                    "action",
                    "reward",
                    "new_observation",
                    "finished",
                ],
            )

            agent_LSFM.model_LSFM.save_weights(folder_WORK + "/agent_LSFM_model")

            data_train_df = data_plot
            data_train_df.to_csv(folder_WORK + "/data.csv")
        else:

            PARAM_AGENT_LSFM["num_steps"] = id_train
            print("id_train", id_train)

            data_train_df, agent_LSFM = experience_offline_LSFM(
                environment, PARAM_AGENT_LSFM, buffer_train
            )

            agent_LSFM.model_LSFM.save_weights(folder_WORK + "/agent_LSFM_model")

        print("train 2", now - datetime.now())

        # *******************************************
        # **************  clustering  **************
        nb_action = environment._action_dim

        SF_clusters_df = SF_clustering(
            buffer_train_df,
            agent_LSFM.model_LSFM,
            nb_r_class=PARAM_AGENT_LSFM["max_r_cluster"],
            nb_action=nb_action,
            r_discretizer=None,
            max_SF_cluster=PARAM_AGENT_LSFM["max_SF_cluster"],
        )
        # --------- plot spider

        # plot_spiders(SF_clusters_df, nb_action=nb_action,
        #              fold_result=folder_WORK+"/"+dt_string)

        # filtre_action = []
        # filtre_comp_int = []

        # spider_cls(
        #     SF_clusters_df,
        #     filtre_action,
        #     filtre_comp_int,
        #     savefig = folder_WORK+"/"+dt_string+"spider.jpg" )

        # for a in range(nb_action) :
        #     spider_cls(
        #         SF_clusters_df,
        #         [a],
        #         filtre_comp_int,
        #         savefig = folder_WORK+"/"+dt_string+"spider_a" +a+".jpg" )

        # split_analysis = [
        #     list(range(0,12)),
        #     list(range(12,24)),
        #     list(range(24,36))
        # ]
        # for filtre_comp_int in split_analysis :
        #     spider_cls(
        #         SF_clusters_df,
        #         filtre_action,
        #         filtre_comp_int,
        #         savefig = folder_WORK+"/"+dt_string+"spider_comp" + str(filtre_comp_int) +".jpg" )
        # *******************************************

        # plot

        # plot_error(data_train_df, folder_WORK+"/"+dt_string)
        #  :

        # df = data_train_df
        # print("data_train_df", data_train_df)
        # fig, axs = plt.subplots(2, 3,figsize=(15, 10))
        # sns.lineplot(x="cum_step", y="Avg_loss", data=df, ax = axs[0,0] )
        # sns.lineplot(x="cum_step", y="Avg_loss_r", data=df, ax = axs[0,1] )
        # sns.lineplot(x="cum_step", y="Avg_loss_N", data=df, ax = axs[0,2] )
        # sns.lineplot(x="cum_step", y="Avg_loss_psi", data=df, ax = axs[1,0] )
        # plt.savefig(folder_WORK+"/"+dt_string+"Loss_LSFM.jpg")
        # plt.close()

    else:

        # *******************************************
        # **************  clustering  **************
        nb_action = environment._action_dim

        agent_LSFM = AgentLSFM(environment, PARAM_AGENT_LSFM, save_model_path)
        model_LSFM = agent_LSFM.model_LSFM

        agent_LSFM_export = AgentLSFM(environment, PARAM_AGENT_LSFM, export=True)
        agent_LSFM_export.model_LSFM.load_weights(save_model_path)
        agent_LSFM_export.model_LSFM.save(folder_WORK + "/agent_LSFM_model_full")

        print("test load ")
        m = tf.keras.models.load_model(folder_WORK + "/agent_LSFM_model_full")

        df = buffer_train_df
        r_discretizer = init_r_discretizer(
            df["reward"].values.reshape(-1, 1), PARAM_AGENT_LSFM["max_r_cluster"]
        )
        df["reward_discrete"] = tranform_r_discretizer(df["reward"], r_discretizer)

        SF_clusters_df = SF_clustering(
            buffer_test_df,
            model_LSFM,
            nb_action=nb_action,
            r_discretizer=r_discretizer,
            max_SF_cluster=PARAM_AGENT_LSFM["max_SF_cluster"],
        )

        # --------- plot spider

        plot_spiders(
            SF_clusters_df,
            nb_action=nb_action,
            fold_result=folder_WORK + "/" + dt_string,
        )

        # filtre_action = []
        # filtre_comp_int = []

        # spider_cls(
        #     SF_clusters_df,
        #     filtre_action,
        #     filtre_comp_int,
        #     savefig = folder_WORK+"/"+dt_string+"spider.jpg" )

        # for a in range(nb_action) :
        #     spider_cls(
        #         SF_clusters_df,
        #         [a],
        #         filtre_comp_int,
        #         savefig = folder_WORK+"/"+dt_string+"spider_a" +a+".jpg" )

        # split_analysis = [
        #     list(range(0,12)),
        #     list(range(12,24)),
        #     list(range(24,36))
        # ]
        # for filtre_comp_int in split_analysis :
        #     spider_cls(
        #         SF_clusters_df,
        #         filtre_action,
        #         filtre_comp_int,
        #         savefig = folder_WORK+"/"+dt_string+"spider_comp" + str(filtre_comp_int) +".jpg" )
        # *******************************************

        PARAM_AGENT_LSFM["num_episodes"] = nb_Ep_test - 1

        y_pred_df, y_true_df = test_error_Ma_rewClassif_offline_LSFM(
            environment,
            PARAM_AGENT_LSFM,
            buffer_test,
            save_model_path,
            PARAM_AGENT_LSFM["reward_parser"],
        )

        parser = PARAM_AGENT_LSFM["reward_parser"]
        reward_classes = (
            [[-np.inf, parser[0]]]
            + [[parser[k], parser[k + 1]] for k in range(len(parser) - 1)]
            + [[parser[-1], np.inf]]
        )

        y_pred = y_pred_df["reward_one-step"].apply(np.argmax)
        y_true = y_true_df["reward_one-step"]

        error_classif_df = classif(
            y_true, y_pred, folder_WORK + "/" + dt_string, label=reward_classes
        )
        print("error_classif_df", error_classif_df)

        plot_classif_reward_error(error_classif_df, folder_WORK + "/" + dt_string)
        # fig , ax = plt.subplots()
        # sns.barplot(x="reward_classes", y="error", hue = "type", data=error_classif_df).set(
        #     title='Test classif reward errors', xlabel="")
        # plt.savefig(folder_WORK+"/"+dt_string+"classif_reward_error.jpg")
        # plt.close()

        cols = ["Norm_phi", "target_psi", "phi_sp1"]
        metric = "MSE"
        global_losses = global_loss_reg(y_pred_df.loc[:, cols], y_true_df.loc[:, cols])

        fig, ax = plt.subplots()
        sns.barplot(x="error", y="MSE", data=global_losses).set(
            title="Test global errors", xlabel=""
        )
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
        plt.savefig(folder_WORK + "/" + dt_string + "global_losses_MSE.jpg")
        plt.close()

        rewards_window = []
        for i in range(len(reward_classes)):
            rewards_window.append(
                losses_on_rewards_global(
                    cols,
                    y_pred_df.loc[:, "action":],
                    y_true_df.loc[:, "action":],
                    reward_class=i,
                    nb_action=nb_action,
                )
            )

        for i in range(len(reward_classes)):
            if (
                rewards_window[i]
                .loc[:, "MSE":"MAE"]
                .applymap(lambda x: x is None)
                .all()
                .all()
            ):
                print("No value for class", i)
            else:
                fig, ax = plt.subplots()
                dataplot_i = rewards_window[i].fillna(value=np.nan)
                sns.boxplot(x="error", y=metric, data=dataplot_i).set(
                    xlabel="",
                    title="Test errors on reward class {} by actions".format(
                        reward_classes[i]
                    ),
                )
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
                plt.savefig(
                    folder_WORK
                    + "/"
                    + dt_string
                    + "MSE_reward_class_"
                    + str(i)
                    + ".jpg"
                )
                plt.close()
    print("Wall time CPU : ", datetime.now() - now)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #         argv,
    #         "e:b:m:o:l:n:s:",
    #         [
    #             "environnement=",
    #             "buffer=",
    #             "model=",
    #             "online=",
    #             "latentdim=",
    #             "namefolder=",
    #             "smoteratio=",
    #         ],
    #     )
    # except getopt.GetoptError:
    #     print(
    #         "main.py -e <environnement> -b <buffer> -m <model> -o <online> -l <latentdim> -n <namefolder> -r <smoteratio>"
    #     )

    # python main_lsfm.py -e SimpleGrid -o True train

    # parser train

    parser.add_argument(
        "--env", type=str, default="param_env.yaml", help="params to define environment"
    )

    parser.add_argument(
        "--agent", type=str, default="param_agent.yaml", help="params to define agent"
    )

    parser.add_argument("--t", type=str, default="train", help="train or test")
    parser.add_argument("--b", type=str, help="Offline buffer for training ")
    parser.add_argument("--m", type=str, help="Model for training ")
    parser.add_argument("--o", type=str, help="True : online, False : offline ")
    parser.add_argument("--l", type=int, default=300, help="latentdim ")
    parser.add_argument("--n", type=str, help="output name folder ")
    parser.add_argument("--s", type=float, help="smote ratio ")

    opt = parser.parse_args()

    if check_file(opt.env):
        with open(opt.env) as f:
            optyaml = yaml.safe_load(f)  # load hyps
            opt.env = optyaml
    else:
        opt.env = {
            "ENV_NAME": "SimpleGrid",
            "PARAM_ENV_LSFM": {
                "agent_pos": [0, 0],
                "goal_pos": [[5, 0]],
                "reward_pos": [[5, 0]],
                "grid_size": 6,
                "reward_minmax": [0.0, 0.0],
                "reward_user_defined": False,
                "pattern": "empty",
                "obs_mode": "index",
            },
        }

    if check_file(opt.agent):
        with open(opt.agent) as f:
            optyaml = yaml.safe_load(f)  # load hyps
            opt.agent = optyaml
    else:
        opt.agent = {
            "PARAM_ENV_LSFM": {
                "gamma": 0.90,
                # "optimizer_LSFM": keras.optimizers.Adam(),
                # "optimizer_Q": keras.optimizers.Adam(),
                "alpha_r": 1.0,
                "alpha_N": 0.1,
                "alpha_psi": 0.01,
                "alpha_phi": 0.01,
                "policy": {"type": ["fix_random_option"]},
                # "policy": {
                #     "type": ["eps-greedy", "exponantial"],
                #     "eps-greedy": {
                #         "exponantial": {
                #             "eps_max": 1.,
                #             "eps_min": 0.01,
                #             "lambda": 0.0002
                #         },
                #         "constant": {
                #             "eps": 0.01
                #         }
                #     }
                # },
                "memory": 500000,
                "latent_space": 100,
                "hidden_dim_ratio": 1.0,
                "num_episodes": 150,
                "steps_max_episode": 200,
                "num_steps": 1,
                "batch_size": 32,
                "RANDOM_REWARD_STD": -1.0,
                "double_model": False,
                "tau": 0.08,
                "filter_done": True,
                "train_LSFM": True,
                "train_on_Q_latent": False,
                "model_Q_Lin": False,
                "train": True,
                "run": 4,
                "render": False,
                "reward_parser": [-1.5, -0.5, 0.5, 1.5],
                "SMOTE_ratio": 0.0,
                "max_SF_cluster": 20,
                "max_r_cluster": 10,
                "eigenoption_number": 16,
                "eigen_exploration": 0.5,
                "start_eigenoption": 10,
                "discoverNegation": True,
            }
        }

    main(opt)

import pandas as pd
import numpy as np
from modules.memory import Memory
import tensorflow as tf
from tensorflow import keras

from modules.dqn import Agent_Q
from modules.lsfm import AgentLSFM, discretize


import time


def clean_duplicate(memory):

    df = pd.DataFrame(memory._samples)
    df[5] = df[5].astype(np.float32)

    col_st = ["st_"+str(k) for k in range(len(df[0][0]))]
    col_ma = ["ma_"+str(k) for k in range(len(df[1][0]))]
    col_nst = ["nst_"+str(k) for k in range(len(df[4][0]))]

    col_st, col_nst
    st_exp = pd.DataFrame(df[0].to_list(), columns=col_st)
    ma_exp = pd.DataFrame(df[1].to_list(), columns=col_ma)
    nst_exp = pd.DataFrame(df[4].to_list(), columns=col_nst)

    exp = pd.concat(
        [st_exp, ma_exp, df.loc[:, [2, 3]], nst_exp, df[5]], axis=1)
    keep_id = list(exp.drop_duplicates().index)

    memory_clean = memory
    memory_clean._samples = [memory_clean._samples[i] for i in keep_id]

    return memory_clean


def init_count(environment):

    states_free_tab = []
    n = environment._state_dim
    grid_size = n**0.5

    states_block = [point[0] * grid_size + point[1]
                    for point in environment.env.blocks]

    for i in range(n):
        if i not in states_block:
            states_free_tab.append(i)

    states_free_tab = np.array(states_free_tab)

    count_explore = np.zeros(len(states_free_tab))

    return count_explore, states_free_tab


def exploration_count(state, count_explore, states_free_tab):

    count = count_explore
    i = np.where(state > 0)[0][0]
    i_free_state = np.where(np.array(states_free_tab) == i)[0][0]
    count[i_free_state] += 1
    return count


def simu_online_eigen(environment, param_agent, eigenoption=True, file_save=""):

    #  --------------------------------------------------------------------------------
    #   init count expore for environment overlapping and exploration ratio
    count_explore, states_free_tab = init_count(environment)
#  --------------------------------------------------------------------------------

    result_compile = []
    steps = 0

    render = param_agent["render"]
    reward_cumul = 0

    #     No mask for simple environment
    if environment.env_name == "SimpleGrid":
        action_space = np.arange(environment.action_space.n)
        action_no_mask = np.full((environment.action_space.n), 1)
        possible_action = action_space

    memory = Memory(param_agent["memory"])

    avg_loss = 0
    avg_loss_r = 0
    avg_loss_N = 0
    avg_loss_psi = 0
    avg_loss_phi = 0

    agent_LSFM = AgentLSFM(environment, param_agent)
    model_LSFM = agent_LSFM.model_LSFM

    value, vect = agent_LSFM.eigen_decomp(model_LSFM)

    count_option = []

    for i in range(param_agent["num_episodes"]):

        reward_ep_cumul = 0
        cnt = 0
        avg_loss = 0

        state = environment.reset()

        if agent_LSFM.dim_latent == 0:
            agent_LSFM.plot_eigenvect(
                model_LSFM, file_save + "_" + "eigenvect_" + str(i) + "_ep.png")

        while True:

            if render:
                environment.render()

                # Update Fa eigen_decomp every 50 steps
            if steps % 50 == 0:
                agent_LSFM.eigen_decomp(agent_LSFM.model_LSFM)

            #  --------------------------------------------------------------------------------
            #   compute count expore for environment overlapping and exploration ratio
            count_explore = exploration_count(
                state, count_explore, states_free_tab)
            overlapping = sum([elem - 1 for elem in count_explore if elem > 1])
            exploration_ratio = 1 - \
                len([elem for elem in count_explore if elem <= 0]) / \
                len(count_explore)

            if agent_LSFM.eigenoption[0] > -1:
                count_option.append(agent_LSFM.eigenoption[0])
            #  --------------------------------------------------------------------------------

    #             -----------------------------
    #             random Policy

            phis = agent_LSFM.model_LSFM(state.reshape(1, -1))["phi"]
            mask_action = [1, 1, 1, 1]

            if eigenoption:
                actions = agent_LSFM.exploration_online(model_LSFM,
                                                        phis,
                                                        mask_action,
                                                        steps,
                                                        start_eigenoption=param_agent["start_eigenoption"])
            else:
                actions = agent_LSFM.choose_action_random()

            action = actions[0]

            next_state, reward, done, info = environment.step(action)

            if param_agent["RANDOM_REWARD_STD"] > 0:
                reward = np.random.normal(1.0, RANDOM_REWARD_STD)

            reward_ep_cumul += reward
            reward_cumul += reward

            if done:
                next_state = np.zeros(environment._state_dim)

    #             -----------------------------
    #             add Sample in memory
    #             -----------------------------
            memory.add_sample(
                (state, action_no_mask, action, reward, next_state, done))
            if steps % 200 == 0:
                memory = clean_duplicate(memory)
                print("memory clean", steps)
    #             -----------------------------
    #             TRAIN LSFM
    #             -----------------------------

            loss_all = agent_LSFM.train_LSFM(
                model_LSFM,
                memory,
                param_agent["filter_done"])

            avg_loss += loss_all[0]
            avg_loss_r += loss_all[1]
            avg_loss_N += loss_all[2]
            avg_loss_psi += loss_all[3]
            avg_loss_phi += loss_all[4]

            state = next_state

            steps += 1
            if steps % 1000 == 0:
                print("steps : {}, episode : {}".format(steps, i))

            if cnt == param_agent["steps_max_episode"]:
                done = True

            if done:

                if cnt != 0:
                    avg_loss /= cnt
                    avg_loss_r /= cnt
                    avg_loss_N /= cnt
                    avg_loss_psi /= cnt
                    avg_loss_phi /= cnt

                else:
                    avg_loss = 0

                result = [i, cnt, steps, reward_ep_cumul,
                          reward_cumul, avg_loss, avg_loss_r, avg_loss_N, avg_loss_psi,
                          avg_loss_phi, overlapping, exploration_ratio,
                          agent_LSFM.eigen_exploration, count_option]

                result_compile.append(result)

                if i % 1 == 0:

                    result_print = [y for x in [
                        result[0:3], result[5:]] for y in x]
                    print(
                        "Ep: {:03d}, stEp: {:03d}, stAll: {:04d}, L: {:0.4f}, Lr: {:0.4f},Ln: {:0.4f}, Lpsi: {:0.4f}, Lphi: {:0.4f}".format(*result_print[:]))

    #             with train_writer.as_default():
    #                 tf.summary.scalar('reward', cnt, step=i)
    #                 tf.summary.scalar('avg loss', avg_loss, step=i)
                break

            cnt += 1
    return result_compile, agent_LSFM, memory


def experience_oneline_eigenoption(environment, param_agent,  eigenoption=True, file_save=""):

    data = pd.DataFrame()

    for k in range(param_agent["run"]):

        result_compile, agent_LSFM, memory = simu_online_eigen(
            environment, param_agent, eigenoption, file_save)

        data_train_df = pd.DataFrame(result_compile, columns=[
            "Episode",
            "Step",
            "cum_step",
            "Reward",
            "Reward_cum",
            "Avg_loss",
            "Avg_loss_r",
            "Avg_loss_N",
            "Avg_loss_psi",
            "Avg_loss_phip1",
            "overlappping",
            "exploration_ratio",
            "eigen_exploration",
            "count_option"])

        data_train_df["carac"] = carac_model(param_agent)
        data_train_df["run"] = k
        data = pd.concat([data, data_train_df])

    return data, agent_LSFM, memory


def carac_model(param):
    if param["train_LSFM"]:
        calcul = "LSFM"
    else:
        if param["train_on_Q_latent"]:
            calcul = "Q_latent"
        else:
            calcul = "Q"
        if param["model_Q_Lin"]:
            calcul += "_Lin"
        else:
            calcul += "_Deep"

    return calcul


def loss_func(ypred, ytrue):
    #     MSE
    return keras.losses.mean_squared_error(ypred, ytrue)


def simu_offline_LSFM(env, param_agent, buffer):

    result_compile = []
    steps = 0

#     No mask for simple env
    if env.env_name == "SimpleGrid":
        action_space = np.arange(env.action_space.n)
        action_no_mask = np.full((env.action_space.n), 1)
        possible_action = action_space

    memory = Memory(param_agent["memory"], buffer)

    avg_loss = 0
    avg_loss_r = 0
    avg_loss_N = 0
    avg_loss_psi = 0
    avg_loss_phi = 0

    agent_LSFM = AgentLSFM(env, param_agent)
    model_LSFM = agent_LSFM.model_LSFM

    for i in range(param_agent["num_steps"]):

        avg_loss = 0

#         TRAIN ON BUFFER ONLY
        if buffer != None:

            loss_all = agent_LSFM.train_LSFM(
                model_LSFM,
                memory,
                param_agent["filter_done"])

            avg_loss += loss_all[0]
            avg_loss_r += loss_all[1]
            avg_loss_N += loss_all[2]
            avg_loss_psi += loss_all[3]
            avg_loss_phi += loss_all[4]
            if steps > 0:
                if steps % 200 == 0:
                    avg_loss /= steps
                    avg_loss_r /= steps
                    avg_loss_N /= steps
                    avg_loss_psi /= steps
                    avg_loss_phi /= steps

                    result = [steps, avg_loss, avg_loss_r,
                              avg_loss_N, avg_loss_psi, avg_loss_phi]

                    print("cumul_step: {:06d}, L: {:0.4f}, Lr: {:0.4f}, LN: {:0.4f}, Lpsi: {:0.4f}, Lphi: {:0.4f}".format(
                        *result[:]))

                    result_compile.append(result)

            steps += 1

    return result_compile, agent_LSFM, memory


def experience_offline_LSFM(env, param_agent, buffer):

    data = pd.DataFrame()

    for k in range(param_agent["run"]):

        result_compile, agent_LSFM, memory = simu_offline_LSFM(
            env, param_agent, buffer)

        memory.write("memory.csv")
        data_train_df = pd.DataFrame(result_compile, columns=[
            "cum_step",
            "Avg_loss",
            "Avg_loss_r",
            "Avg_loss_N",
            "Avg_loss_psi"])

        data_train_df["carac"] = carac_model(param_agent)
        data_train_df["run"] = k
        data = pd.concat([data, data_train_df])

    return data, agent_LSFM


def test_error_Ma_rewClassif_offline_LSFM(env, param_agent, buffer, save_model_path, reward_parser):

    result_compile = []
    steps = 0

#     No mask for simple env
    if env.env_name == "SimpleGrid":
        action_space = np.arange(env.action_space.n)

    agent_LSFM = AgentLSFM(env, param_agent, save_model_path)
    model_LSFM = agent_LSFM.model_LSFM

    y_true_compile = []
    y_pred_compile = []

    for i in range(param_agent["num_episodes"]):

        cnt = 0

        state, *_ = buffer[steps]

        states = np.array(state).reshape(1, -1)
        phi_pred = model_LSFM(states)["phi"]

        while True:

            state, action_mask, action, reward, next_state, done = buffer[steps]

            states = np.array(state).reshape(1, -1)
            actions = [action]
            rewards = [reward]
            next_states = np.array(next_state).reshape(1, -1)
            terminates = [done]

            target_psi = agent_LSFM.target_psi(
                model_LSFM, states, actions, next_states, terminates, rewards, param_agent["filter_done"])

#             loss_tot, loss_r ,  loss_N , loss_psi , loss_phi = agent_LSFM.losses_all(
#                 states, actions, rewards,next_states, target_psi, agent_LSFM)

            phi = model_LSFM(states)["phi"]
            phi_sp1 = agent_LSFM.model_LSFM(next_states)["phi"]

            logits_r = agent_LSFM.get_from_actions(
                states, actions, model_LSFM, "ra")
            reward_onestep = tf.keras.backend.get_value(logits_r)[0]

            norm_phi = tf.keras.backend.get_value(tf.norm(phi, axis=1))[0]

            logits_psi = agent_LSFM.get_from_actions(
                states, actions, model_LSFM, "Fa")
            phi_sp1_pred = agent_LSFM.get_from_actions(
                states, actions, model_LSFM, "Ma")

            reward_n_step = tf.keras.backend.get_value(
                agent_LSFM.next_rewards(phi_pred, actions, model_LSFM))[0]

            y_true = [i, cnt, steps,  done, action, discretize(reward, reward_parser),
                      1., target_psi,   phi_sp1, discretize(reward, reward_parser)]

            y_pred = [i, cnt, steps,  done, action, reward_onestep,
                      norm_phi, logits_psi, phi_sp1_pred, reward_n_step]

            y_true_compile.append(y_true)
            y_pred_compile.append(y_pred)

            phi_pred = phi_sp1_pred

            steps += 1

            if done:

                if i % 1 == 0:
                    print(
                        "Ep: {:03d}, stEp: {:03d}, stAll: {:04d}".format(*y_true[:]))

                break

            cnt += 1

    y_true_df = pd.DataFrame(y_true_compile, columns=[
        "Ep",
        "stEp",
        "stAll",
        "finished",
        "action",
        "reward_one-step",
        "Norm_phi",
        "target_psi",
        "phi_sp1",
        "reward_n-step"])
    y_pred_df = pd.DataFrame(y_pred_compile, columns=[
        "Ep",
        "stEp",
        "stAll",
        "finished",
        "action",
        "reward_one-step",
        "Norm_phi",
        "target_psi",
        "phi_sp1",
        "reward_n-step"])

    return y_pred_df, y_true_df

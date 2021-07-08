import pandas as pd
import numpy as np
from modules.memory import Memory
import tensorflow as tf
from tensorflow import keras

from modules.dqn import Agent_Q
from modules.lsfm import Agent, discretize


import time

def carac_model(param) : 
    if param["train_LSFM"] : calcul = "LSFM"
    else : 
        if param["train_on_Q_latent"]  : calcul = "Q_latent"
        else : calcul = "Q"
        if param["model_Q_Lin"]  : calcul+="_Lin"
        else : calcul+="_Deep"
    
    return calcul

def loss_func(ypred, ytrue) : 
#     MSE
    return keras.losses.mean_squared_error(ypred, ytrue)
    

def simu_online_LSFM(env,param_agent, buffer=None) : 
      
    result_compile = []
    steps = 0
    render = param_agent["render"]
    reward_cumul = 0

#     No mask for simple env
    if env.env_name == "SimpleGrid" : 
        action_space = np.arange(env.action_space.n)
        action_no_mask = np.full((env.action_space.n), 1)   
        possible_action = action_space

    
    memory = Memory(param_agent["memory"], buffer)

    avg_loss = 0
    avg_loss_r  = 0
    avg_loss_N  = 0
    avg_loss_psi  = 0
    
    agent_LSFM = Agent(env, param_agent)
    model_LSFM = agent_LSFM.model_LSFM
    
 ### TRANSITION MATRIX : get the period for the calculation, and init M
    step_loss_phisp1 = param_agent["avg_loss_phisp1" ]
    if step_loss_phisp1 > 0 : 
        agent_LSFM.M_transi(model_LSFM)
    
    
    for i in range(param_agent["num_episodes"]):

        reward_ep_cumul = 0
        cnt = 0
        avg_loss = 0
        
### TRANSITION MATRIX :  initialize the loss
        loss_phisp1  = 0

        state = env.reset() 
        if step_loss_phisp1 > 0 :  
            phis_pred = model_LSFM(state.reshape(1,-1))["phi"]

        while True:
            if render:
                env.render()


#             -----------------------------
#             random Policy
            action, eps = agent_LSFM.choose_action_random()
            next_state, reward, done, info = env.step(action)

            phi_prime = agent_LSFM.model_LSFM(next_state.reshape(1,-1))["phi"]
            next_state_latent = np.array(tf.reshape(phi_prime, phi_prime.shape[-1]))


### TRANSITION MATRIX :update every step_loss_phisp1 the transition matrix
            if step_loss_phisp1 > 0 and (steps + 1) % step_loss_phisp1 == 0 :    
                start_time = time.time()
                agent_LSFM.M_transi(model_LSFM)
                print("--- steps {} : M_transi take {} seconds ---".format(
                    steps, time.time() - start_time) )
            
### TRANSITION MATRIX : Get the prediction error of the next latent space
            if step_loss_phisp1 > 0 :             
                phisp1_pred = agent_LSFM.next_phi_pred(model_LSFM, phis_pred,[action])
                loss_phisp1   += loss_func(phi_prime, phisp1_pred).numpy()[0]

#                 print('loss_phisp1', loss_func(phi_prime, phisp1_pred).numpy()[0], loss_phisp1)

            if param_agent["RANDOM_REWARD_STD"] > 0:
                reward = np.random.normal(1.0, RANDOM_REWARD_STD)

            reward_ep_cumul += reward
            reward_cumul += reward

            if done:
                next_state = np.zeros(env._state_dim)
                next_state_latent = np.zeros(param_agent["latent_space"])


#             -----------------------------
#             add Sample in memory
#             -----------------------------      
            memory.add_sample((state, action_no_mask, action, reward, next_state, done))
                
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
           

            state = next_state

            if step_loss_phisp1 > 0 :                     
                phis_pred = phisp1_pred

            steps += 1
            if steps % 1000 == 0 : print("steps : {}, episode : {}, eps : {}".format(steps, i, eps))

                
            
            if done:

                if cnt != 0:
                    avg_loss /= cnt
                    avg_loss_r /= cnt
                    avg_loss_N /= cnt
                    avg_loss_psi /= cnt
                    loss_phisp1 /= cnt

                else:
                    avg_loss = 0
                    loss_phisp1 = 0

                result = [i, cnt, steps, reward_ep_cumul,
                          reward_cumul, eps, avg_loss, avg_loss_r, avg_loss_N, avg_loss_psi, loss_phisp1]
                if i % 1  == 0 : 

                    result_print = [y for x in [result[0:3], result[5:]] for y in x]
                    print(
                    "Ep: {:03d}, stEp: {:03d}, stAll: {:04d}, eps: {:0.4f}, L: {:0.4f}, Lr: {:0.4f},Ln: {:0.4f}, Lpsi: {:0.4f}, Lphi: {:0.4f}".format(*result_print[:]))

                result_compile.append(result)
    #             with train_writer.as_default():
    #                 tf.summary.scalar('reward', cnt, step=i)
    #                 tf.summary.scalar('avg loss', avg_loss, step=i)
                break

            cnt += 1

                
    
    return result_compile, agent_LSFM, memory
    
    



def experience_online_LSFM(env, param_agent, buffer = None) : 
    
    
    data = pd.DataFrame()
    
    for k in range(param_agent["run"]) : 
                
        
        result_compile, agent_LSFM, memory = simu_online_LSFM(env,param_agent, buffer=None)
    
        memory.write("memory.csv")
        data_train_df = pd.DataFrame(result_compile, columns=[ 
        "Episode", 
        "Step", 
        "cum_step",
        "Reward", 
        "Reward_cum",
        "Eps",  
        "Avg_loss" ,  
        "Avg_loss_r",  
        "Avg_loss_N",  
        "Avg_loss_psi", "Avg_loss_phi"])

        data_train_df["carac"] = carac_model(param_agent) 
        data_train_df["run"] = k
        data = pd.concat([data, data_train_df])

    return data, agent_LSFM


def simu_offline_LSFM(env,param_agent, buffer) : 
    
    result_compile = []
    steps = 0

#     No mask for simple env
    if env.env_name == "SimpleGrid" : 
        action_space = np.arange(env.action_space.n)
        action_no_mask = np.full((env.action_space.n), 1)   
        possible_action = action_space

    memory = Memory(param_agent["memory"], buffer)

    avg_loss = 0
    avg_loss_r  = 0
    avg_loss_N  = 0
    avg_loss_psi  = 0

    agent_LSFM = Agent(env, param_agent)
    model_LSFM = agent_LSFM.model_LSFM
    
    

    for i in range(param_agent["num_steps"]):

        avg_loss = 0

#         TRAIN ON BUFFER ONLY 
        if buffer != None : 

            loss_all = agent_LSFM.train_LSFM(
                        model_LSFM, 
                        memory, 
                        param_agent["filter_done"])
            
            avg_loss += loss_all[0]
            avg_loss_r += loss_all[1]
            avg_loss_N += loss_all[2]
            avg_loss_psi += loss_all[3]


            if steps>0 : 
                if steps % 200 == 0:
                    avg_loss /= steps
                    avg_loss_r /= steps
                    avg_loss_N /= steps
                    avg_loss_psi /= steps
#                     loss_phisp1 /= steps

                    result = [steps, avg_loss, avg_loss_r, avg_loss_N, avg_loss_psi]
    
                    print("cumul_step: {:06d}, L: {:0.4f}, Lr: {:0.4f}, LN: {:0.4f}, Lpsi: {:0.4f}".format(*result[:]))

                    result_compile.append(result)

            steps += 1

    return result_compile, agent_LSFM, memory
       
    
    
    


def experience_offline_LSFM(env,param_agent,buffer) : 
    
    data = pd.DataFrame()
    
    for k in range(param_agent["run"]) : 
    
        result_compile, agent_LSFM, memory = simu_offline_LSFM(env,param_agent, buffer)
    
        memory.write("memory.csv")
        data_train_df = pd.DataFrame(result_compile, columns=[ 
        "cum_step",
        "Avg_loss" ,  
        "Avg_loss_r",  
        "Avg_loss_N",  
        "Avg_loss_psi"])

        data_train_df["carac"] = carac_model(param_agent) 
        data_train_df["run"] = k
        data = pd.concat([data, data_train_df])
        
        
    return data, agent_LSFM

def test_error_Ma_rewClassif_offline_LSFM(env,param_agent, buffer, save_model_path, reward_parser) : 
    
    result_compile = []
    steps = 0

#     No mask for simple env
    if env.env_name == "SimpleGrid" : 
        action_space = np.arange(env.action_space.n)
        action_no_mask = np.full((env.action_space.n), 1)   
        possible_action = action_space

    memory = Memory(param_agent["memory"], buffer)

    avg_loss = 0
    avg_loss_r  = 0
    avg_loss_N  = 0
    avg_loss_psi  = 0

    agent_LSFM = Agent(env, param_agent, save_model_path)
    model_LSFM = agent_LSFM.model_LSFM

    y_true_compile = []
    y_pred_compile = []

    for i in range(param_agent["num_episodes"]):
        
        cnt = 0
        
        state,*_= buffer[steps]
        
        states = np.array(state).reshape(1,-1)
        phi_pred = model_LSFM(states)["phi"]
        
        while True:

            avg_loss = 0

            state, action_mask, action, reward, next_state, done = buffer[steps]
            
            states = np.array(state).reshape(1,-1)
            action_masks = np.array(action_mask).reshape(1,-1)
            actions = [action]
            rewards = [reward]
            next_states = np.array(next_state).reshape(1,-1)
            terminates = [done]
            
            
            target_psi = agent_LSFM.target_psi(
                model_LSFM, states, actions, next_states,terminates,rewards, param_agent["filter_done"])

#             loss_tot, loss_r ,  loss_N , loss_psi , loss_phi = agent_LSFM.losses_all(
#                 states, actions, rewards,next_states, target_psi, agent_LSFM)
            
            
            phi = model_LSFM(states)["phi"]
            phi_sp1 = agent_LSFM.model_LSFM(next_states)["phi"]
#             next_state_latent = np.array(tf.reshape(phi_sp1, phi_sp1.shape[-1]))


#             y_true = [reward, 1., target_psi, next_state_latent]

            logits_r = agent_LSFM.get_from_actions( states, actions, model_LSFM, "ra")
            reward_onestep = tf.keras.backend.get_value(logits_r)[0]
#             reward_onestep = tf.keras.backend.get_value(logits_r)[0]
            
            norm_phi = tf.keras.backend.get_value(tf.norm(phi, axis=1))[0]
            
            logits_psi = agent_LSFM.get_from_actions( states, actions, model_LSFM, "Fa")
            phi_sp1_pred = agent_LSFM.get_from_actions( states, actions, model_LSFM, "Ma") 

            reward_n_step = tf.keras.backend.get_value(
                agent_LSFM.next_rewards( phi_pred, actions, model_LSFM))[0]

            
            
            y_true = [i, cnt, steps,  done, action, discretize(reward, reward_parser),
                      1., target_psi,   phi_sp1, discretize(reward, reward_parser)]
            
            
            
            y_pred = [i, cnt, steps,  done, action, reward_onestep,
                      norm_phi, logits_psi, phi_sp1_pred, reward_n_step]
            
            y_true_compile.append(y_true)
            y_pred_compile.append(y_pred)
            
            
            
            
            phi_pred = phi_sp1_pred
            
            
            steps += 1
            
            if done:

#                 if cnt != 0:
#                     avg_loss /= cnt
#                     avg_loss_r /= cnt
#                     avg_loss_N /= cnt
#                     avg_loss_psi /= cnt
#                     loss_phip1 /= cnt

#                 else:
#                     avg_loss = 0
#                     loss_phip1 = 0

               
    
                if i % 1  == 0 : 
                    print(
                    "Ep: {:03d}, stEp: {:03d}, stAll: {:04d}".format(*y_true[:]))

#                 y_true_compile.append(y_true)
#                 y_pred_compile.append(y_true)
    #             with train_writer.as_default():
    #                 tf.summary.scalar('reward', cnt, step=i)
    #                 tf.summary.scalar('avg loss', avg_loss, step=i)
                break

            cnt += 1
            
    y_true_df = pd.DataFrame(y_true_compile, columns=[ 
    "Ep",
    "stEp" ,  
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
    "stEp" ,  
    "stAll",
    "finished",
    "action",
    "reward_one-step",  
    "Norm_phi", 
    "target_psi",
    "phi_sp1",
    "reward_n-step"])
        


    return y_pred_df, y_true_df
       
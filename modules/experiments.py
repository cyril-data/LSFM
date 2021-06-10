import pandas as pd
from modules.dqn import Agent_Q
import numpy as np
from modules.memory import Memory
import tensorflow as tf


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
#     return keras.losses.mean_squared_error(ypred, ytrue)
#     MAE
    return keras.losses.mean_absolute_error(ypred, ytrue)
    

def simu(env,param_agent,agent_LSFM = None, agent_Q = None, buffer = None, buffer_latent = None) : 
      
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
    memory_latent = Memory(param_agent["memory"], buffer_latent)

    avg_loss = 0
    avg_loss_r  = 0
    avg_loss_N  = 0
    avg_loss_psi  = 0

    
    if agent_LSFM!= None : 
        model_LSFM = agent_LSFM.model_LSFM
        
        ### TRANSITION MATRIX : Calcule the M transition matrix on latent space
        if param_agent["avg_loss_phisp1" ] : 
            agent_LSFM.M_transi(model_LSFM)
        
        
    if agent_Q != None : 
        model_Q = agent_Q.model
        model_Q_prev = agent_Q.model_prev
    
    for i in range(param_agent["num_episodes"]):
        state = env.reset()

        reward_ep_cumul = 0
        cnt = 0
        avg_loss = 0
        
### TRANSITION MATRIX :  Get the prediction error of the next latent space
        loss_phisp1  = 0
        if param_agent["avg_loss_phisp1" ] :         
            phis_pred = model_LSFM(state.reshape(1,-1))["phi"]
        
        while True:
            if render:
                env.render()

                
#             -----------------------------
#             Policy
#             -----------------------------
            if param_agent["train_LSFM"] : 
                action, eps = agent_LSFM.choose_action_random()
                next_state, reward, done, info = env.step(action)
                
                phi_prime = agent_LSFM.model_LSFM(next_state.reshape(1,-1))["phi"]
                next_state_latent = np.array(tf.reshape(phi_prime, phi_prime.shape[-1]))
                
                
            else : 
                if param_agent["train_on_Q_latent"] : 
                    
                    state_resh = state.reshape(1,-1)
                    phi = agent_LSFM.model_LSFM(state_resh)["phi"]
                    state_latent = np.array(tf.reshape(phi, phi.shape[-1]))

                    # eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)
                                        
                    action, eps = agent_Q.choose_action(
                                        state_latent, 
                                        model_Q, 
                                        steps, 
                                        possible_action
                                    )
                    
                    next_state, reward, done, info = env.step(action)

                    next_state_resh = next_state.reshape(1,-1)
                    phi_prime = agent_LSFM.model_LSFM(next_state_resh)["phi"]
                    next_state_latent = np.array(tf.reshape(phi_prime, phi_prime.shape[-1]))

                    
                else : 
                # eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)
                    action, eps = agent_Q.choose_action(
                        state, 
                        model_Q, 
                        steps, 
                        possible_action
                    )

                    next_state, reward, done, info = env.step(action)
                    
                    if agent_LSFM!= None : 
                        phi_prime = agent_LSFM.model_LSFM(next_state.reshape(1,-1))["phi"]
                        next_state_latent = np.array(tf.reshape(phi_prime, phi_prime.shape[-1]))

                    
### TRANSITION MATRIX : Get the prediction error of the next latent space
            if param_agent["avg_loss_phisp1"] :                     
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
#             TRAIN
#             -----------------------------
            
            if param_agent["train"] :

                if param_agent["train_LSFM"] : 

                    
                    if buffer == None : 
                        memory.add_sample((state, action_no_mask, action, reward, next_state, done))
                    
                    
                    loss_all = agent_LSFM.train_LSFM(
                        model_LSFM, 
                        memory, 
                        param_agent["filter_done"])

                    avg_loss += loss_all[0]
                    avg_loss_r += loss_all[1]
                    avg_loss_N += loss_all[2]
                    avg_loss_psi += loss_all[3]
                else : 
                    if param_agent["train_on_Q_latent"] : 
                        if buffer_latent == None :
                            memory_latent.add_sample((state_latent,action_no_mask,  action, reward, next_state_latent, done))

                        loss = agent_Q.train_Q(
                        model_Q, 
                        model_Q_prev if param_agent["double_model"] else None,
                        memory_latent, param_agent["filter_done"])
                        avg_loss += loss
                    else :                 
                        memory.add_sample((state,action_no_mask, action, reward, next_state, done))

                        loss = agent_Q.train_Q(
                        model_Q, 
                        model_Q_prev if param_agent["double_model"] else None,
                        memory, param_agent["filter_done"])
                        avg_loss += loss
                
            state = next_state
            
            if param_agent["avg_loss_phisp1"] :                     
                phis_pred = phisp1_pred
                
            steps += 1
            if steps % 1000 == 0 : print("steps : {}, episode : {}, eps : {}".format(steps, i, eps))
#             print("steps : {}, episode : {}".format(steps, i))
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
                if i % 1  == 0 : print(
                    "Episode: {:03d}, step: {:03d}, cumul_step: {:04d}, Reward: {:3.1f}, Reward_cumul: {:3.1f}, eps: {:0.4f}, avg loss: {:0.4f}".format(*result[:-1]))

                result_compile.append(result)
    #             with train_writer.as_default():
    #                 tf.summary.scalar('reward', cnt, step=i)
    #                 tf.summary.scalar('avg loss', avg_loss, step=i)
                break

            cnt += 1
    return result_compile, memory,memory_latent
    
    
def experience(environment, param, agent= None, save_model = None, buffer = None, buffer_latent = None) : 
    
    
    data = pd.DataFrame()
    
    for k in range(param["run"]) : 
                
        agent_Q_simu = Agent_Q(environment, param = param, save_model = save_model)      
        # Restore the weights
#         if save_model != None : 
#             for t in zip(agent_Q_simu.model.trainable_variables):
#                 print("agent_Q_simu", t)

        
        result_compile, memory,memory_latent = simu(environment,param, agent, agent_Q_simu, buffer=buffer,buffer_latent=buffer_latent )
    
    
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

        data_train_df["carac"] = carac_model(param) 
        data_train_df["run"] = k
        data = pd.concat([data, data_train_df])

    return data, agent_Q_simu
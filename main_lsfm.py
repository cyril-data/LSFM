import numpy as np
import pandas as pd


from modules.lsfm import Agent
from modules.environnement import custom_env
from modules.params import PARAM_ENV,PARAM_AGENT
from modules.experiments import experience





env_name = "SimpleGrid"
env = custom_env(env_name, PARAM_ENV)

agent = Agent(env, PARAM_AGENT)

state = env.reset()
env.render()

# initialisation data
data = pd.DataFrame()
        
# launch experience of simulations
data, agent_Q = experience(env, PARAM_AGENT, agent)





# train

# for _ in range(PARAM_AGENT.num_episodes): 
        
#     # prepare for the next trial
#     sfql.reset()
#     q.reset()
    
#     # next trial
#     for _ in range(n_tasks):
        
#         # define new task
#         rewards = dict(zip(['1', '2', '3'], list(np.random.uniform(low=-1.0, high=1.0, size=3))))
#         task = Shapes(maze, rewards)
        
#         # solve the task with sfql
#         print('\nsolving with SFQL')
#         sfql.add_task(task)
#         sfql.set_active_task()
#         for _ in range(n_samples):
#             sfql.next_sample()
        
#         # solve the same task with q
#         print('\nsolving with QL')
#         q.next_task(task)
#         for _ in range(n_samples):
#             q.next_sample()
    
#     # update performance statistics
#     avg_data_sfql = avg_data_sfql + np.array(sfql.reward_hist) / float(n_trials)
#     cum_data_sfql = cum_data_sfql + np.array(sfql.cum_reward_hist) / float(n_trials)
#     avg_data_q = avg_data_q + np.array(q.reward_hist) / float(n_trials)
#     cum_data_q = cum_data_q + np.array(q.cum_reward_hist) / float(n_trials)

# # plot the cumulative return per trial, averaged 
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot(avg_data_sfql, label='SFQL')
# plt.plot(avg_data_q, label='Q')
# plt.xlabel('samples')
# plt.ylabel('cumulative reward')
# plt.legend()
# plt.title('Cumulative Training Reward Per Task')
# plt.savefig('figures/sfql_cumulative_return_per_task.png')
# plt.show()

# # plot the gross cumulative return, averaged
# plt.clf()
# plt.figure(figsize=(5, 5))
# plt.plot(cum_data_sfql, label='SFQl')
# plt.plot(cum_data_q, label='Q')
# plt.xlabel('samples')
# plt.ylabel('cumulative reward')
# plt.legend()
# plt.title('Total Cumulative Training Reward')
# plt.savefig('figures/sfql_cumulative_return_total.png')
# plt.show()
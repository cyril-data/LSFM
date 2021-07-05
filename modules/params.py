from tensorflow import keras





# param_env = {
#     "agent_pos":[0, 0], 
#     "goal_pos":[[9,0] ],
#     "reward_minmax":[-1.,1.],
#     "reward_pos":[[0, 0],[0, 1],[0, 2],[0, 3],[0, 4],[0, 5],[0, 6],[0, 7],[0, 8],[0, 9], 
#                   [1, 0],[1, 1],[1, 2],[1, 3],[1, 4],[1, 5],[1, 6],[1, 7],[1, 8],[1, 9], 
#                                                             [2, 6],[2, 7],[2, 8],[2, 9], 
#                                                             [3, 6],[3, 7],[3, 8],[3, 9], 
#                                                             [4, 6],[4, 7],[4, 8],[4, 9], 
#                                                             [5, 6],[5, 7],[5, 8],[5, 9], 
#                                                             [6, 6],[6, 7],[6, 8],[6, 9], 
#                                                             [7, 6],[7, 7],[7, 8],[7, 9], 
#                   [8, 0],[8, 1],[8, 2],[8, 3],[8, 4],[8, 5],[8, 6],[8, 7],[8, 8],[8, 9], 
#                   [9, 0],[9, 1],[9, 2],[9, 3],[9, 4],[9, 5],[9, 6],[9, 7],[9, 8],[9, 9]],
#     "grid_size" : 10,
#     "pattern": "empty",
# #     "pattern":["user_defined", [ [2,0],[2,1],[2,2],[2,3],[2,4],[2,5],
# #                                [7,0],[7,1],[7,2],[7,3],[7,4],[7,5]]],
#     "obs_mode":"index",
# }

# param_env = {
#     "agent_pos":[0, 0], 
#     "goal_pos":[[10,0] ],
#     "reward_minmax":[-1.,0.],
#     "reward_user_defined":True,
#     "reward_pos":[
#         [0, 0],[0, 1],[0, 2],[0, 3],[0, 4],[0, 5],[0, 6],[0, 7],[0, 8],[0, 9],[0, 10],
#         [1, 0],[1, 1],[1, 2],[1, 3],[1, 4],[1, 5],[1, 6],[1, 7],[1, 8],[1, 9],[1, 10],
#                                                                 [2, 8],[2, 9],[2, 10],
#                                                                 [3, 8],[3, 9],[3, 10],
#                                                                 [4, 8],[4, 9],[4, 10],
#                                                                 [5, 8],[5, 9],[5, 10],
#                                                                 [6, 8],[6, 9],[6, 10],
#                                                                 [7, 8],[7, 9],[7, 10],
#                                                                 [8, 8],[8, 9],[8, 10],
#         [9, 0],[9, 1],[9, 2],[9, 3],[9, 4],[9, 5],[9, 6],[9, 7],[9, 8],[9, 9],[9, 10],
#         [10,0],[10,1],[10,2],[10,3],[10,4],[10,5],[10,6],[10,7],[10,8],[10,9],[10,10]
#                 ],
#     "grid_size" : 11,
#     "pattern": "empty",
#     "obs_mode":"index",
# }



# ENV_NAME = "SimpleGrid"
# PARAM_ENV = {
#     "agent_pos":[0, 0], 
#     "goal_pos":[[5,0] ],
#     "reward_minmax":[-1.,0.],
#     "reward_user_defined":True,
#     "reward_pos":[[0, 0],[0, 1],[0, 2],[0, 3],[0, 4],[0, 5], 
#                   [1, 0],[1, 1],[1, 2],[1, 3],[1, 4],[1, 5], 
#                                               [2, 4],[2, 5], 
#                                               [3, 4],[3, 5], 
#                   [4, 0],[4, 1],[4, 2],[4, 3],[4, 4],[4, 5], 
#                   [5, 0],[5, 1],[5, 2],[5, 3],[5, 4],[5, 5]],
#     "grid_size" : 6,
#     "pattern": "empty",
#     "obs_mode":"index",
# }

ENV_NAME = "custom"
PARAM_ENV = {
    "action_space" : 88, 
    "state_dim"  : 324
}



# dimension du réseau : 
latent_dimension = 250
hidden_ratio = 1.

# paramètre de l'agent 
PARAM_AGENT = {
    "gamma": 0.90,
    "optimizer_LSFM": keras.optimizers.Adam(),
    "optimizer_Q": keras.optimizers.Adam(),
    "alpha_r" : 1.,
    "alpha_N" : 0.1,
    "alpha_psi" : 0.01,
    "alpha_phi" : 0.01,
    "policy": {
        "type": ["eps-greedy", "exponantial"],
        "eps-greedy": {
            "exponantial": {
                "eps_max": 1.,
                "eps_min": 0.01,
                "lambda": 0.0002
            },
            "constant": {
                "eps": 0.01
            }
        }
    },
    "memory" : 500000,
    "latent_space": latent_dimension,
    "hidden_dim_ratio": hidden_ratio,
    "num_episodes" : 10,
    "num_steps" : 10,
    "batch_size" : 32,
    "RANDOM_REWARD_STD" : -1.0, 
    "double_model" : False,
    "tau": 0.08,
    "filter_done" : True,
    "train_LSFM" : True,
    "train_on_Q_latent": False,
    "model_Q_Lin": False,
    "train" : True,
    "avg_loss_phisp1" : False,
    "run" : 1,
    "render": False,
    "avg_loss_phisp1" : 1000,
}






#   actions_down = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# for i in range(10) : 
#     action = actions_down[i]
#     next_state, reward, done, info = env.step(action)

#     print("action", action)
#     print(" 0 - Up  /  1 - Down  /  2 - Left  / 3 - Right")
# #     print("next_state", next_state)
#     print("reward", reward)
#     print("done", done)
#     env.render()
    
#     state = next_state




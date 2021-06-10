
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input

class Agent_Q:
        
    def __init__(self, enviroment, param = {}, save_model=None):
        self.param = param
        self._action_size = enviroment.action_space.n
        self._state_dim = enviroment._state_dim
        self.dim_latent = self.param["latent_space"]
        self.hidden_dim = int(self.param["latent_space"] * self.param["hidden_dim_ratio"])
        self.batch_size = self.param["batch_size"]

        
        if save_model == None : 
            if  param["model_Q_Lin"] : 
                print("Agent_Q.model : initialization model_Q_Lin")
                self.model = keras.Sequential()
                self.model.add(layers.Dense(self._action_size, use_bias=False))
                self.model_prev = keras.models.clone_model(self.model)
                

                
            else : 
                print("Agent_Q.model : initialization model_DQN")
            
                self.model = keras.Sequential()
                self.model.add(layers.Dense(self.dim_latent, activation="relu", name="latent"))
                self.model.add(layers.Dense(self._action_size))
                self.model_prev = keras.models.clone_model(self.model)

                
        else : 
            print("Agent_Q.model : load model ")

            self.model               = keras.models.load_model(save_model)
            self.model_prev          = keras.models.load_model(save_model)
            
            for i in range(len(self.model.layers)-1):
                print("layers name",i,  self.model.layers[i].name)
                self.model.layers[i].trainable = False  
                self.model_prev.layers[i].trainable = False  

            
         # Initialize policy
#         policy = radom by default
        self.policy = "random"
        self.epsilon = 1.    
        if (self.param["policy"]["type"][0] == "eps-greedy"):
            self.policy = "eps-greedy"

            if (self.param["policy"]["type"][1] == "exponantial"):
                self.pi_1 = "exponantial"

                self.epsilon = self.param["policy"][self.policy][self.pi_1]["eps_max"]
                self.eps_max = self.param["policy"][self.policy][self.pi_1]["eps_max"]
                self.eps_min = self.param["policy"][self.policy][self.pi_1]["eps_min"]
                self._lambda = self.param["policy"][self.policy][self.pi_1]["lambda"]

            if (self.param["policy"]["type"][1] == "constant"):
                self.pi_1 = "constant"
                self.epsilon = self.param["policy"][self.policy][self.pi_1]["eps"]
        
        
        
    def choose_action(self, state, model_Q, steps, possible_action):
        
        if (self.policy == "eps-greedy"):
            if (self.pi_1 == "exponantial"):
                self.epsilon = self.eps_min + \
                    (self.eps_max - self.eps_min) * \
                    math.exp(-self._lambda * steps)

            if random.random() < self.epsilon:
                return random.randint(0, self._action_size - 1), self.epsilon
            else:
                
                action_greedy, eps = self.choose_action_greedy(state, model_Q, steps, possible_action)
                
                return action_greedy, self.epsilon
        else : 
            if (self.policy == "random") : 
                return random.randint(0, self._action_size - 1), self.epsilon
             
    def choose_action_random(self):
        return random.randint(0, self._action_size - 1), self.epsilon
            
    def choose_action_greedy(self, state, model_Q , steps, possible_action):

        Q_values = model_Q(state.reshape(1, -1))
#         print("Q_values , a", Q_values,np.argmax(Q_values) )       
        return np.argmax(Q_values), self.epsilon

    def get_from_actions( self, states, actions, model, var) : 
        var_all_a = model(states)[var]
        aggr = []
        for k in range(var_all_a.shape[0]) : 
            aggr.append(var_all_a[k, actions[k]])
        return tf.convert_to_tensor(aggr)
    
    
    
        

    def loss(self,y_pred,y_true) : 
#         MSE
        return keras.losses.mean_squared_error(y_pred,y_true)
#         MAE
#         return keras.losses.mean_absolute_error(y_pred,y_true)
            
        

#           ************************************************
#           target_Q on latent reward-predictive state space
#           ************************************************
    def target_Q(self, model_Q, model_Q_prev, states,actions, next_states,terminate,rewards):
        
        
        
#       prediction of Q(s,*) and Q(s',*)
        prim_qt =  model_Q(states)
        prim_qtp1 = model_Q_prev(next_states)
        
#       creation of a filter to determine whitch episode of batch is a terminal state
        filter_idxs = tf.dtypes.cast(tf.logical_not(terminate), tf.float32)
        
#       recompose id of the batch 
        batch_idxs = tf.range(self.batch_size)

        if model_Q_prev is None:
#          updates =  rewards + gamma * max(  Q(s',*)   )
#          for s not terminated
            updates = tf.add(
                rewards ,tf.multiply(
                    filter_idxs ,tf.multiply(
                        self.param["gamma"], tf.reduce_max(prim_qtp1, axis=1)
                    )
                ) 
            )
            
        else:
#          updates =  rewards + gamma * max(  Q^old(s',*))
#          for s not terminated

            prim_action_tp1 = tf.argmax(prim_qtp1, axis=1)
            q_from_target = model_Q_prev(next_states)
            indices = np.transpose(np.array([batch_idxs,prim_action_tp1]))
            
            updates = tf.add(
                rewards ,tf.multiply(
                    filter_idxs ,tf.multiply(
                        self.param["gamma"], tf.gather_nd(q_from_target, indices)
                    )
                ) 
            )

#       create the target_q / 
#                           target_q = Q(s,*) 
#                           target_q[batch_idxs,actions] = updates
        indices = np.transpose(np.array([batch_idxs,actions]))
        target_q = tf.tensor_scatter_nd_update(prim_qt, indices, updates)
                
        return target_q
#         calculate the loss : prim_qt / target_q


#           ***********************************************
#           train_Q on latent reward-predictive state space
#           ***********************************************

    def train_Q(self, model_Q, model_Q_prev, memory, filter_done):
        if memory.num_samples < self.batch_size * 3:
            return 0
        batch = memory.sample(self.batch_size)
        
#        tensor conversion
        states = tf.convert_to_tensor(np.array([val[0] for val in batch]))
        actions = tf.convert_to_tensor(np.array([val[1] for val in batch]))
        rewards = tf.convert_to_tensor(np.array([val[2] for val in batch]).astype(np.float32))        
        next_states = tf.convert_to_tensor(np.array([val[3] for val in batch]))
        terminate = tf.convert_to_tensor(np.array([val[4] for val in batch]))


        target_Q = self.target_Q(model_Q, 
            model_Q_prev, 
            states,
            actions, 
            next_states,
            terminate,
            rewards)       
        
        with tf.GradientTape(persistent=True) as tape:
            
            logits = model_Q(states)
            
            loss = self.loss(logits, target_Q )
        #         *****************************************************

        grads = tape.gradient(loss, model_Q.trainable_weights)
    
        self.param["optimizer_Q"].apply_gradients(zip(grads, model_Q.trainable_weights))            
         
            
        if model_Q_prev is not None:
            # update target network parameters slowly from primary network
            
            for t, e in zip(model_Q_prev.trainable_variables, model_Q.trainable_variables):
                t.assign(
                    tf.add(
                        tf.multiply(
                            t,(1 - self.param["tau"]) 
                        ), 
                        tf.multiply(
                            e , self.param["tau"]  
                        )
                    ) 
                )

        return tf.reduce_mean(loss).numpy()
    
        
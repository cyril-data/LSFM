import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import random
import numpy as np



class Agent:
    
        
    def __init__(self, enviroment, param={}):
        self.param = param
        self._action_size = enviroment.action_space.n
        self._state_dim = enviroment._state_dim
        self.dim_latent = self.param["latent_space"]
        self.hidden_dim = int(self.param["latent_space"] * self.param["hidden_dim_ratio"])
        self.batch_size = self.param["batch_size"]
        
#         LSFM model for construction of a reward-predictive space representation  
#         LSFM must be train on policy independant of state
        inputs = Input(shape=(self._state_dim,), name="input")
        x = layers.Dense(self.hidden_dim, activation='relu', name="hidden_latent")(inputs)
        layer_latent = layers.Dense(self.dim_latent, activation='relu', name="latent")(x)
        layers_ouputs = {}

        x = layers.Dense(self.dim_latent * self._action_size,use_bias=False, name = "Fa")(layer_latent)
        layers_ouputs["Fa"] = layers.Reshape((self._action_size, self.dim_latent))(x)
        layers_ouputs["ra"]  = layers.Dense(self._action_size,use_bias=False, name = "ra")(layer_latent)

        
#         layers_ouputs["dot_reward"] = layers.Dense(1, use_bias=False, name="dot_layer")(layer_latent)
        layers_ouputs["phi"] = layer_latent    
        self.model_LSFM = keras.Model(inputs=inputs, outputs=layers_ouputs, name="model_LSFM")     
        self.model_LSFM_prev = keras.models.clone_model(self.model_LSFM)
      
         # Initialize policy
            
#             policy = random by default
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
        
        return tf.stack(aggr, axis=0)
    
    
    
    def target_psi(self, model, states,actions, next_states,terminate,rewards, filter_done, action_space):

        psi = model(states)["Fa"]
        
        psi_prime = model(next_states)["Fa"]
        
        psi_bar = tf.math.reduce_mean(psi_prime, axis=1 )
        
        phi = model(states)["phi"] 
        
#         if model_prev is None:
#             psi_bar = self.psi_bar(next_states, action_space, model) 
#         else : 
#             psi_bar = self.psi_bar_prev(next_states, action_space, model, model_prev         
        #  filter_done = True : - we don't take into account terminal states : 
        #                       - creation of a filter to determine whitch episode of batch is a terminal state
        #  filter_done = False : we take into account all states
        
        if filter_done  : 
            
            gamma_psibar = tf.multiply(self.param["gamma"],psi_bar)
            
            filter_idxs = tf.dtypes.cast(tf.logical_not(terminate), tf.float32)

            filter_expand  = tf.transpose(tf.tile(
                tf.reshape(filter_idxs, (1,-1) ), [gamma_psibar.shape[1],1]
            ))

            gamma_psibar_filter = tf.math.multiply(filter_expand, gamma_psibar)
            
            y = tf.add(
                phi, 
                gamma_psibar_filter
            )
            
        else : 
            
            y = tf.add(
                phi, 
                tf.multiply(
                        self.param["gamma"],
                        psi_bar
                ) 
            )
            
        return y
    
    def loss(self,y_pred,y_true) : 
#         MSE
#         return keras.losses.mean_squared_error(y_pred,y_true)
#         MAE
        return keras.losses.mean_absolute_error(y_pred,y_true)
            
        
    def loss_N(self,y_pred) : 
        vector_ONE = tf.constant(1., shape=y_pred.shape[0], dtype = y_pred.dtype)
        loss =    tf.pow(tf.subtract(tf.norm(y_pred, axis=1), vector_ONE ), 2.) 
        return loss

#         agent.train_LSFM(
#                         model_LSFM, 
#                         memory, 
#                         model_LSFM_prev if param_agent["double_model"] else None, 
#                         param_agent["filter_done"])

    def M_transi(self, model) : 
        
        for idl, layer in enumerate(model.layers):
            name_a = "Fa"
            if layer.name == name_a :         
                Fa = layer.weights[0]

        Fa_reshape = tf.reshape(Fa, (Fa.shape[0],4,Fa.shape[0]))
        F_bar = tf.math.reduce_mean( Fa_reshape, axis=1 )

        I = tf.eye(F_bar.shape[0], dtype=tf.dtypes.float32)
        gamma = self.param["gamma"]
        
        
        Mlist = []
        
        for a in range(Fa_reshape.shape[1]) :

            MaFbar = tf.multiply(1./gamma , ( Fa_reshape[:,a,:] - I ))

            Ma = tf.matmul( MaFbar , tf.linalg.inv(F_bar)   )

            Mlist.append(Ma)
        
        self.M = tf.stack(Mlist, axis=0)
        
        return self.M
    
    
    def next_states_pred(self, model, states,actions) : 
        
        phi = model(states)["phi"] 

        M = self.M

        aggr = []
        
        
        for k in range(states.shape[0]) : 
            phik = tf.reshape(phi[k], (1, phi.shape[1]))
            Ma=tf.matmul(phik, M[actions[k]])   
            Ma = tf.reshape(Ma, (Ma.shape[1]))

            aggr.append(Ma)
            
        return tf.stack(aggr, axis=0)
        

    def next_phi_pred(self, model, phi,actions) : 

        M = self.M

        aggr = []        
        
        for k in range(phi.shape[0]) : 
            phik = tf.reshape(phi[k], (1, phi.shape[1]))
            Ma=tf.matmul(phik, M[actions[k]])   
            Ma = tf.reshape(Ma, (Ma.shape[1]))

            aggr.append(Ma)
            
        return tf.stack(aggr, axis=0)
        
        
        
    
    def train_LSFM(self, model, memory, filter_done):
        
        
        if memory.num_samples < self.batch_size * 3:
            return [0,0,0,0,0]
        batch = memory.sample(self.batch_size)
        
        action_space = np.arange(self._action_size)
#        tensor conversion
        states = tf.convert_to_tensor(np.array([val[0] for val in batch]))
    
        action_mask = batch[1]
        actions = tf.convert_to_tensor(np.array([val[2] for val in batch]))
        rewards = tf.convert_to_tensor(np.array([val[3] for val in batch]).astype(np.float32))
        next_states = tf.convert_to_tensor(np.array([(np.zeros(self._state_dim)
                                 if val[3] is None else val[4]) for val in batch]))
        terminate = tf.convert_to_tensor(np.array([val[5] for val in batch]))

#         print("states ",states)
#         print("action_mask ",action_mask)
#         print("actions ",actions)
#         print("rewards ",rewards)
#         print("next_states ",next_states)
#         print("terminate ",terminate)
        #         ************* A Calculate Loss on reward and normalisation first  *******************
        # 1) calculate the losses 
            
        target_psi = self.target_psi(model, states,actions, next_states,terminate,rewards, filter_done, action_space)
        
        
        alpha_r = self.param["alpha_r"]
        alpha_N = self.param["alpha_N"]
        alpha_psi = self.param["alpha_psi"]  
        
        with tf.GradientTape() as tape:
            logits_r = self.get_from_actions( states, actions, model, "ra")
            loss_r =  self.loss(logits_r, rewards )

            logits_N = model(states)["phi"]
            loss_N =  self.loss_N(logits_N)

            logits_psi = self.get_from_actions( states, actions, model, "Fa")
            loss_psi = self.loss( logits_psi, target_psi )

            loss_tot = alpha_r * loss_r + alpha_N * loss_N + alpha_psi * loss_psi
        
        # 2) : calculate the gradient   
        grads = tape.gradient(loss_tot, model.trainable_weights) 
        
#         print("grads", grads)

        # 3) : apply the gradient 
        self.param["optimizer_LSFM"].apply_gradients(zip(grads, model.trainable_weights)) 
          
        #         ****************************************************

#         if model_prev is not None:
#             # update model_prev parameters slowly from model (primary network) 
#             for t, e in zip(model_prev.trainable_variables, model.trainable_variables):
#                 t.assign(
#                     tf.add(
#                         tf.multiply(
#                             t,(1 - self.param["tau"]) 
#                         ), 
#                         tf.multiply(
#                             e , self.param["tau"]  
#                         )
#                     ) 
#                 )
        
        return [tf.reduce_mean(loss_tot).numpy(), 
                tf.reduce_mean(loss_r).numpy(), 
                tf.reduce_mean(loss_N).numpy(), 
                tf.reduce_mean(loss_psi).numpy()]

    
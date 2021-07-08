import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import random
import numpy as np
from tensorflow.keras.layers import Lambda



def discretize(r, reward_parser) : 
    k = 0
    while r > sorted(reward_parser)[k] : 
        k+=1
        if k > len(reward_parser)-1 : 
            break
    return k

def discretize_batch(batch, reward_parser)  : 
    return tf.map_fn(lambda x: discretize(x, reward_parser), batch,  fn_output_signature=tf.int32)

    
def my_init(shape,nb_action, dtype=None):
    
    initializer_zero = tf.keras.initializers.Zeros()
    F = tf.Variable(
        initial_value=initializer_zero(shape=shape, dtype=dtype))
    

    Fa_reshape = tf.Variable(tf.reshape(F, (F.shape[0],nb_action,F.shape[0])))

    initializer_Id = tf.keras.initializers.Identity()
    Id = initializer_Id(shape=(shape[0], int(shape[1]/nb_action)))
    
    
    for k in range(nb_action) : 
        Fa_reshape[:,k,:].assign(Id)
        

        
        
    return tf.reshape(Fa_reshape, shape)
    

class My_init:
    
    def __init__(self, shape, nb_action):
        self.shape = shape
        self.nb_action = nb_action
        
    def __call__(self, shape, dtype=None):
        return my_init(self.shape, self.nb_action, dtype=dtype)


class Agent:
    
        
    def __init__(self, enviroment, param={}, save_model_path = None):
        self.param = param
        self._action_size = enviroment._action_dim
        self._state_dim = enviroment._state_dim
        self.dim_latent = self.param["latent_space"]
        self.hidden_dim = int(self.param["latent_space"] * self.param["hidden_dim_ratio"])
        self.batch_size = self.param["batch_size"]
        self.reward_parser = self.param["reward_parser"]
        
        
        
#         LSFM model for construction of a reward-predictive space representation  
#         LSFM must be train on policy independant of state
        layers_inputs = []
        input_state = Input(shape=(self._state_dim,), name="input_state")
#         input_latent = Input(shape=(self.dim_latent,), name="input_latent")
        
        layers_inputs.append(input_state)
#         layers_inputs.append(input_latent)
        
#         layers_inputs["state"] = input_state
#         layers_inputs["latent"] = input_latent
    
        x = layers.Dense(self.hidden_dim, activation='relu', name="hidden_latent")(input_state)

        layer_latent = layers.Dense(self.dim_latent, activation='relu', name="latent")(x)
        layers_ouputs = {}

        # init Identity Fa Matrix          
        shape = (self.dim_latent, self.dim_latent * self._action_size)
        x = layers.Dense(
            self.dim_latent * self._action_size,
            use_bias=False, 
            name = "Fa", 
            kernel_initializer=My_init(shape, self._action_size)
#             kernel_initializer=my_init(shape,self._action_size)
        )(layer_latent)
        
        layers_ouputs["Fa"] = layers.Reshape((self._action_size, self.dim_latent))(x)
        
#         classification on rewards
        x  = layers.Dense(self._action_size * (len(self.reward_parser)+2),use_bias=False, name = "ra")(layer_latent)
        x = layers.Reshape((self._action_size, len(self.reward_parser)+2))(x)
        layers_ouputs["ra"]  = Lambda(lambda x: tf.keras.activations.softmax(x, axis=2))(x)
        
#         out of phi
        layers_ouputs["phi"] = layer_latent    
        
        
        
        
#         ---------------------------------------------------------------------------
#         Ma layers : transition state latent matrices
        initializer_zero = tf.keras.initializers.Zeros()
        
        x_1_stop_grad = Lambda(lambda x: tf.stop_gradient(x),name="Stop_grad_prop")(layer_latent)
        shape = (self.dim_latent, self.dim_latent * self._action_size)
        x = layers.Dense(
            self.dim_latent * self._action_size,
            use_bias=False, 
            name = "Ma", 
            kernel_initializer=initializer_zero
#             kernel_initializer=my_init(shape,self._action_size)
        )(x_1_stop_grad)
        layers_ouputs["Ma"] = layers.Reshape((self._action_size, self.dim_latent))(x)
#         ---------------------------------------------------------------------------
        
    
    
#         Model compilation
        self.model_LSFM = keras.Model(inputs=input_state, outputs=layers_ouputs, name="model_LSFM")  
#         In case we activate the double model as in DQN
        self.model_LSFM_prev = keras.Model(inputs=input_state, outputs=layers_ouputs, name="model_LSFM_prev")   
        
        
        if save_model_path != None : 
            self.model_LSFM.load_weights(save_model_path)
            self.model_LSFM_prev.load_weights(save_model_path)
        
         # Initialize policy
            
            
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
    
    
    
    def target_psi(self, model, states,actions, next_states,terminate,rewards, filter_done):

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

    def loss_mse(self,y_true,y_pred) : 
        return keras.losses.mean_squared_error(y_true,y_pred)
            
        
    def loss_classif(self,y_true,  y_pred) : 
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        return scce(y_true, y_pred)
        
    def loss_N(self,y_pred) : 
        vector_ONE = tf.constant(1., shape=y_pred.shape[0], dtype = y_pred.dtype)
        loss =    tf.pow(tf.subtract(tf.norm(y_pred, axis=1), vector_ONE ), 2.) 
        return loss
        
    def losses_all(self,states, actions, rewards, next_states, target_psi, model) : 
        
        
        alpha_r = self.param["alpha_r"]
        alpha_N = self.param["alpha_N"]
        alpha_psi = self.param["alpha_psi"]  
        alpha_phi = self.param["alpha_phi"]  
        
        
        
        logits_r = self.get_from_actions( states, actions, model, "ra")
        target_r = discretize_batch(rewards, self.reward_parser)
        loss_r =  self.loss_classif(target_r, logits_r)

        logits_N = model(states)["phi"]
        loss_N =  self.loss_N(logits_N)

        logits_psi = self.get_from_actions( states, actions, model, "Fa")
        loss_psi = self.loss_mse( target_psi, logits_psi )

        logits_phi = self.get_from_actions( states, actions, model, "Ma")            
        loss_phi = self.loss_mse(  model(next_states)["phi"]  , logits_phi)

        loss_tot = alpha_r * loss_r + alpha_N * loss_N + alpha_psi * loss_psi + alpha_phi * loss_phi
        
        return loss_tot, loss_r ,  loss_N , loss_psi , loss_phi
    
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
    
#         actions = tf.convert_to_tensor(np.array([val[1] for val in batch]))
#         rewards = tf.convert_to_tensor(np.array([val[2] for val in batch]).astype(np.float32))
#         next_states = tf.convert_to_tensor(np.array([(np.zeros(self._state_dim)
#                                  if val[3] is None else val[3]) for val in batch]))
#         terminate = tf.convert_to_tensor(np.array([val[4] for val in batch]))

        
        #         ************* A Calculate Loss on reward and normalisation first  *******************
        # 1) calculate the losses 
            
        target_psi = self.target_psi(model, states,actions, next_states,terminate,rewards, filter_done)
        
        
        with tf.GradientTape() as tape:
            loss_tot, loss_r ,  loss_N , loss_psi , loss_phi = self.losses_all(
                states, actions, rewards,next_states, target_psi, model)
#             logits_r = self.get_from_actions( states, actions, model, "ra")
#             target_r = discretize_batch(rewards, self.reward_parser)
#             loss_r =  self.loss_classif(target_r, logits_r)

#             logits_N = model(states)["phi"]
#             loss_N =  self.loss_N(logits_N)

#             logits_psi = self.get_from_actions( states, actions, model, "Fa")
#             loss_psi = self.loss_mse( target_psi, logits_psi )
            
#             logits_phi = self.get_from_actions( states, actions, model, "Ma")            
#             loss_phi = self.loss_mse(  model(next_states)["phi"]  , logits_phi)

#             loss_tot = alpha_r * loss_r + alpha_N * loss_N + alpha_psi * loss_psi + alpha_phi * loss_phi
        
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
                tf.reduce_mean(loss_psi).numpy(), 
                tf.reduce_mean(loss_phi).numpy()]



    def M_transi(self, model) : 
        
        for idl, layer in enumerate(model.layers):
            name_a = "Fa"
            if layer.name == name_a :         
                Fa = layer.weights[0]

        Fa_reshape = tf.reshape(Fa, (Fa.shape[0],self._action_size,Fa.shape[0]))
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
        
        
#     def next_rewards( self, phi, actions, model) : 
        
#         x  = layers.Dense(self._action_size * (len(self.reward_parser)+2),use_bias=False, name = "ra")(layer_latent)
#         x = layers.Reshape((self._action_size, len(self.reward_parser)+2))(x)
#         layers_ouputs["ra"]  = Lambda(lambda x: tf.keras.activations.softmax(x, axis=2))(x)
        
        
        
#         for idl, layer in enumerate(model.layers):
#             name_a = "ra"
#             if layer.name == name_a : 
#                 wa = layer.weights[0]
                

        
#         var_all_a = tf.matmul( phi , wa   )
#         aggr = []
#         for k in range(var_all_a.shape[0]) : 
#             aggr.append(var_all_a[k, actions[k]])
        
#         return tf.stack(aggr, axis=0)
    
    
    

    def rewards_latent( self, phi,model) : 
        for idl, layer in enumerate(model.layers):
            name_a = "ra"
            if layer.name == name_a : 
                wa = layer.weights[0]

        var_all_a = tf.matmul( phi , wa   )

        var_all_a_reshape = tf.reshape(
            var_all_a, (var_all_a.shape[0], self._action_size, len(self.reward_parser)+2), name=None
        )

        out = []

        for k in range(var_all_a_reshape.shape[0]) : 

            out.append(Lambda(lambda x: tf.keras.activations.softmax(x, axis=2))(var_all_a_reshape[k]))

        return tf.stack(out, axis=0)
    
    
    def next_rewards( self, phi, actions, model) : 
        rew = self.rewards_latent( phi, model)

        out = []
        for k in range(rew.shape[0]) : 
            out.append(rew[k,actions[k] ])
            
        return tf.stack(out, axis=0)
    

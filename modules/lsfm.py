import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import random
import numpy as np
from tensorflow.keras.layers import Lambda
import matplotlib.pyplot as plt
import math


def discretize(r, reward_parser):
    k = 0
    while r > sorted(reward_parser)[k]:
        k += 1
        if k > len(reward_parser) - 1:
            break
    return k


def discretize_batch(batch, reward_parser):
    return tf.map_fn(
        lambda x: discretize(x, reward_parser), batch, fn_output_signature=tf.int32
    )


def my_init(shape, nb_action, dtype=None):

    initializer_zero = tf.keras.initializers.Zeros()
    F = tf.Variable(initial_value=initializer_zero(shape=shape, dtype=dtype))

    Fa_reshape = tf.Variable(tf.reshape(F, (F.shape[0], nb_action, F.shape[0])))

    initializer_Id = tf.keras.initializers.Identity()
    Id = initializer_Id(shape=(shape[0], int(shape[1] / nb_action)))

    for k in range(nb_action):
        Fa_reshape[:, k, :].assign(Id)

    return tf.reshape(Fa_reshape, shape)


class My_init:
    def __init__(self, shape, nb_action):
        self.shape = shape
        self.nb_action = nb_action

    def __call__(self, shape, dtype=None):
        return my_init(self.shape, self.nb_action, dtype=dtype)


class AgentLSFM:
    def __init__(self, environment, param={}, save_model_path=None):
        self.environment = environment
        self.param = param
        self._action_size = environment._action_dim
        self._state_dim = environment._state_dim
        self.dim_latent = self.param["latent_space"]
        self.hidden_dim = int(
            self.param["latent_space"] * self.param["hidden_dim_ratio"]
        )
        self.batch_size = self.param["batch_size"]
        self.reward_parser = self.param["reward_parser"]
        #         no Eigenoption discovery by default
        self.eigenoption = [-1]
        self.eigenoption_number = self.param["eigenoption_number"]
        self.error_phisp1 = 0.0
        self.discoverNegation = self.param["discoverNegation"]
        self.eigen_exploration = self.param["eigen_exploration"]
        self.negation_vect = False

        self.fig, self.ax = plt.subplots()

        #         LSFM model for construction of a reward-predictive space representation
        #         LSFM must be train on policy independant of state
        layers_inputs = []
        input_state = Input(shape=(self._state_dim,), name="input_state")

        layers_inputs.append(input_state)
        if self.dim_latent == 0:
            layer_latent = input_state
        else:
            if self.hidden_dim:
                x = layers.Dense(
                    self.hidden_dim, activation="relu", name="hidden_latent"
                )(input_state)

                x = layers.Dense(
                    self.hidden_dim, activation="relu", name="hidden_latent2"
                )(x)
            else:
                x = input_state
        layer_latent = layers.Dense(self.dim_latent, activation="relu", name="latent")(
            x
        )

        layers_ouputs = {}

        # init Identity Fa Matrix
        shape = (layer_latent.shape[-1], layer_latent.shape[-1] * self._action_size)
        x = layers.Dense(
            layer_latent.shape[-1] * self._action_size,
            use_bias=False,
            name="Fa",
            kernel_initializer=My_init(shape, self._action_size)
            #             kernel_initializer=my_init(shape,self._action_size)
        )(layer_latent)

        layers_ouputs["Fa"] = layers.Reshape(
            (self._action_size, layer_latent.shape[-1])
        )(x)

        #         classification on rewards
        x = layers.Dense(
            self._action_size * (len(self.reward_parser) + 2), use_bias=False, name="ra"
        )(layer_latent)
        x = layers.Reshape((self._action_size, len(self.reward_parser) + 2))(x)
        #         layers_ouputs["ra"]  = Lambda(lambda x: tf.keras.activations.softmax(x, axis=2))(x)
        layers_ouputs["ra"] = tf.keras.activations.softmax(x, axis=2)

        #         out of phi
        layers_ouputs["phi"] = layer_latent

        #         ---------------------------------------------------------------------------
        #         Ma layers : transition state latent matrices
        initializer_zero = tf.keras.initializers.Zeros()

        #         x_1_stop_grad = Lambda(lambda x: tf.stop_gradient(x),name="Stop_grad_prop")(layer_latent)

        x_1_stop_grad = layer_latent
        tf.stop_gradient(x_1_stop_grad)

        shape = (layer_latent.shape[-1], layer_latent.shape[-1] * self._action_size)
        x = layers.Dense(
            layer_latent.shape[-1] * self._action_size,
            use_bias=False,
            name="Ma",
            kernel_initializer=initializer_zero
            #             kernel_initializer=my_init(shape,self._action_size)
        )(x_1_stop_grad)
        layers_ouputs["Ma"] = layers.Reshape(
            (self._action_size, layer_latent.shape[-1])
        )(x)
        #         ---------------------------------------------------------------------------

        #         Model compilation
        self.model_LSFM = keras.Model(
            inputs=input_state, outputs=layers_ouputs, name="model_LSFM"
        )
        #         In case we activate the double model as in DQN
        self.model_LSFM_prev = keras.Model(
            inputs=input_state, outputs=layers_ouputs, name="model_LSFM_prev"
        )

        if save_model_path != None:
            self.model_LSFM.load_weights(save_model_path)
            self.model_LSFM_prev.load_weights(save_model_path)

        # Initialize policy

        if self.param["policy"]["type"][0] == "eps-greedy":
            self.policy = "eps-greedy"

            if self.param["policy"]["type"][1] == "exponantial":
                self.pi_1 = "exponantial"

                self.epsilon = self.param["policy"][self.policy][self.pi_1]["eps_max"]
                self.eps_max = self.param["policy"][self.policy][self.pi_1]["eps_max"]
                self.eps_min = self.param["policy"][self.policy][self.pi_1]["eps_min"]
                self._lambda = self.param["policy"][self.policy][self.pi_1]["lambda"]

            if self.param["policy"]["type"][1] == "constant":
                self.pi_1 = "constant"
                self.epsilon = self.param["policy"][self.policy][self.pi_1]["eps"]
        else:
            self.policy = self.param["policy"]["type"][0]

        self.optimizer_LSFM = keras.optimizers.Adam()
        self.optimizer_Q: keras.optimizers.Adam()

    def choose_action(self, state, model_Q, steps, possible_action):

        if self.policy == "eps-greedy":
            if self.pi_1 == "exponantial":
                self.epsilon = self.eps_min + (self.eps_max - self.eps_min) * math.exp(
                    -self._lambda * steps
                )

            if random.random() < self.epsilon:
                return random.randint(0, self._action_size - 1), self.epsilon
            else:

                action_greedy, eps = self.choose_action_greedy(
                    state, model_Q, steps, possible_action
                )

                return action_greedy, self.epsilon

    def choose_action_random(self):
        return random.randint(0, self._action_size - 1), self.epsilon

    def choose_action_greedy(self, state, model_Q, steps, possible_action):
        Q_values = model_Q(tf.reshape(state, (1, -1)))
        return tf.math.argmax(Q_values, 0), self.epsilon

    def get_from_actions(self, states, actions, model, var):

        var_all_a = model(states)[var]

        #         aggr = []
        #         for k in range(var_all_a.shape[0]) :
        #             aggr.append(var_all_a[k, actions[k]])

        #         var_actions =  tf.stack(aggr, axis=0)

        var_actions = tf.map_fn(
            fn=lambda t: var_all_a[t[0], t[1]],
            elems=[tf.range(actions.shape[0]), actions],
            dtype=tf.float32,
        )

        return var_actions

    def target_psi(
        self, model, states, actions, next_states, terminate, rewards, filter_done
    ):

        psi = model(states)["Fa"]

        psi_prime = model(next_states)["Fa"]

        psi_bar = tf.math.reduce_mean(psi_prime, axis=1)

        phi = model(states)["phi"]

        #         if model_prev is None:
        #             psi_bar = self.psi_bar(next_states, action_space, model)
        #         else :
        #             psi_bar = self.psi_bar_prev(next_states, action_space, model, model_prev
        #  filter_done = True : - we don't take into account terminal states :
        #                       - creation of a filter to determine whitch episode of batch is a terminal state
        #  filter_done = False : we take into account all states

        if filter_done:

            gamma_psibar = tf.multiply(self.param["gamma"], psi_bar)

            filter_idxs = tf.dtypes.cast(tf.logical_not(terminate), tf.float32)

            filter_expand = tf.transpose(
                tf.tile(tf.reshape(filter_idxs, (1, -1)), [gamma_psibar.shape[1], 1])
            )

            gamma_psibar_filter = tf.math.multiply(filter_expand, gamma_psibar)

            y = tf.add(phi, gamma_psibar_filter)

        else:

            y = tf.add(phi, tf.multiply(self.param["gamma"], psi_bar))

        return y

    def loss_mse(self, y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred)

    def loss_classif(self, y_true, y_pred):
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        return scce(y_true, y_pred)

    def loss_N(self, y_pred):
        vector_ONE = tf.constant(1.0, shape=y_pred.shape[0], dtype=y_pred.dtype)
        loss = tf.pow(tf.subtract(tf.norm(y_pred, axis=1), vector_ONE), 2.0)
        return loss

    def losses_all(self, states, actions, rewards, next_states, target_psi, model):

        alpha_r = self.param["alpha_r"]
        alpha_N = self.param["alpha_N"]
        alpha_psi = self.param["alpha_psi"]
        alpha_phi = self.param["alpha_phi"]

        logits_r = self.get_from_actions(states, actions, model, "ra")
        target_r = discretize_batch(rewards, self.reward_parser)
        loss_r = self.loss_classif(target_r, logits_r)

        logits_N = model(states)["phi"]
        loss_N = self.loss_N(logits_N)

        logits_psi = self.get_from_actions(states, actions, model, "Fa")
        loss_psi = self.loss_mse(target_psi, logits_psi)

        logits_phi = self.get_from_actions(states, actions, model, "Ma")
        loss_phi = self.loss_mse(model(next_states)["phi"], logits_phi)

        loss_tot = (
            alpha_r * loss_r
            + alpha_N * loss_N
            + alpha_psi * loss_psi
            + alpha_phi * loss_phi
        )

        return loss_tot, loss_r, loss_N, loss_psi, loss_phi

    def train_LSFM(self, model, memory, filter_done):
        if memory.num_samples < self.batch_size * 3:
            return [0, 0, 0, 0, 0]
        batch = memory.sample(self.batch_size)

        action_space = tf.range(self._action_size)
        #        tensor conversion
        states = tf.convert_to_tensor(
            np.array([val[0] for val in batch]), dtype=tf.float32
        )

        # states = tf.map_fn(fn=lambda t: tf.convert_to_tensor(
        #     batch)[t, 0], elems=tf.range(len(batch)), dtype=tf.float32)

        action_mask = batch[1]
        actions = tf.convert_to_tensor(
            np.array([val[2] for val in batch]), dtype=tf.int32
        )
        rewards = tf.convert_to_tensor(
            np.array([val[3] for val in batch]).astype(np.float32), dtype=tf.float32
        )
        next_states = tf.convert_to_tensor(
            np.array(
                [
                    (np.zeros(self._state_dim) if val[3] is None else val[4])
                    for val in batch
                ]
            ),
            dtype=tf.float32,
        )
        terminate = tf.convert_to_tensor(
            np.array([val[5] for val in batch]), dtype=tf.bool
        )

        #         actions = tf.convert_to_tensor(np.array([val[1] for val in batch]))
        #         rewards = tf.convert_to_tensor(np.array([val[2] for val in batch]).astype(np.float32))
        #         next_states = tf.convert_to_tensor(np.array([(np.zeros(self._state_dim)
        #                                  if val[3] is None else val[3]) for val in batch]))
        #         terminate = tf.convert_to_tensor(np.array([val[4] for val in batch]))

        #         ************* A Calculate Loss on reward and normalisation first  *******************
        # 1) calculate the losses

        target_psi = self.target_psi(
            model, states, actions, next_states, terminate, rewards, filter_done
        )

        with tf.GradientTape() as tape:
            loss_tot, loss_r, loss_N, loss_psi, loss_phi = self.losses_all(
                states, actions, rewards, next_states, target_psi, model
            )
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
        self.optimizer_LSFM.apply_gradients(zip(grads, model.trainable_weights))

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

        return [
            tf.reduce_mean(loss_tot).numpy(),
            tf.reduce_mean(loss_r).numpy(),
            tf.reduce_mean(loss_N).numpy(),
            tf.reduce_mean(loss_psi).numpy(),
            tf.reduce_mean(loss_phi).numpy(),
        ]

    def next_phi_pred(self, model, phis, actions):

        for idl, layer in enumerate(model.layers):
            name_a = "Ma"
            if layer.name == name_a:
                Ma = layer.weights[0]

        Ma_reshape = tf.reshape(Ma, (Ma.shape[0], self._action_size, Ma.shape[0]))

        agrr = []

        for k, action in enumerate(actions):
            agrr.append(
                tf.reshape(
                    (tf.matmul(tf.reshape(phis[k], (1, -1)), Ma_reshape[:, action, :])),
                    (-1),
                )
            )
        phisp1_pred = tf.stack(agrr, axis=0)

        return phisp1_pred

    def rewards_latent(self, phi, model):
        for idl, layer in enumerate(model.layers):
            name_a = "ra"
            if layer.name == name_a:
                wa = layer.weights[0]

        var_all_a = tf.matmul(phi, wa)

        var_all_a_reshape = tf.reshape(
            var_all_a,
            (var_all_a.shape[0], self._action_size, len(self.reward_parser) + 2),
            name=None,
        )

        out = []

        for k in range(var_all_a_reshape.shape[0]):

            out.append(
                Lambda(lambda x: tf.keras.activations.softmax(x, axis=2))(
                    var_all_a_reshape[k]
                )
            )

        return tf.stack(out, axis=0)

    def next_rewards(self, phi, actions, model):
        rew = self.rewards_latent(phi, model)

        out = []
        for k in range(rew.shape[0]):
            out.append(rew[k, actions[k]])

        return tf.stack(out, axis=0)

    def F_bar(self, model):
        for idl, layer in enumerate(model.layers):
            name_a = "Fa"
            if layer.name == name_a:
                Fa = layer.weights[0]
        Fa_reshape = tf.reshape(Fa, (Fa.shape[0], self._action_size, Fa.shape[0]))
        F_bar = tf.math.reduce_mean(Fa_reshape, axis=1)
        return F_bar

    def eigen_decomp(self, model):
        F_bar = self.F_bar(model)
        d, u, v = tf.linalg.svd(F_bar, full_matrices=True, compute_uv=True)

        self.eigenvalue = d

        self.eigenvect = tf.transpose(u)

        return self.eigenvalue, self.eigenvect

    def eigenpurpose(self, model, phis, phisp1, eigvalue=0, negation_vect=False):
        #         print("eigvalue", eigvalue)

        eigenvect = self.eigenvect[eigvalue]

        if negation_vect:
            eigenvect = -1 * self.eigenvect[eigvalue]
        return tf.matmul(tf.reshape(eigenvect, (1, -1)), tf.transpose(phisp1 - phis))[0]

    def intrinsec_rewards(self, model, phis, actions, eigvalue=0, negation_vect=False):

        phisp1 = self.next_phi_pred(model, phis, actions)
        #         print("phi : ", np.array(phis))
        #         print("next_phi_pred : ", np.array(phisp1))

        return self.eigenpurpose(model, phis, phisp1, eigvalue, negation_vect)

    def eigenoption_fct(self, k, terminate, eigenopt):
        if k in terminate:
            return -1
        else:
            return eigenopt[k]

    def tf_random_choice(self, action_available):
        cat = tf.constant(1.0, shape=(1, action_available.shape[0]), dtype=tf.float32)
        idx = tf.random.categorical(tf.math.log(cat), 1)[0][0]
        return action_available[idx]

    def action_greedy_mask_fct(self, k, terminate, action_available, action_greedy):
        if k in terminate:
            return self.tf_random_choice(action_available)
        else:
            return action_greedy[k]

    def one_step_greedy_eigenpurpose(
        self, model, phis, mask_action, eigvalue=0, negation_vect=False
    ):
        # do a one-step lookahead and act greedily with respect to the eigenpurpose
        # the one-step lookahead is done by the prediction of next latent state
        agrr = []
        switcher = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

        action_available = tf.reshape(tf.where(mask_action), (-1))
        action_available = tf.cast(action_available, tf.int32)

        actions = tf.map_fn(
            fn=lambda t: tf.constant(t, shape=(phis.shape[0])),
            elems=action_available,
            dtype=tf.int32,
        )

        next_possible_phis = tf.map_fn(
            fn=lambda t: self.intrinsec_rewards(
                model, phis, t, eigvalue, negation_vect
            ),
            elems=actions,
            dtype=tf.float32,
        )

        samples = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 1)
        samples = tf.cast(samples, tf.int32)
        samples

        action_greedy = tf.math.argmax(next_possible_phis, 0)
        action_greedy = tf.cast(action_greedy, tf.int32)

        #         print("action_greedy", action_greedy)

        intrisec_reward_max = tf.reduce_max(next_possible_phis, 0)

        #         print("intrisec_reward_max", intrisec_reward_max)

        terminate = tf.where(intrisec_reward_max <= 0.0)[:, 0]
        terminate = tf.cast(terminate, tf.int32)

        # random.choice(action_available)    ==     tf_random_choice(action_available)

        action_greedy_mask = tf.map_fn(
            fn=lambda t: self.action_greedy_mask_fct(
                t, terminate, action_available, action_greedy
            ),
            elems=tf.range(action_greedy.shape[0]),
            dtype=tf.int32,
        )

        #         action_greedy_mask

        self.eigenoption = tf.map_fn(
            fn=lambda t: self.eigenoption_fct(t, terminate, self.eigenoption),
            elems=tf.range(action_greedy.shape[0]),
            dtype=tf.int32,
        )

        #         aggr = []
        #         aggr_eioption = []
        #         for k in range(action_greedy.shape[0]) :
        #             if k in terminate :
        #                 aggr.append(tf_random_choice(action_available))
        #                 aggr_eioption.append(-1)
        #             else :
        #                 aggr.append(action_greedy[k])
        #                 aggr_eioption.append(self.eigenoption[k])

        #         self.eigenoption = tf.stack(aggr_eioption, axis=0)
        #         action_greedy_mask = tf.stack(aggr, axis=0)

        #         print("terminate greedy", np.array(terminate))
        #         print("action_greedy_mask", switcher[np.array(action_greedy_mask)[0]])

        return action_greedy_mask

    def eigenbehavior(self, model, phis, mask_action, steps):

        if self.eigenoption[0] == -1:
            #             choose the random action because eigen value option = -1
            actions = []
            for k in range(phis.shape[0]):
                actions.append(random.choice(np.where(np.array(mask_action) > 0)[0]))

            if self.policy == "eps-greedy":
                if self.param["policy"]["type"][1] == "exponantial":
                    self.eigen_exploration = self.eps_min + (
                        self.eps_max - self.eps_min
                    ) * math.exp(-self._lambda * steps)

            #           Once the random action is taken, choose between a option or random action (1-ei_expl / ei_expl)
            eigenchoice = random.choices(
                [0, 1], weights=(self.eigen_exploration, 1.0 - self.eigen_exploration)
            )[0]
            #           if self.eigen_exploration --> 1  => eigenchoice --> 0

            if eigenchoice == 1:  # next action will be an eigen option
                if (
                    self.eigenoption_number > 1
                ):  # self.eigenoption_number = 0 is to force random action
                    self.eigenoption = tf.reshape(
                        self.tf_random_choice(tf.range(self.eigenoption_number - 1)),
                        (1),
                    )

                if self.discoverNegation:
                    self.negation_vect = self.tf_random_choice(
                        tf.convert_to_tensor([True, False], dtype=tf.bool)
                    )
                    #          take the negative eigen vect orientation if self.negation_vect = True

        else:
            actions = self.one_step_greedy_eigenpurpose(
                model,
                phis,
                mask_action,
                eigvalue=self.eigenoption[0],
                negation_vect=self.negation_vect,
            )
        #             print("action optiondiscovery", np.array(actions))

        return np.array(actions)

    def exploration_online(
        self, model, phis, mask_action, steps, start_eigenoption=300
    ):
        if steps <= start_eigenoption:
            return [random.randint(0, self._action_size - 1)]
        else:
            return self.eigenbehavior(model, phis, mask_action, steps)

    def visu_grid(self, visu_state, grid_size, axs):

        axs.imshow(visu_state, aspect="auto")

        axs.set_xticks(tf.range(-0.5, grid_size - 1, 1), minor=True)
        axs.set_yticks(tf.range(-0.5, grid_size - 1, 1), minor=True)
        axs.grid(which="both", color="w", linestyle="-", linewidth=1.5)

        axs.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def plot_eigenvect(self, model_LSFM, file):

        n_eigen_value = self.eigenoption_number

        ncol = 5
        nrow = int(n_eigen_value / ncol) + 1

        value, vect = self.eigen_decomp(model_LSFM)

        n = self._state_dim
        ngrid = int(n**0.5)

        states = tf.map_fn(
            fn=lambda t: tf.reshape(keras.utils.to_categorical(t, num_classes=n), n),
            elems=tf.range(n),
            dtype=tf.float32,
        )
        phis = tf.map_fn(
            fn=lambda t: model_LSFM(tf.reshape(t, (1, -1)))["phi"],
            elems=states,
            dtype=tf.float32,
        )
        phis = tf.reshape(phis, (phis.shape[0], phis.shape[-1]))

        fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * ncol * nrow / ncol))
        for idx in range(n_eigen_value):

            grid_size = ngrid

            visu_state = tf.reshape(
                tf.matmul(tf.reshape(vect[idx], (1, -1)), tf.transpose(phis))[0],
                (ngrid, ngrid),
            )

            axs[idx // ncol, idx % ncol].imshow(visu_state, aspect="auto")

            axs[idx // ncol, idx % ncol].set_xticks(
                np.arange(-0.5, grid_size - 1, 1), minor=True
            )
            axs[idx // ncol, idx % ncol].set_yticks(
                np.arange(-0.5, grid_size - 1, 1), minor=True
            )
            axs[idx // ncol, idx % ncol].grid(
                which="both", color="w", linestyle="-", linewidth=1.5
            )
            axs[idx // ncol, idx % ncol].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

        #             self.visu_grid(tf.reshape(vect[ idx ], (ngrid, ngrid)) ,ngrid, axs[idx//ncol,idx%ncol])

        fig.savefig(file)

        plt.close()

        return value, vect

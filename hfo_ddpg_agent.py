import os
from copy import deepcopy

from keras.callbacks import History
from rl.callbacks import TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
from rl.core import Agent
from rl.util import *


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class HFODDPGAgent(Agent):
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001, ):
        super(Agent, self).__init__()
        self.processor = None
        self.memory = memory

        self.nb_actions = nb_actions
        self.actor = actor
        self.critic = critic
        self.startE = 1
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects
        self.endE = 0.1
        annealing_steps = 10000
        self.evaluateE = 0
        self.step_drop = (self.startE - self.endE) / annealing_steps
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError(
                    'More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')
        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        combined_inputs = []
        critic_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append(self.actor.output)
            else:
                combined_inputs.append(i)
                critic_inputs.append(i)
        combined_output = self.critic(combined_inputs)

        grads = K.gradients(combined_output, self.actor.trainable_weights)
        grads = [g / float(self.batch_size) for g in grads]  # since TF sums over the batch
        # We now have the gradients (`grads`) of the combined model wrt to the actor's weights and
        # the output (`output`). Compute the necessary updates using a clone of the actor's optimizer.
        clipnorm = getattr(actor_optimizer, 'clipnorm', 0.)
        clipvalue = getattr(actor_optimizer, 'clipvalue', 0.)

        def get_gradients(loss, params):
            # We want to follow the gradient, but the optimizer goes in the opposite direction to
            # minimize loss. Hence the double inversion.
            assert len(grads) == len(params)
            modified_grads = [-g for g in grads]
            if clipnorm > 0.:
                norm = K.sqrt(sum([K.sum(K.square(g)) for g in modified_grads]))
                modified_grads = [optimizers.clip_norm(g, clipnorm, norm) for g in modified_grads]
            if clipvalue > 0.:
                modified_grads = [K.clip(g, -clipvalue, clipvalue) for g in modified_grads]

            return modified_grads

        actor_optimizer.get_gradients = get_gradients
        updates = actor_optimizer.get_updates(self.actor.trainable_weights, self.actor.constraints, None)
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        inputs = self.actor.inputs[:] + critic_inputs
        if self.uses_learning_phase:
            inputs += [K.learning_phase()]
        self.actor_train_fn = K.function(inputs, [self.actor.output], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def forward(self, observation, env):
        # Select an action.
        state = np.reshape(self.memory.get_recent_state(observation), [1, 58])
        action = self.select_action(state)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def select_action(self, state):
        action_arr = self.actor.predict(state)[0]
        dice = np.random.uniform(0, 1)
        if dice < self.epsilon and self.training and self.step>self.nb_steps_warmup_actor:
            print "Random action is taken for exploration, e = " + str(self.epsilon)+'\n'
            new_action_arr = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                              np.random.uniform(0, 100), np.random.uniform(-180, 180),
                              np.random.uniform(-180, 180), np.random.uniform(0, 100),
                              np.random.uniform(-180, 180)]
            action_arr = new_action_arr
        if self.training and self.epsilon >= self.endE:
            self.epsilon -= self.step_drop

        # Take an action and get the current game status
        print action_arr

        return action_arr

    def load_weights(self, file_path):
        filename, extension = os.path.splitext(file_path)
        actor_file_path = filename + '_actor' + extension
        critic_file_path = filename + '_critic' + extension
        self.actor.load_weights(actor_file_path)
        self.critic.load_weights(critic_file_path)
        self.update_target_models_hard()

    def save_weights(self, file_path, overwrite=False):
        filename, extension = os.path.splitext(file_path)
        actor_file_path = filename + '_actor' + extension
        critic_file_path = filename + '_critic' + extension
        self.actor.save_weights(actor_file_path, overwrite=overwrite)
        self.critic.save_weights(critic_file_path, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    def process_state_batch(self, batch):
        batch = np.squeeze(np.array(batch))
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []

            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:] + state0_batch[:]
                else:
                    inputs = [state0_batch, + state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                action_values = self.actor_train_fn(inputs)[0]
                assert action_values.shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        pass

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=2000000,
            nb_max_episode_steps=None, nb_episodes=10000):
        self.training = True
        callbacks = [] if not callbacks else callbacks[:]
        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()
        episode = 0
        self.step = 0
        did_abort = False
        if self.training:
            self.epsilon = self.startE
        else:
            self.epsilon = self.evaluateE
        try:
            while episode < nb_episodes:
                callbacks.on_episode_begin(episode)
                episode_step = 0
                episode_reward = 0.

                # Obtain the initial observation by resetting the environment.
                observation = env.env.getState()
                if self.processor is not None:
                    observation = self.processor.process_observation(observation)
                assert observation is not None

                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation, env)
                reward = 0.
                accumulated_info = {}
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(observation, r, done, info)
                callbacks.on_action_end(action)
                reward += r
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward
                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1
                if done:
                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    if np.mod(episode, 10) == 0 and self.training:
                        self.save_weights(file_path="",overwrite=True)

        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history

from keras import Input, layers
from keras.engine import Model
from keras.layers import LeakyReLU, Dense, initializers
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from Utils import hfoENV

from hfo_ddpg_agent import HFODDPGAgent

batch_size = 32  # batch size for training
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
evaluate_e = 0  # Epsilon used in evaluation
discount_factor = 0.99
annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 10000  # How many episodes of game environment to train network with.
pre_train_steps = 1000  # How many steps of random actions before training begins.
num_players = 1
num_opponents = 0
tau = 0.001  # Tau value used in target network update
num_features = (58 + (num_players - 1) * 8 + num_opponents * 8) * num_players
step_counter = 0
load_model = False  # Load the model
train = True
num_games = 0
relu_neg_slope = 0.01
team_size = 1
enemy_size = 0
action_dim = 8
state_size = 58 + (team_size - 1) * 8 + enemy_size * 8

memory = SequentialMemory(limit=10000, window_length=1)

critic_input_action = Input(shape=[action_dim], name='critic_ain')
critic_input_state = Input(shape=[state_size], name='critic_sin')
critic_input_final = layers.concatenate([critic_input_state, critic_input_action], axis=1, name='critic_in')
dense1 = Dense(1024, kernel_initializer=initializers.glorot_normal(),
               bias_initializer=initializers.glorot_normal(), name='critic_d1')(critic_input_final)
relu1 = LeakyReLU(alpha=relu_neg_slope, name='critic_re1')(dense1)
dense2 = Dense(512, kernel_initializer=initializers.glorot_normal(),
               bias_initializer=initializers.glorot_normal(), name='critic_d2')(relu1)
relu2 = LeakyReLU(alpha=relu_neg_slope, name='critic_re2')(dense2)
dense3 = Dense(256, kernel_initializer=initializers.glorot_normal(),
               bias_initializer=initializers.glorot_normal(), name='critic_d3')(relu2)
relu3 = LeakyReLU(alpha=relu_neg_slope, name='critic_re3')(dense3)
dense4 = Dense(128, kernel_initializer=initializers.glorot_normal(),
               bias_initializer=initializers.glorot_normal(), name='critic_d4')(relu3)
relu4 = LeakyReLU(alpha=relu_neg_slope, name='critic_re4')(dense4)
critic_out = Dense(1, kernel_initializer=initializers.glorot_normal(),
                   bias_initializer=initializers.glorot_normal())(relu4)

critic = Model(inputs=[critic_input_state, critic_input_action], outputs=critic_out)
critic.summary()

actor_input = Input(shape=[state_size], name='actor_in')
dense1 = Dense(1024, kernel_initializer=initializers.glorot_normal(),
               bias_initializer=initializers.glorot_normal(), name='actor_d1')(
    actor_input)
relu1 = LeakyReLU(alpha=relu_neg_slope, name='actor_re1')(dense1)
dense2 = Dense(512, kernel_initializer=initializers.glorot_normal(),
               bias_initializer=initializers.glorot_normal(), name='actor_d2')(
    relu1)
relu2 = LeakyReLU(alpha=relu_neg_slope, name='actor_re2')(dense2)
dense3 = Dense(256, kernel_initializer=initializers.glorot_normal(),
               bias_initializer=initializers.glorot_normal(), name='actor_d3')(
    relu2)
relu3 = LeakyReLU(alpha=relu_neg_slope, name='actor_re3')(dense3)
dense4 = Dense(128, kernel_initializer=initializers.glorot_normal(),
               bias_initializer=initializers.glorot_normal(), name='actor_d4')(
    relu3)
relu4 = LeakyReLU(alpha=relu_neg_slope, name='actor_re4')(dense4)
action_out = Dense(3, kernel_initializer=initializers.glorot_normal(),
                   bias_initializer=initializers.glorot_normal(), name='actor_aout')(
    relu4)
param_out = Dense(5, kernel_initializer=initializers.glorot_normal(),
                  bias_initializer=initializers.glorot_normal(), name='actor_pout')(
    relu4)
actor_out = layers.concatenate([action_out, param_out], axis=1, name='actor_out')
actor = Model(inputs=actor_input, outputs=actor_out)
actor.summary()
'''
hfo = HFOEnvironment()
hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                    '/Users/eclipse/HFO/bin/teams/base/config/formations-dt', 6000,
                    'localhost', 'base_left', False)
hfoENV = hfoENV(hfo)
'''
random_process = OrnsteinUhlenbeckProcess(size=10, theta=.15, mu=0., sigma=.3)
env = hfoENV()
agent = HFODDPGAgent(nb_actions=8, actor=actor, critic=critic, critic_action_input=critic_input_action,
                     memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                     random_process=random_process, gamma=.99, target_model_update=1e-3)

agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=['mse'])
print "Agent Compiled, start training"
agent.fit(env=env, nb_steps=2000000, verbose=1)
agent.save_weights('ddpg_{}_weights.h5f'.format('hfo'), overwrite=True)

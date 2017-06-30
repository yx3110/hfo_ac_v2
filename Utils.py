import signal
import subprocess
import time

import hfo_py
from gym import spaces
from hfo import *
from rl.core import Env

max_turn_angle = 180
min_turn_angle = -180
max_power = 100
min_power = 0
min_act_val = -1
max_act_val = 1


class hfoENV(Env):
    def __init__(self):
        super(hfoENV, self)
        self.viewer = None
        self.server_process = None
        self.server_port = None
        self.hfo_path = hfo_py.get_hfo_path()
        self.configure()
        self.env = hfo_py.HFOEnvironment()
        self.env.connectToServer(config_dir=hfo_py.get_config_path())
        self.game_info = GameInfo(1)
        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(low=0, high=100, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1),
                                          spaces.Box(low=0, high=100, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1)))

        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.env.getStateSize()))

    def close(self):
        self.env.act(hfo_py.QUIT)
        self.env.step()
        os.kill(self.server_process.pid, signal.SIGINT)
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)

    def configure(self, *args, **kwargs):
        self._start_hfo_server()

    def reset(self):
        while self.game_info.status == hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        while self.game_info.status != hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        self.game_info.reset()
        return self.env.getState()

    def seed(self, seed=None):
        return seed

    def step(self, action):
        action = self.get_action(action)
        return self.take_action(action)

    def render(self, mode='human', close=False):

        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()

    def _start_viewer(self):
        """
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        """
        cmd = hfo_py.get_viewer_path() +\
              " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

    def _start_hfo_server(self, frames_per_trial=100,
                          untouched_time=100, offense_agents=1,
                          defense_agents=0, offense_npcs=0,
                          defense_npcs=0, sync_mode=False, port=6000,
                          offense_on_ball=0, fullstate=True, seed=-1,
                          ball_x_min=0.0, ball_x_max=0.2,
                          verbose=False, log_game=False,
                          log_dir="log"):
        """
        Starts the Half-Field-Offense server.
        frames_per_trial: Episodes end after this many steps.
        untouched_time: Episodes end if the ball is untouched for this many steps.
        offense_agents: Number of user-controlled offensive players.
        defense_agents: Number of user-controlled defenders.
        offense_npcs: Number of offensive bots.
        defense_npcs: Number of defense bots.
        sync_mode: Disabling sync mode runs server in real time (SLOW!).
        port: Port to start the server on.
        offense_on_ball: Player to give the ball to at beginning of episode.
        fullstate: Enable noise-free perception.
        seed: Seed the starting positions of the players and ball.
        ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
        verbose: Verbose server messages.
        log_game: Enable game logging. Logs can be used for replay + visualization.
        log_dir: Directory to place game logs (*.rcg).
        """
        self.server_port = port
        cmd = self.hfo_path + \
              " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i" \
              " --defense-agents %i --offense-npcs %i --defense-npcs %i" \
              " --port %i --offense-on-ball %i --seed %i --ball-x-min %f" \
              " --ball-x-max %f --log-dir %s" \
              % (frames_per_trial, untouched_time, offense_agents,
                 defense_agents, offense_npcs, defense_npcs, port,
                 offense_on_ball, seed, ball_x_min, ball_x_max,
                 log_dir)
        if not sync_mode: cmd += " --no-sync"
        if fullstate:     cmd += " --fullstate"
        if verbose:       cmd += " --verbose"
        if not log_game:  cmd += " --no-logging"
        print('Starting server with command: %s' % cmd)
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        time.sleep(5)  # Wait for server to startup before connecting a player

    def take_action(self, action):
        action_type = ACTION_LOOKUP[action[0]]
        if action_type == hfo_py.DASH:
            self.env.act(action_type, action[1], action[2])
        elif action_type == hfo_py.TURN:
            self.env.act(action_type, action[3])
        elif action_type == hfo_py.KICK:
            self.env.act(action_type, action[4], action[5])
        else:
            print('Unrecognized action %d' % action_type)
            self.env.act(hfo_py.NOOP)

        return self.game_info.update(self.env)

    def get_action(self, action_arr):
        res = []
        res_action = 0

        action_arr_copy = []
        for i in xrange(len(action_arr)):
            action_arr_copy.append(action_arr[i])
        cur_max = action_arr_copy[0]
        action_arr_copy[2] = -99999
        for i in xrange(0, 2):
            if action_arr_copy[i] >= cur_max:
                res_action = i

        res.append(res_action)

        for i in range(3, len(action_arr)):
            res.append(action_arr[i])

        print res

        return res


def bound(grad, param, max_val, min_val):
    if grad > 0:
        return grad * np.divide(max_val - param, max_val - min_val)
    else:
        return grad * np.divide(param - min_val, max_val - min_val)


def bound_grads(cur_grads, cur_actions, index):
    if 0 <= index < 4:
        cur_grads[index] = bound(cur_grads[index], cur_actions[index], max_act_val, min_act_val)
    if index == 4 or index == 8:
        cur_grads[index] = bound(cur_grads[index], cur_actions[index], max_power, min_power)
    elif index == 5 or index == 6 or index == 7 or index == 9:
        cur_grads[index] = bound(cur_grads[index], cur_actions[index], max_turn_angle, min_turn_angle)


kPassVelThreshold = -.5


class GameInfo:
    def __init__(self, unum):
        self.got_kickable_reward = False
        self.prev_ball_prox = 0
        self.ball_prox_delta = 0
        self.prev_kickable = 0
        self.kickable_delta = 0
        self.prev_ball_dist_goal = 0
        self.ball_dist_goal_delta = 0
        self.steps = 0
        self.total_reward = 0
        self.extrinsic_reward = 0
        self.status = IN_GAME
        self.episode_over = False
        self.our_unum = unum
        self.prev_player_on_ball = 0
        self.player_on_ball = 0
        self.pass_active = False

    def reset(self):
        self.got_kickable_reward = False
        self.prev_ball_prox = 0
        self.ball_prox_delta = 0
        self.prev_kickable = 0
        self.kickable_delta = 0
        self.prev_ball_dist_goal = 0
        self.ball_dist_goal_delta = 0
        self.steps = 0
        self.total_reward = 0
        self.extrinsic_reward = 0
        self.status = IN_GAME
        self.episode_over = False
        self.prev_player_on_ball = 0
        self.player_on_ball = 0
        self.pass_active = False

    def update(self, hfo_env):
        self.status = hfo_env.step()
        if self.status != IN_GAME:
            self.episode_over = True
        cur_obs = hfo_env.getState()
        ball_prox = cur_obs[53]
        goal_prox = cur_obs[15]
        ball_dist = 1.0 - ball_prox
        goal_dist = 1.0 - goal_prox
        kickable = cur_obs[12]
        ball_ang_sin_rad = cur_obs[51]
        ball_ang_cos_rad = cur_obs[52]
        ball_ang_rad = np.arccos(ball_ang_cos_rad)
        if ball_ang_sin_rad < 0:
            ball_ang_rad *= -1.
        goal_ang_sin_rad = cur_obs[13]
        goal_ang_cos_rad = cur_obs[14]
        goal_ang_rad = np.arccos(goal_ang_cos_rad)
        if goal_ang_sin_rad < 0:
            goal_ang_rad *= -1.
        alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
        ball_dist_goal = np.sqrt(ball_dist * ball_dist + goal_dist * goal_dist - 2. * ball_dist * goal_dist *
                                 np.cos(alpha))

        ball_vel_valid = cur_obs[54]
        ball_vel = cur_obs[55]
        if ball_vel_valid and ball_vel > kPassVelThreshold:
            self.pass_active = True

        if self.steps > 0:
            self.ball_prox_delta = ball_prox - self.prev_ball_prox
            self.kickable_delta = kickable - self.prev_kickable
            self.ball_dist_goal_delta = ball_dist_goal - self.prev_ball_dist_goal
        self.prev_ball_prox = ball_prox
        self.prev_kickable = kickable
        self.prev_ball_dist_goal = ball_dist_goal
        if self.episode_over:
            self.ball_prox_delta = 0
            self.kickable_delta = 0
            self.ball_dist_goal_delta = 0
        self.prev_player_on_ball = self.player_on_ball
        self.player_on_ball = hfo_env.playerOnBall()
        self.steps += 1
        return cur_obs, self.get_reward(), self.episode_over, None

    def get_reward(self):
        res = 0
        res += self.move_to_ball_reward()
        res += self.kick_reward() * 3
        EOT_reward = self.EOT_reward()
        res += EOT_reward
        self.extrinsic_reward += EOT_reward
        self.total_reward += res
        return res

    def move_to_ball_reward(self):
        reward = 0
        if self.player_on_ball.unum < 0 or self.player_on_ball.unum == self.our_unum:
            reward += self.ball_prox_delta
        if self.kickable_delta >= 1 and (not self.got_kickable_reward):
            reward += 1.0
            self.got_kickable_reward = True
        return reward

    def kick_reward(self):
        if self.player_on_ball.unum == self.our_unum:
            return -self.ball_dist_goal_delta
        elif self.got_kickable_reward:
            return 0.2 * -self.ball_dist_goal_delta
        return 0

    def EOT_reward(self):
        if self.status == GOAL:
            if self.player_on_ball.unum == self.our_unum:
                return 5
            else:
                return 1
        else:
            return 0


ACTION_LOOKUP = {
    0: hfo_py.DASH,
    1: hfo_py.TURN,
    2: hfo_py.KICK,
    3: hfo_py.TACKLE,  # Used on defense to slide tackle the ball
    4: hfo_py.CATCH,  # Used only by goalie to catch the ball
}

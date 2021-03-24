import tensorflow as tf
import tf_agents
import env
import time
import logging
import random

from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tensorflow.keras import layers, initializers, optimizers

logger = logging.getLogger(__name__)


class PathFinder:
    def __init__(self,
                 environment,
                 batch_size,
                 learning_rate=0.001):
        environment: suite_gym.py_environment.PyEnvironment
        self._env = environment
        self._batch_size = batch_size

        self._num_actions = self._env.action_spec().maximum - self._env.action_spec().minimum + 1
        self._optimizer = optimizers.Adam(learning_rate)
        self._train_step_counter = tf.Variable(0)

        self._dqn = self._build_dqn()
        self._agent = tf_agents.agents.dqn.dqn_agent.DqnAgent(
            time_step_spec=self._env.time_step_spec(),
            action_spec=self._env.action_spec(),
            q_network=self._dqn,
            optimizer=self._optimizer,
            td_errors_loss_fn=tf_agents.utils.common.element_wise_squared_loss,
            train_step_counter=self._train_step_counter
        )
        self._random_policy = self.get_random_policy()
        self._reply_buffer = self._build_reply_buffer()
        self._reply_buffer_dataset = self._reply_buffer.as_dataset(sample_batch_size=self._batch_size,
                                                                   num_steps=2,
                                                                   num_parallel_calls=1).prefetch(1)

    def _build_dqn(self):
        units = [100, 50]
        dqn_layers = []
        for u in units:
            dqn_layers.append(layers.Dense(units=u, activation="relu"))
        dqn_layers.append(tf.keras.layers.Dense(self._num_actions, activation=None))
        dqn_net = sequential.Sequential(dqn_layers)

        return dqn_net

    def _build_reply_buffer(self):
        logger.debug("Env.batch_size: %s", self._env.batch_size)
        return tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self._agent.collect_data_spec,
            batch_size=self._env.batch_size,
            max_length=10000
        )

    def collect_data(self, steps):
        for _ in range(steps):
            time_step = self._env.current_time_step()
            action = self._agent.policy.action(time_step)
            next_time_step = self._env.step(action)
            # self._env.render()
            traj = tf_agents.trajectories.trajectory.from_transition(time_step, action, next_time_step)

            self._reply_buffer.add_batch(traj)

    def train(self, iterations):
        self._agent.train_step_counter.assign(0)
        dataset_iterator = iter(self._reply_buffer_dataset)
        self.collect_data(100)
        for i in range(iterations):
            self.collect_data(1)
            exp, info = next(dataset_iterator)
            train_loss = self._agent.train(exp).loss
            step = self._agent.train_step_counter.numpy()
            logger.debug("Step %s, loss = %s", step, train_loss)

    def metric(self, policy: tf_agents.policies.tf_policy.TFPolicy, episodes=20):
        total_return = 0.0
        for _ in range(episodes):
            stat = self._env.reset()
            while not stat.is_last():
                action = policy.action(stat)
                stat = self._env.step(action)
                total_return += stat.reward.numpy()
                # logger.debug("[Metric] action: %s", action.action.numpy())

        return total_return / episodes

    def get_random_policy(self):
        return tf_agents.policies.random_tf_policy.RandomTFPolicy(time_step_spec=self._env.time_step_spec(),
                                                                  action_spec=self._env.action_spec())

    def demo(self):
        policy = self._agent.policy
        stat = self._env.reset()
        while True:
            img = self._env.render()
            stat = self._env.step(policy.action(stat))
            logger.debug("observation: %s | reward: %s", stat.observation.numpy(), stat.reward.numpy())
            time.sleep(0.2)


def simple_test_demo():
    simple_py_env = suite_gym.load("SimpleTestEnv-v0")
    step_spec = simple_py_env.reset()
    simple_py_env.render()

    logger.debug("Reset return time_step_spec: %s", step_spec)
    logger.debug("Observation: %s", simple_py_env.time_step_spec().observation)
    logger.debug("Reward: %s", simple_py_env.time_step_spec().reward)
    logger.debug("Action: %s", simple_py_env.action_spec())

    train_env = tf_py_environment.TFPyEnvironment(simple_py_env)
    inst = PathFinder(train_env, 64)
    time_step = train_env.reset()
    print(time_step)
    print(inst.get_random_policy().action(time_step))
    print(inst.metric(inst.get_random_policy()))
    inst.train(5000)
    inst.demo()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    simple_test_demo()

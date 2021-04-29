import tensorflow as tf
import tf_agents
import time
import logging
import os

from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tensorflow.keras import layers, optimizers

logger = logging.getLogger(__name__)


class PathFinder:
    class Config:
        def __init__(self,
                     train_env: suite_gym.py_environment.PyEnvironment,
                     eval_env: suite_gym.py_environment.PyEnvironment,
                     batch_size: int,
                     log_dir: str,
                     data_collect_interval: int = 100,
                     data_collect_num: int = 200,
                     metric_interval: int = 500,
                     learning_rate: float = 0.001):
            self.train_env: tf_py_environment.TFPyEnvironment = tf_py_environment.TFPyEnvironment(train_env)
            self.eval_env: tf_py_environment.TFPyEnvironment = tf_py_environment.TFPyEnvironment(eval_env)
            self.batch_size = batch_size
            self.lr = learning_rate
            self.log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d-%H%M%S", time.localtime()))
            self.data_collection_interval = data_collect_interval
            self.data_collect_num = data_collect_num
            self.metric_interval = metric_interval

    def __init__(self, cfg: Config):
        self._cfg = cfg

        self._tf_board = tf.summary.create_file_writer(cfg.log_dir)
        self._num_actions = cfg.train_env.action_spec().maximum - cfg.train_env.action_spec().minimum + 1
        self._optimizer = optimizers.Adam(cfg.lr)
        self._train_step_counter = tf.Variable(0, dtype=tf.int64)

        self._dqn = self._build_dqn()
        self._agent = tf_agents.agents.dqn.dqn_agent.DqnAgent(
            time_step_spec=self._cfg.train_env.time_step_spec(),
            action_spec=self._cfg.train_env.action_spec(),
            q_network=self._dqn,
            optimizer=self._optimizer,
            td_errors_loss_fn=tf_agents.utils.common.element_wise_squared_loss,
            train_step_counter=self._train_step_counter
        )
        self._agent.initialize()
        self._random_policy = self.get_random_policy()
        self._reply_buffer = self._build_reply_buffer()
        self._reply_buffer_dataset = self._reply_buffer.as_dataset(sample_batch_size=cfg.batch_size,
                                                                   num_steps=2,
                                                                   num_parallel_calls=2).prefetch(100)

    def _build_dqn(self):
        units = [100, 100]
        dqn_layers = []
        for u in units:
            dqn_layers.append(layers.Dense(units=u, activation="relu"))
        dqn_layers.append(tf.keras.layers.Dense(self._num_actions, activation=None))
        dqn_net = sequential.Sequential(dqn_layers)

        return dqn_net

    def _build_reply_buffer(self):
        logger.debug("Env.batch_size: %s", self._cfg.train_env.batch_size)
        return tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self._agent.collect_data_spec,
            batch_size=self._cfg.train_env.batch_size,
            max_length=20000
        )

    def _collect_data(self, steps):
        for _ in range(steps):
            time_step = self._cfg.train_env.current_time_step()
            action = self._agent.collect_policy.action(time_step)
            next_time_step = self._cfg.train_env.step(action)
            traj = tf_agents.trajectories.trajectory.from_transition(time_step, action, next_time_step)

            self._reply_buffer.add_batch(traj)

    def train(self, iterations):
        self._agent.train_step_counter.assign(0)
        dataset_iterator = iter(self._reply_buffer_dataset)
        # collect some data before training
        self._collect_data(self._cfg.batch_size * 10)
        for i in range(iterations):
            with self._tf_board.as_default(step=i):  # write summary
                if i % self._cfg.data_collection_interval == 0:
                    logger.debug("Collect %s sample", self._cfg.data_collect_num)
                    self._collect_data(self._cfg.data_collect_num)

                if i % self._cfg.metric_interval == 0:
                    avg_reward = self._metric(episodes=100)
                    logger.debug("Average reward: %s", avg_reward)
                    tf.summary.scalar("train/avg reward", avg_reward)

                exp, info = next(dataset_iterator)
                train_loss = self._agent.train(exp).loss
                step = self._agent.train_step_counter.numpy()
                logger.debug("Step %s, loss = %s", step, train_loss)
                tf.summary.scalar("train/loss", train_loss)

        with self._tf_board.as_default(step=iterations):
            avg_reward = self._metric(episodes=100)
            logger.debug("Average reward: %s", avg_reward)
            tf.summary.scalar("train/avg reward", avg_reward)

    def _metric(self, episodes: int) -> float:
        logger.debug("Compute average reward for %s episodes", episodes)

        total_return = 0.0
        policy = self._agent.policy
        for _ in range(episodes):
            stat = self._cfg.train_env.reset()
            while not stat.is_last():
                action = policy.action(stat)
                stat = self._cfg.train_env.step(action)
                total_return += stat.reward.numpy()[0]

        return total_return / episodes

    def get_random_policy(self):
        return tf_agents.policies.random_tf_policy.RandomTFPolicy(time_step_spec=self._cfg.train_env.time_step_spec(),
                                                                  action_spec=self._cfg.train_env.action_spec())

    def visualize_demo(self):
        policy = self._agent.policy
        eval_time_step = self._cfg.eval_env.reset()
        self._cfg.eval_env.render()
        while not eval_time_step.is_last():
            self._cfg.eval_env.render()
            eval_time_step = self._cfg.eval_env.step(policy.action(eval_time_step))
            # logger.debug("reward: %s", eval_time_step.reward.numpy())

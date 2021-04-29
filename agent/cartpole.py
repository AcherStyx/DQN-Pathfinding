import tensorflow as tf
import cv2

from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from tf_agents.environments import suite_gym
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

gym_env = suite_gym.load('CartPole-v0')

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

train_env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))
eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))

print("observation and reward: ", train_env.time_step_spec())
print("action: ", train_env.action_spec())

# define DQN
action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
q_net = sequential.Sequential([
    Dense(100, activation="relu", kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal')),
    Dense(50, activation="relu", kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal')),
    Dense(num_actions, activation=None, kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
          bias_initializer=tf.keras.initializers.Constant(-0.2))])
# create agent
optimizer = optimizers.Adam(0.001)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(time_step_spec=train_env.time_step_spec(),
                           action_spec=train_env.action_spec(),
                           q_network=q_net,
                           optimizer=optimizer,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           train_step_counter=train_step_counter)
agent.initialize()

# test on a random policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


print("evaluate random policy (baseline)", compute_avg_return(eval_env, random_policy, 100))

# buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=1000)


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


collect_data(train_env, random_policy, replay_buffer, 5000)  # collect initial data

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=2).prefetch(3)
iterator = iter(dataset)

# train
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# # Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, 100)
returns = [avg_return]

summary_writer = tf.summary.create_file_writer("../log/")
print("start train")

for i in range(20000):
    collect_data(train_env, agent.collect_policy, replay_buffer, 100)

    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % 100 == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % 1000 == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, 100)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
        tf.summary.scalar("reward", avg_return, step=i)
        # visualize
        trained_policy = agent.policy
        while True:
            time_step = eval_env.reset()
            eval_env.render()

            while not time_step.is_last:
                time_step = eval_env.step(trained_policy.action(time_step))
                eval_env.render()
                ch = cv2.waitKey(1)
                if ch == 'q':
                    break
            ch = cv2.waitKey(1)
            if ch == 'q':
                break

    with summary_writer.as_default():
        tf.summary.scalar("loss", train_loss, step=i)

# visualize
trained_policy = agent.policy
while True:
    eval_time_step = eval_env.reset()
    eval_env.render()

    while not eval_time_step.is_last:
        eval_time_step = eval_env.step(trained_policy.action(eval_time_step))
        eval_env.render()
        ch = cv2.waitKey(1)
        if ch == 'q':
            break
    ch = cv2.waitKey(1)
    if ch == 'q':
        break

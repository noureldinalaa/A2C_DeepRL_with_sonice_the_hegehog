"""
this model file contains :
* model object (train function)
* runner object (generates mini batches of experiences)
* learn function : handle A2C process

"""


import os
import time
import numpy as np
import os.path as osp
import tensorflow.compat.v1 as tf
from baselines import logger
import cv2

import matplotlib.pyplot as plt

#calculate cross entropy
from baselines.a2c.utils import cat_entropy
from utilities import make_path, find_trainable_variables, discount_with_dones

from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner

class Model(object):
  """
  We use this object to :
  __init__:
  - Creates the step_model(this generates experiences as experience (i.e experience batch  from environments))
  - Creates the train_model (that trains the experiences)
  train():
  - Make the training part (feedforward and retropropagation of gradients)
  save/load():
  - Save load the model
  """

  """
  
                      Train_Model ---------------->
                           ^                      |
                           |                      |
                           |(experience_batch)    | (update step_model)
                           |                      |
                      Step_Model<------------------ 
                      ^   ^  ^  ^
                      |   |  |  |
                    Env1  | Env3|
                         Env2   |
                               Env4

* Creates a vector of n environments using the multiprocessing library
* Creates a runner object that handles the different environments, executing in parallel.
* Has two versions of the network:
    - step_model: that generates experiences from environments
    - train_model: that trains the experiences.
When the runner takes a step (single step model), this performs a step for each of the n environments. This outputs a batch of experience.
Then we compute the gradient all at once using train_model and our batch of experience.

Finally, we update the step model with the new weights.
  """

  def __init__(self,policy,ob_space,action_space,nenvs,nsteps,ent_coef,vf_coef,max_grad_norm):


    sess = tf.get_default_session()

    #here we create the place holders
    actions_ = tf.placeholder(tf.int32, [None], name="actions_")
    advantages_ = tf.placeholder(tf.float32, [None], name="advantages_")
    rewards_ = tf.placeholder(tf.float32, [None], name="rewards_")
    lr_ = tf.placeholder(tf.float32, name="learning_rate_")


    """
    Reminder for myself:
    
    As Value method have high variability we use advantages action 
    Advantages action : the advantage of taking that action at that state
    (how much better is to take this action versus all other possible actions at that state).
    A(s,a) = Q(s,a) -V(s)
     * If A(s,a) > 0: our gradient is pushed in that direction.
     * If A(s,a) < 0 (our action does worse than the average value of that state) our gradient is pushed in the opposite direction.
    """
    # Here we create our two models:
    # step_model that is used for sampling
    # step model takes one step time for each environment
    #the 1 here is number of steps
    step_model = policy(sess, ob_space, action_space, nenvs, 1, reuse=False)


    #train model for traing
    # nenvs*nsteps will give me the whole training steps because we have many environments

    train_model = policy(sess, ob_space, action_space, nenvs * nsteps, nsteps, reuse=True)
    """
    Calculate the loss
    Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
    """
    # Policy loss
    # Output -log(pi)
    neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=actions_)

    # 1/n * sum A(si, ai) * -logpi(ai|si)
    pg_loss = tf.reduce_mean(advantages_ * neglogpac)

    #value loss  1/2 SUM [R - V(s)]^2
    vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf),rewards_))

    # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
    entropy = tf.reduce_mean(train_model.pd.entropy())

    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

    # Update parameters using loss
    # 1. Get the model parameters
    params = find_trainable_variables("model")

    # 2. Calculate the gradients
    grads = tf.gradients(loss, params)

    if max_grad_norm is not None:

        # Clip the gradients (normalize)
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

    grads = list(zip(grads, params))
    # zip aggregate each gradient with parameters associated
    # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

    # 3. Build our trainer
    trainer = tf.train.RMSPropOptimizer(learning_rate=lr_, decay=0.99, epsilon=1e-5)

    # 4. Backpropagation
    _train = trainer.apply_gradients(grads)

    def train (states_in, actions, returns, values, lr):
      # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
      # Returns = R + yV(s')
      advantages = returns - values

      # We create the feed dictionary

      td_map = {train_model.inputs_: states_in,
                actions_: actions,
                advantages_: advantages,  # Use to calculate our policy loss
                rewards_: returns,  # Use as a bootstrap for real value
                lr_: lr}

      policy_loss, value_loss, policy_entropy, _ = sess.run([pg_loss, vf_loss, entropy, _train], td_map)
      return policy_loss, value_loss, policy_entropy

      def save(save_path):
        """
        Save the model
        :param save_path:
        :return:
        """

        saver = tf.train.Saver()
        saver.save(sess, save_path)

      def load(load_path):
        """
        load the model
        :param load_path:
        :return:
        """
        saver = tf.train.Saver()
        print('Loading' + load_path)
        saver.restore(sess, load_path)

      self.train = train
      self.train_model = train_model
      self.step_model = step_model
      self.step = step_model.step
      self.value = step_model.value
      self.initial_state = step_model.initial_state
      self.save = save
      self.load = load
      tf.global_variables_initializer().run(session=sess)

class Runner(AbstractEnvRunner):

  """
  We use this object to make a mini batch of experiences
  __init__:
  - Initialize the runner
  run():
  - Make a mini batch
  """

  def __init__(self, env, model, nsteps, total_timesteps, gamma, lam):
    super().__init__(env=env, model=model, nsteps=nsteps)

    #Discount rate
    self.gamma = gamma

    # Lambda used in GAE (General Advantage Estimation)
    self.lam = lam

    #total timesteps taken
    self.total_timesteps = total_timesteps

  def run(self):
    # Here, we init the lists that will contain the mb of experiences
    mb_obs, mb_actions, mb_rewards, mb_values, mb_dones = [], [], [], [], []

    # For n in range of steps
    for n in range(self.nsteps):
      # Given observations, take action and value (V(s))
      # we already have self.obs because AbstractEnvRunner run self.obs[:] = env.reset()
      actions, values = self.model.step(self.obs, self.dones)

      print("actions runner runner", actions)

      # Append the observations into the mb
      mb_obs.append(np.copy(self.obs))  # obs len nenvs (1 step per env)

      # Append the actions taken into the mb
      mb_actions.append(actions)

      # Append the values calculated into the mb
      mb_values.append(values)

      # Append the dones situations into the mb
      mb_dones.append(self.dones)

      # Take actions in env and look the results
      self.obs[:], rewards, self.dones, _ = self.env.step(actions)

      mb_rewards.append(rewards)

    #batch of steps to batch of rollouts
    #we will put them in numpy array so it is easier to feed the network
    mb_obs = np.asarray(mb_obs, dtype=np.uint8)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_actions = np.asarray(mb_actions, dtype=np.int32)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    last_values = self.model.value(self.obs)

    ### GENERALIZED ADVANTAGE ESTIMATION
    # discount/bootstrap off value fn
    # We create mb_returns and mb_advantages
    # mb_returns will contain Advantage + value

    mb_returns = np.zeros_like(mb_rewards)
    mb_advantages = np.zeros_like(mb_rewards)

    lastgaelam = 0

    # from last step to first step
    for t in reversed(range(self.nsteps)):
      # IF t == before last step
      if t == self.nsteps - 1 :
        # If a state is done , nextnonterminal = 0 (this means no other states left)
        # In fact nextnonterminal allows us to do that logic

        # if done (so nextnonterminal = 0):
        #   delta = R - V(s) (because self.gamma * nextvalues * nextnonterminal = 0)
        # else (not done)
            # delta = R + gamma * V(st+1)

        nextnonterminal = 1.0 - self.dones

        # V(t+1)
        nextvalues = last_values
      else:
        nextnonterminal = 1.0 - mb_dones[t+1]

        nextvalues = mb_values[t+1]

      # Delta = R(st) + gamma * V(t+1) * nextnonterminal  - V(st)
      delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]

      #  Advantage = delta + gamma *  λ (lambda) * nextnonterminal  * lastgaelam
      mb_advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

    # Returns
    mb_returns = mb_advantages + mb_values

    return map(sf01, (mb_obs,mb_actions, mb_returns, mb_values))


def sf01 (arr):
  """
  swap and then flatten axes 0 and 1
  :param arr:
  :return:
  """
  s = arr.shape
  return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def learn(policy,
            env,
            nsteps,
            total_timesteps,
            gamma,
            lam,
            vf_coef,
            ent_coef,
            lr,
            max_grad_norm,
            log_interval):


  # number of optimization epochs
  noptepochs = 4
  nminibatches = 8

  # Get the nb of env
  nenvs  = env.num_envs

  # Get state_space and action_space
  ob_space = env.observation_space
  ac_space = env.action_space

  # calculate the batch size

  batch_size = nenvs * nsteps  # For instance if we take 5 steps and we have 5 environments batch_size = 25

  batch_train_size = batch_size // nminibatches

  assert batch_size % nminibatches == 0


  # Instantiate the model object (that creates step_model and train_model)
  model = Model(policy=policy,
              ob_space=ob_space,
              action_space=ac_space,
              nenvs=nenvs,
              nsteps=nsteps,
              ent_coef=ent_coef,
              vf_coef=vf_coef,
              max_grad_norm=max_grad_norm)
  # Load the model
  # If you want to continue training
  load_path = "./models/260/model.ckpt"
  model.load(load_path)

  # Instantiate the runner object
  runner = Runner(env, model, nsteps=nsteps, total_timesteps=total_timesteps, gamma=gamma, lam=lam)


  # start total timer
  tfirststart = time.time()

  for update in range(1, total_timesteps // batch_size + 1):
    # Start timer
    tstart = time.time()

    # Get minibatch
    obs, actions, returns, values = runner.run()

    # Here what we're going to do is for each minibatch calculate the loss and append it.
    mb_losses = []
    total_batches_train = 0

    # Index of each element of batch_size
    # Create the indices array
    indices = np.arange(batch_size)

    for _ in range(noptepochs):
      # Randomize the indexes
      np.random.shuffle(indices)

      # 0 to batch_size with batch_train_size step
      for start in range(0, batch_size, batch_train_size):
        end = start + batch_train_size
        mbinds = indices[start:end]
        slices = (arr[mbinds] for arr in (obs, actions, returns, values))
        mb_losses.append(model.train(*slices, lr))

    # Feedforward --> get losses --> update
    lossvalues = np.mean(mb_losses, axis=0)

    # End timer
    tnow = time.time()

    # Calculate the fps (frame per second)
    fps = int(batch_size / (tnow - tstart))

    if update % log_interval == 0 or update == 1:
      """
      Computes fraction of variance that ypred explains about y.
      Returns 1 - Var[y-ypred] / Var[y]
      interpretation:
      ev=0  =>  might as well have predicted zero(there is no benefit to it)
      ev=1  =>  perfect prediction
      ev<0  =>  worse than just predicting zero(we are in case of over fitting and we must choose better parmeters )
      
      the ideal case we need variance to be closer to one
      """
      ev = explained_variance(values, returns)
      logger.record_tabular("nupdates", update)
      logger.record_tabular("total_timesteps", update * batch_size)
      logger.record_tabular("fps", fps)
      logger.record_tabular("policy_loss", float(lossvalues[0]))
      logger.record_tabular("policy_entropy", float(lossvalues[2]))
      logger.record_tabular("value_loss", float(lossvalues[1]))
      logger.record_tabular("explained_variance", float(ev))
      logger.record_tabular("time elapsed", float(tnow - tfirststart))
      logger.dump_tabular()

      savepath = "./models/" + str(update) + "/model.ckpt"
      model.save(savepath)
      print('Saving to', savepath)

  env.close()


def play(policy , env):

  #  Get state_space and action_space

  ob_space = env.observation_space
  ac_space = env.action_space

  # Instantiate the model object (that creates step_model and train_model)
  model = Model(policy=policy,
                ob_space=ob_space,
                action_space=ac_space,
                nenvs=1,
                nsteps=1,
                ent_coef=0,
                vf_coef=0,
                max_grad_norm=0)

  # Load the model
  load_path = "./models/260/model.ckpt"
  model.load(load_path)

  obs = env.reset()

  # Play
  score = 0
  boom = 0
  done = False
  while done == False:
    boom += 1

    # Get the action
    actions, values = model.step(obs)

    # Take actions in env and look the results
    obs, rewards, done, _ = env.step(actions)

    score += rewards

    env.render()

  print("Score ", score)
  env.close()


def mse(pred, target):
  return tf.square(pred-target)/2.















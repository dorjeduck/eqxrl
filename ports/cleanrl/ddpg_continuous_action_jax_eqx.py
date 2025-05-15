# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_action_jaxpy
import os
import random
import time
from dataclasses import dataclass
from equinox import tree_pformat

import equinox as eqx

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from dataclasses import replace

from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:


class QNetwork(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, key):
        keys = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(obs_dim + action_dim, 256, key=keys[0])
        self.layer2 = eqx.nn.Linear(256, 256, key=keys[1])
        self.layer3 = eqx.nn.Linear(256, 1, key=keys[2])

    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    @jax.jit
    def forward_batch(self, x, a):
        result = jax.vmap(self.__call__)(x, a)
        return result


class Actor(eqx.Module):
    action_scale: jnp.array
    action_bias: jnp.array

    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, action_scale, action_bias, key):
        self.action_scale = action_scale
        self.action_bias = action_bias

        keys = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(obs_dim, 256, key=keys[0])
        self.layer2 = eqx.nn.Linear(256, 256, key=keys[1])
        self.layer3 = eqx.nn.Linear(256, action_dim, key=keys[2])

    def __call__(self, x: jnp.ndarray):

        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        x = jax.nn.tanh(self.layer3(x))
        x = x * self.action_scale + self.action_bias

        return x

    @jax.jit
    def forward_batch(self, x):
        result = jax.vmap(self.__call__)(x)
        return result


class TrainState(eqx.Module):
    model: eqx.Module
    target_model: eqx.Module
    tx: optax.GradientTransformation = eqx.static_field()
    opt_state: optax.OptState
    step: int

    def apply_gradients(self, grads):
        # filtered_model = eqx.filter(self.model, eqx.is_array)
        updates, new_opt_state = self.tx.update(grads, self.opt_state)
        new_model = eqx.apply_updates(self.model, updates)

        return self.replace(
            model=new_model,
            opt_state=new_opt_state,
            step=self.step + 1,
        )

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    @classmethod
    def create(cls, *, model, target_model, tx, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.
        opt_state = tx.init(eqx.filter(model, eqx.is_array))

        return cls(
            model=model,
            target_model=target_model,
            tx=tx,
            opt_state=opt_state,
            step=0,
            **kwargs,
        )


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.key(args.seed)
    key, actor_key, qf1_key = jax.random.split(key, 3)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device="cpu",
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = np.prod(envs.single_action_space.shape)

    # squeeze as for non batch eqx network
    action_scale = jnp.array(
        (envs.action_space.high - envs.action_space.low) / 2.0
    ).squeeze()
    action_bias = jnp.array(
        (envs.action_space.high + envs.action_space.low) / 2.0
    ).squeeze()

    actor = Actor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        action_bias=action_bias,
        key=actor_key,
    )

    t_actor = Actor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        action_bias=action_bias,
        key=actor_key,
    )

    actor_state = TrainState.create(
        model=actor,
        target_model=t_actor,
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    qf = QNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        key=qf1_key,
    )

    t_qf = QNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        key=qf1_key,
    )

    qf1_state = TrainState.create(
        model=qf,
        target_model=t_qf,
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
    ):

        next_state_actions = actor_state.target_model.forward_batch(
            next_observations
        ).clip(
            -1, 1
        )  # TODO: proper clip

        qf1_next_target = qf1_state.target_model.forward_batch(
            next_observations, next_state_actions
        ).reshape(-1)

        next_q_value = (
            rewards + (1 - terminations) * args.gamma * (qf1_next_target)
        ).reshape(-1)

        def mse_loss(model):
            qf_a_values = model.forward_batch(observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf1_state.model)
        qf1_state = qf1_state.apply_gradients(grads=grads1)

        return qf1_state, qf1_loss_value, qf1_a_values

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
    ):

        def actor_loss(model):
            return -qf1_state.model.forward_batch(
                observations, model.forward_batch(observations)
            ).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.model)

        actor_state = actor_state.apply_gradients(grads=grads)

        actor_state = actor_state.replace(
            target_model=optax.incremental_update(
                actor_state.model, actor_state.target_model, args.tau
            )
        )

        qf1_state = qf1_state.replace(
            target_model=optax.incremental_update(
                qf1_state.model, qf1_state.target_model, args.tau
            )
        )

        return actor_state, qf1_state, actor_loss_value

    start_time = time.time()
    for global_step in range(args.total_timesteps):

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = actor_state.model.forward_batch(obs)

            print(actor.action_scale)
            print(np.random.normal(0, actor.action_scale * args.exploration_noise))
            print("----")
            print(np.random.normal(0, actor.action_scale * args.exploration_noise)[0])
            exit()
            actions = np.array(
                [
                    (
                        jax.device_get(actions)[0]
                        + np.random.normal(
                            0, actor.action_scale * args.exploration_noise
                        )
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )

        # TRY NOT TO MODIFY: execute the game and log data.

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):

            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            qf1_state, qf1_loss_value, qf1_a_values = update_critic(
                actor_state,
                qf1_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
            )
            if global_step % args.policy_frequency == 0:

                actor_state, qf1_state, actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    data.observations.numpy(),
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar(
                    "losses/actor_loss", actor_loss_value.item(), global_step
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    """
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        actor_state.params,
                        qf1_state.params,
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ddpg_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DDPG", f"runs/{run_name}", f"videos/{run_name}-eval")
    """

    envs.close()
    writer.close()

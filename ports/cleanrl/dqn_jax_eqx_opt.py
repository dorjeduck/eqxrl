# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import os
import random
import time
from dataclasses import dataclass, replace

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from jax.tree_util import PyTreeDef

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
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


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

    def __init__(self, obs_dim: int, action_dim, key):
        keys = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(obs_dim, 120, key=keys[0])
        self.layer2 = eqx.nn.Linear(120, 84, key=keys[1])
        self.layer3 = eqx.nn.Linear(84, action_dim, key=keys[2])

    def __call__(self, x: jnp.ndarray):
        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        x = self.layer3(x)
        return x

@jax.jit
def forward_batch(model, *batch_inputs):
    return jax.vmap(model)(*batch_inputs)


class TrainState(eqx.Module):
    flat_model: list
    flat_opt_state: list

    flat_target_model: list

    treedef_model: PyTreeDef = eqx.static_field()
    treedef_opt_state: PyTreeDef = eqx.static_field()

    tx: optax.GradientTransformation = eqx.static_field()

    step: int

    def apply_gradients(self, grads):

        model = jax.tree.unflatten(self.treedef_model, self.flat_model)
        opt_state = jax.tree.unflatten(self.treedef_opt_state, self.flat_opt_state)

        updates, update_opt_state = self.tx.update(grads, opt_state)
        update_model = eqx.apply_updates(model, updates)

        flat_update_model = jax.tree.leaves(update_model)
        flat_update_opt_state = jax.tree.leaves(update_opt_state)

        return self.replace(
            flat_model=flat_update_model,
            flat_opt_state=flat_update_opt_state,
            step=self.step + 1,
        )

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    @classmethod
    def create(cls, *, model, target_model, tx, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.

        opt_state = tx.init(eqx.filter(model, eqx.is_array))

        flat_model, treedef_model = jax.tree.flatten(model)
        flat_target_model, _ = jax.tree.flatten(target_model)
        flat_opt_state, treedef_opt_state = jax.tree.flatten(opt_state)

        return cls(
            flat_model=flat_model,
            flat_opt_state=flat_opt_state,
            treedef_model=treedef_model,
            treedef_opt_state=treedef_opt_state,
            tx=tx,
            flat_target_model=flat_target_model,
            step=0,
            **kwargs,
        )


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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
    key, q_key = jax.random.split(key)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)
    q_network = QNetwork(
        obs_dim=envs.single_observation_space.shape[0],
        action_dim=envs.single_action_space.n,
        key=q_key,
    )

    q_state = TrainState.create(
        model=q_network,
        target_model=q_network,
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):

        target_model = jax.tree.unflatten(
            q_state.treedef_model, q_state.flat_target_model
        )

        q_next_target = forward_batch(target_model,next_observations)

        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(model):
            q_pred = forward_batch(model,observations)  # (batch_size, num_actions)
            q_pred = q_pred[
                jnp.arange(q_pred.shape[0]), actions.squeeze()
            ]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        model = jax.tree.unflatten(q_state.treedef_model, q_state.flat_model)

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(model)

        q_state = q_state.apply_gradients(grads=grads)

        return loss_value, q_pred, q_state

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            model = jax.tree.unflatten(q_state.treedef_model, q_state.flat_model)
            q_values = forward_batch(model,obs)

            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                # perform a gradient-descent step
                loss, old_val, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar(
                        "losses/td_loss", jax.device_get(loss), global_step
                    )
                    writer.add_scalar(
                        "losses/q_values", jax.device_get(old_val).mean(), global_step
                    )
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

            # update target network

            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    flat_target_model=optax.incremental_update(
                        q_state.flat_model, q_state.flat_target_model, args.tau
                    )
                )

    envs.close()
    writer.close()

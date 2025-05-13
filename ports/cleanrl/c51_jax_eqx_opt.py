# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51_jaxpy
import os
import random
import time
from dataclasses import dataclass

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

from dataclasses import replace


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
    n_atoms: int = 101
    """the number of atoms"""
    v_min: float = -100
    """the return lower bound"""
    v_max: float = 100
    """the return upper bound"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
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


class QNetwork(eqx.Module):

    n_atoms: int = eqx.static_field()
    action_dim: int = eqx.static_field()

    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim, n_atoms: int, key):

        self.action_dim = action_dim
        self.n_atoms = n_atoms

        keys = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(obs_dim, 120, key=keys[0])
        self.layer2 = eqx.nn.Linear(120, 84, key=keys[1])
        self.layer3 = eqx.nn.Linear(84, action_dim * n_atoms, key=keys[2])

    def __call__(self, x: jnp.ndarray):
        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        x = self.layer3(x)
        x = x.reshape(self.action_dim, self.n_atoms)
        x = jax.nn.softmax(x, axis=-1)  # pmfs
        return x


@jax.jit
def forward_batch(model, batch_x):
    return jax.vmap(model)(batch_x)


class TrainState(eqx.Module):
    flat_model: list
    flat_opt_state: list

    flat_target_model: list

    treedef_model: PyTreeDef = eqx.static_field()
    treedef_opt_state: PyTreeDef = eqx.static_field()
    tx: optax.GradientTransformation = eqx.static_field()
    atoms: jnp.ndarray
    step: int

    def apply_gradients(self, grads):
        # filtered_model = eqx.filter(self.model, eqx.is_array)
       
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
    key, q_key = jax.random.split(key, 2)

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
        n_atoms=args.n_atoms,
        key=q_key,
    )

    t_network = QNetwork(
        obs_dim=envs.single_observation_space.shape[0],
        action_dim=envs.single_action_space.n,
        n_atoms=args.n_atoms,
        key=q_key,
    )

    q_state = TrainState.create(
        model=q_network,
        target_model=t_network,
        # directly using jnp.linspace leads to numerical errors
        atoms=jnp.asarray(np.linspace(args.v_min, args.v_max, num=args.n_atoms)),
        tx=optax.adam(learning_rate=args.learning_rate, eps=0.01 / args.batch_size),
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

        next_pmfs = forward_batch(
            target_model, next_observations
        )  # (batch_size, num_actions, num_atoms)
        next_vals = (next_pmfs * q_state.atoms).sum(
            axis=-1
        )  # (batch_size, num_actions)
        next_action = jnp.argmax(next_vals, axis=-1)  # (batch_size,)
        next_pmfs = next_pmfs[np.arange(next_pmfs.shape[0]), next_action]
        next_atoms = rewards + args.gamma * q_state.atoms * (1 - dones)
        # projection
        delta_z = q_state.atoms[1] - q_state.atoms[0]
        tz = jnp.clip(next_atoms, a_min=(args.v_min), a_max=(args.v_max))

        b = (tz - args.v_min) / delta_z
        l = jnp.clip(jnp.floor(b), a_min=0, a_max=args.n_atoms - 1)
        u = jnp.clip(jnp.ceil(b), a_min=0, a_max=args.n_atoms - 1)
        # (l == u).astype(jnp.float) handles the case where bj is exactly an integer
        # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
        d_m_l = (u + (l == u).astype(jnp.float32) - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs
        target_pmfs = jnp.zeros_like(next_pmfs)

        def project_to_bins(i, val):
            val = val.at[i, l[i].astype(jnp.int32)].add(d_m_l[i])
            val = val.at[i, u[i].astype(jnp.int32)].add(d_m_u[i])
            return val

        target_pmfs = jax.lax.fori_loop(
            0, target_pmfs.shape[0], project_to_bins, target_pmfs
        )

        def loss(model, observations, actions, target_pmfs):
            pmfs = forward_batch(model, observations)

            old_pmfs = pmfs[np.arange(pmfs.shape[0]), actions.squeeze()]

            old_pmfs_l = jnp.clip(old_pmfs, a_min=1e-5, a_max=1 - 1e-5)
            loss = (-(target_pmfs * jnp.log(old_pmfs_l)).sum(-1)).mean()
            return loss, (old_pmfs * q_state.atoms).sum(-1)
        
        model = jax.tree.unflatten(q_state.treedef_model, q_state.flat_model)

        (loss_value, old_values), grads = jax.value_and_grad(loss, has_aux=True)(
            model, observations, actions, target_pmfs
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, old_values, q_state

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
          
            pmfs = forward_batch(model, obs)
            q_vals = (pmfs * q_state.atoms).sum(axis=-1)
            actions = q_vals.argmax(axis=-1)
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
        if (
            global_step > args.learning_starts
            and global_step % args.train_frequency == 0
        ):
            data = rb.sample(args.batch_size)
            loss, old_val, q_state = update(
                q_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.numpy(),
                data.dones.numpy(),
            )

            if global_step % 100 == 0:
                writer.add_scalar("losses/loss", jax.device_get(loss), global_step)
                writer.add_scalar(
                    "losses/q_values", jax.device_get(old_val.mean()), global_step
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
                        q_state.flat_model, q_state.flat_target_model, 1
                    )
                )
    """
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_state.params,
            "args": vars(args),
        }
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(model_data))
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.c51_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        
        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "C51", f"runs/{run_name}", f"videos/{run_name}-eval")
        """

    envs.close()
    writer.close()

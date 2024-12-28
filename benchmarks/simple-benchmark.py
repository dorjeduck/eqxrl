import jax
import jax.numpy as jnp

import equinox as eqx
import gymnax
import time
import optax
import os


from libs import (
    ActorLinen,
    TrainingStateLinen,
    ActorEqx,
    TrainingStateEqx,
    collect_experience_linen,
    collect_experience_eqx,
    update_policy_linen,
    update_policy_eqx,
    forward_pass_linen,
    forward_pass_eqx,
)


def init_training(seed=0, framework="linen"):
    env, env_params = gymnax.make("CartPole-v1")
    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)

    obs_size = env.observation_space(env_params).shape[0]
    action_size = env.action_space(env_params).n

    optimizer = optax.adam(1e-3)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset(key, env_params)

    if framework == "linen":
        actor = ActorLinen(action_size=action_size)
        params = actor.init(key, jnp.ones((obs_size,)))
        opt_state = optimizer.init(params)

        train_state = TrainingStateLinen(
            params=params, opt_state=opt_state, rng=rng, env_state=env_state, obs=obs
        )

    else:
        actor = ActorEqx(key, obs_size, action_size)

        opt_state = optimizer.init(eqx.filter(actor, eqx.is_array))

        train_state = TrainingStateEqx(
            actor=actor, opt_state=opt_state, rng=rng, env_state=env_state, obs=obs
        )

    return train_state, env, env_params, actor


def profile_training(steps=1000):
    frameworks = ["linen", "eqx"]
    results = {}

    for fw in frameworks:
        state, env, env_params, actor = init_training(framework=fw)

        if fw == "linen":
            forward_fn = forward_pass_linen
            collect_fn = collect_experience_linen
            update_fn = update_policy_linen
        else:
            forward_fn = forward_pass_eqx
            collect_fn = collect_experience_eqx
            update_fn = update_policy_eqx

        # Warmup
        for _ in range(5):
            _ = forward_fn(state, actor)
            state = collect_fn(state, env, env_params, actor)
            state = update_fn(state)

        # Profile full step
        start = time.perf_counter()
        for _ in range(steps):
            state = collect_fn(state, env, env_params, actor)
            state = update_fn(state)
            state = jax.block_until_ready(state)
        full_time = (time.perf_counter() - start) / steps

        # Profile components
        times = {"forward": 0.0, "collect": 0.0, "policy": 0.0}
        for _ in range(steps):

            start = time.perf_counter()
            _ = jax.block_until_ready(forward_fn(state, actor))
            times["forward"] += (time.perf_counter() - start) / steps

            start = time.perf_counter()
            state = jax.block_until_ready(collect_fn(state, env, env_params, actor))
            times["collect"] += (time.perf_counter() - start) / steps

            start = time.perf_counter()
            state = jax.block_until_ready(update_fn(state))
            times["policy"] += (time.perf_counter() - start) / steps

        results[fw] = {
            "full_time": full_time,
            "forward_time": times["forward"],
            "collect_time": times["collect"],
            "policy_time": times["policy"],
            "sum_components": sum(times.values()),
            "overhead": full_time - sum(times.values()),
        }

    print("\nBenchmarking Results Comparison:")
    print(f"{'Metric':<25} {'Linen':<15} {'Equinox':<15} {'Difference (%)':<15}")
    print(f"{'-'*70}")
    print(
        f"{'Full step (ms)':<25} {results['linen']['full_time']*1000:<15.3f} {results['eqx']['full_time']*1000:<15.3f} "
        f"{((results['eqx']['full_time'] - results['linen']['full_time']) / results['linen']['full_time']) * 100:<15.2f}"
    )
    print(
        f"{'Forward (ms)':<25} {results['linen']['forward_time']*1000:<15.3f} {results['eqx']['forward_time']*1000:<15.3f} "
        f"{((results['eqx']['forward_time'] - results['linen']['forward_time']) / results['linen']['forward_time']) * 100:<15.2f}"
    )
    print(
        f"{'Collect (ms)':<25} {results['linen']['collect_time']*1000:<15.3f} {results['eqx']['collect_time']*1000:<15.3f} "
        f"{((results['eqx']['collect_time'] - results['linen']['collect_time']) / results['linen']['collect_time']) * 100:<15.2f}"
    )
    print(
        f"{'Policy update (ms)':<25} {results['linen']['policy_time']*1000:<15.3f} {results['eqx']['policy_time']*1000:<15.3f} "
        f"{((results['eqx']['policy_time'] - results['linen']['policy_time']) / results['linen']['policy_time']) * 100:<15.2f}"
    )

    # Write results to Markdown file
    with open("benchmark_results.md", "w") as f:
        f.write("# Benchmarking Results Comparison\n\n")
        f.write(
            f"| {'Metric':<25} | {'Linen':<15} | {'Equinox':<15} | {'Difference (%)':<15} |\n"
        )
        f.write(f"|{'-'*25}|{'-'*15}|{'-'*15}|{'-'*15}|\n")
        f.write(
            f"| {'Full step (ms)':<25} | {results['linen']['full_time']*1000:<15.3f} | {results['eqx']['full_time']*1000:<15.3f} | "
            f"{((results['eqx']['full_time'] - results['linen']['full_time']) / results['linen']['full_time']) * 100:<15.2f} |\n"
        )
        f.write(
            f"| {'Forward (ms)':<25} | {results['linen']['forward_time']*1000:<15.3f} | {results['eqx']['forward_time']*1000:<15.3f} | "
            f"{((results['eqx']['forward_time'] - results['linen']['forward_time']) / results['linen']['forward_time']) * 100:<15.2f} |\n"
        )
        f.write(
            f"| {'Collect (ms)':<25} | {results['linen']['collect_time']*1000:<15.3f} | {results['eqx']['collect_time']*1000:<15.3f} | "
            f"{((results['eqx']['collect_time'] - results['linen']['collect_time']) / results['linen']['collect_time']) * 100:<15.2f} |\n"
        )
        f.write(
            f"| {'Policy update (ms)':<25} | {results['linen']['policy_time']*1000:<15.3f} | {results['eqx']['policy_time']*1000:<15.3f} | "
            f"{((results['eqx']['policy_time'] - results['linen']['policy_time']) / results['linen']['policy_time']) * 100:<15.2f} |\n"
        )


if __name__ == "__main__":
    profile_training()

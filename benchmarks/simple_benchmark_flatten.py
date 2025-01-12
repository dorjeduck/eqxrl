import jax
import jax.numpy as jnp

import equinox as eqx
import gymnax
import time
import optax


from libs import (
    ActorLinen,
    TrainingStateLinen,
    ActorEqx,
    TrainingStateEqxFlatten,
    collect_experience_linen,
    collect_experience_eqx_flatten,
    update_policy_linen,
    update_policy_eqx_flatten,
    forward_pass_linen,
    forward_pass_eqx_flatten,
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
        treedef = None

    else:
        actor = ActorEqx(key, obs_size, action_size)

        opt_state = optimizer.init(eqx.filter(actor, eqx.is_array))

        leaf_values, treedef = jax.tree.flatten(actor)

        train_state = TrainingStateEqxFlatten(
            leaf_values=leaf_values,
            opt_state=opt_state,
            rng=rng,
            env_state=env_state,
            obs=obs,
        )

    return train_state, env, env_params, actor, treedef


def profile_training(steps=100, runs=10):
    frameworks = ["linen", "eqx"]
    results = {
        fw: {"full_time": [], "forward_time": [], "collect_time": [], "policy_time": []}
        for fw in frameworks
    }

    for _ in range(runs):
        for fw in frameworks:
            state, env, env_params, actor, treedef = init_training(framework=fw)

            if fw == "linen":
                forward_fn = forward_pass_linen
                collect_fn = collect_experience_linen
                update_fn = update_policy_linen
                extra = actor
            else:
                forward_fn = forward_pass_eqx_flatten
                collect_fn = collect_experience_eqx_flatten
                update_fn = update_policy_eqx_flatten
                extra = treedef

            # Warmup
            for _ in range(5):
                _ = forward_fn(state, extra)
                state = collect_fn(state, env, env_params, extra)
                state = update_fn(state, extra)

            # Profile full step
            start = time.perf_counter()
            for _ in range(steps):
                state = collect_fn(state, env, env_params, extra)
                state = jax.block_until_ready(update_fn(state, extra))

            full_time = (time.perf_counter() - start) / steps
            results[fw]["full_time"].append(full_time)

            # Profile components
            times = {"forward": 0.0, "collect": 0.0, "policy": 0.0}
            for _ in range(steps):
                start = time.perf_counter()
                _ = jax.block_until_ready(forward_fn(state, extra))
                times["forward"] += (time.perf_counter() - start) / steps

                start = time.perf_counter()
                state = jax.block_until_ready(collect_fn(state, env, env_params, extra))
                times["collect"] += (time.perf_counter() - start) / steps

                start = time.perf_counter()
                state = jax.block_until_ready(update_fn(state, extra))
                times["policy"] += (time.perf_counter() - start) / steps

            results[fw]["forward_time"].append(times["forward"])
            results[fw]["collect_time"].append(times["collect"])
            results[fw]["policy_time"].append(times["policy"])

    avg_results = {
        fw: {k: sum(v) / runs for k, v in results[fw].items()} for fw in frameworks
    }
    avg_results["linen"]["sum_components"] = sum(avg_results["linen"].values())
    avg_results["eqx"]["sum_components"] = sum(avg_results["eqx"].values())
    avg_results["linen"]["overhead"] = (
        avg_results["linen"]["full_time"] - avg_results["linen"]["sum_components"]
    )
    avg_results["eqx"]["overhead"] = (
        avg_results["eqx"]["full_time"] - avg_results["eqx"]["sum_components"]
    )

    metrics = ["full_time", "forward_time", "collect_time", "policy_time"]
    metric_names = [
        "Full step",
        "Forward pass",
        "Experience collection",
        "Policy update",
    ]

    # Print and write results to Markdown file
    with open("benchmark_results.md", "w") as f:
        print("\nBenchmarking Results Comparison:")
        print(
            f"{'Metric':<25} {'Linen (ms)':<15} {'Equinox (ms)':<15} {'Difference (%)':<15}"
        )
        print(f"{'-'*70}")

        f.write("# Benchmarking Results Comparison\n\n")
        f.write(
            f"| {'Metric':<25} | {'Linen (ms)':<15} | {'Equinox (ms)':<15} | {'Difference (%)':<15} |\n"
        )
        f.write(f"|{'-'*25}|{'-'*15}|{'-'*15}|{'-'*15}|\n")

        for metric, name in zip(metrics, metric_names):
            linen_time = avg_results["linen"][metric] * 1000
            eqx_time = avg_results["eqx"][metric] * 1000
            diff_percent = ((eqx_time - linen_time) / linen_time) * 100
            print(
                f"{name:<25} {linen_time:<15.3f} {eqx_time:<15.3f} {diff_percent:<15.2f}"
            )
            f.write(
                f"| {name:<25} | {linen_time:<15.3f} | {eqx_time:<15.3f} | {diff_percent:<15.2f} |\n"
            )


if __name__ == "__main__":
    profile_training()

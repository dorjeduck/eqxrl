from jax import numpy as jnp


def print_results(duration, results, num_results=3):
    for idx in jnp.lexsort(
        (
            -jnp.arange(len(results["metrics"]["returns"][0])),
            results["metrics"]["returns"][0],
        )
    )[-num_results:][::-1]:
        print(f"Index: {idx}, Value: {results['metrics']['returns'][0][idx]}")
    print(f"Duration: {duration:.2f} seconds")

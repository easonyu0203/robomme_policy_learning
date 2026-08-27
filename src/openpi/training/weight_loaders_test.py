import jax
import numpy as np

from openpi.training.weight_loaders import _merge_params


def test_merge_params_string_keys():
    """Plain paligemma-style merge: loaded weights win, dtype cast to ref."""
    ref = {"a": {"b": jax.ShapeDtypeStruct((2, 2), np.float32)},
           "c": jax.ShapeDtypeStruct((3,), np.float32)}
    loaded = {"a": {"b": np.ones((2, 2), np.float64)}}
    merged = _merge_params(loaded, ref, missing_regex=".*")
    assert merged["a"]["b"].dtype == np.float32
    assert merged["c"] is ref["c"]  # missing -> filled from ref


def test_merge_params_int_keys_from_list_submodule():
    """An nnx list attribute (e.g. Selector.blocks) yields int path segments;
    _merge_params must not choke trying to "/".join them."""
    ref = {
        "PaliGemma": {"llm": {"kernel": jax.ShapeDtypeStruct((4, 4), np.float32)}},
        "mem_encoder": {"selector": {"blocks": {
            0: {"norm1": {"scale": jax.ShapeDtypeStruct((8,), np.float32)}},
            1: {"q_proj": {"kernel": jax.ShapeDtypeStruct((8, 8), np.float32)}},
        }}},
    }
    loaded = {"PaliGemma": {"llm": {"kernel": np.ones((4, 4), np.float64)}}}

    merged = _merge_params(loaded, ref, missing_regex=".*")

    assert merged["PaliGemma"]["llm"]["kernel"].dtype == np.float32
    assert set(merged["mem_encoder"]["selector"]["blocks"]) == {0, 1}
    assert merged["mem_encoder"]["selector"]["blocks"][0]["norm1"]["scale"].shape == (8,)


def test_merge_params_missing_regex_filters():
    ref = {"keep": {"x": jax.ShapeDtypeStruct((1,), np.float32)},
           "drop": {"y": jax.ShapeDtypeStruct((1,), np.float32)}}
    merged = _merge_params({}, ref, missing_regex=r"keep/.*")
    assert "keep" in merged and "drop" not in merged

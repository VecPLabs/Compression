from benchmark_dynamic_mlp_compression import allocate


def test_allocate_spends_rank_on_the_more_sensitive_layer():
    profile = [
        [{"rank": 16, "mean_kl": 5.0}, {"rank": 32, "mean_kl": 1.0}],
        [{"rank": 16, "mean_kl": 2.0}, {"rank": 32, "mean_kl": 1.9}],
    ]
    choices, used, cost = allocate(profile, [16, 32], budget=48)
    assert choices == [32, 16]
    assert used == 48
    assert cost == 3.0

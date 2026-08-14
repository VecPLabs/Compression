import torch

from benchmark_dynamic_residual_width import run_with_widths


def test_full_width_skips_intervention():
    class Model:
        gpt_neox = type("NeoX", (), {"layers": []})()

        def __call__(self, token_ids, use_cache=False):
            return type("Output", (), {"logits": token_ids.float().unsqueeze(-1)})()

    token_ids = torch.tensor([[1, 2, 3]])
    logits = run_with_widths(Model(), token_ids, [], [])
    assert torch.equal(logits.squeeze(-1), token_ids.float())

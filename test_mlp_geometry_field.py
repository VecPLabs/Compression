import torch

from analyze_mlp_geometry_field import joint_basis, subspace_similarity


def test_subspace_similarity_is_one_for_same_orthonormal_basis():
    basis = torch.eye(4)[:, :2]
    assert subspace_similarity(basis, basis) == 1.0


def test_joint_basis_is_orthonormal():
    write = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    fisher = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    basis = joint_basis(write, fisher, rank=2)
    assert torch.allclose(basis.T @ basis, torch.eye(2), atol=1e-5)

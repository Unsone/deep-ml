import torch

def cramers_rule(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve system of linear equations Ax = b using Cramer's Rule.
    Returns solution vector x or -1 if no unique solution exists.
    """
    det_A = torch.linalg.det(A)
    if torch.isclose(det_A, torch.zeros_like(det_A)):
        return -1

    n = A.shape[0]
    x = torch.zeros(n, dtype=det_A.dtype)
    for i in range(n):
        A_i = A.clone()
        A_i[:, i] = b          # 用 b 替换第 i 列
        x[i] = torch.linalg.det(A_i) / det_A
    return x
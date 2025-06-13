import torch
import torch.nn as nn
import torch.nn.functional as F


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins Loss implementation.
    This loss function encourages the cross-correlation matrix between the representations
    of two augmented views of a batch of samples to be close to the identity matrix.
    In the context of JEPA, this can be applied to a single set of embeddings (e.g., online_s_t_emb)
    to regularize its feature space.

    Args:
        lambda_param (float): Weight for the redundancy reduction term (off-diagonal elements).
                              Defaults to 5e-3, a common value from the paper.
        eps (float): Small epsilon value for numerical stability in normalization.
                     Defaults to 1e-5.
        scale_loss (float): Scaling factor for the total loss. Not in original paper, but can be useful.
                            Defaults to 1.0.
    """

    def __init__(self, lambda_param=5e-3, eps=1e-5, scale_loss=1.0):
        super().__init__()
        self.lambda_param = lambda_param
        self.eps = eps
        self.scale_loss = scale_loss

    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the Barlow Twins loss.

        If only z1 is provided, it computes the loss for z1 against itself (auto-correlation).
        This is suitable for JEPA's auxiliary loss on a single set of embeddings.
        If z2 is provided, it computes the loss between z1 and z2.

        Args:
            z1 (torch.Tensor): First batch of embeddings, shape (N, D).
            z2 (torch.Tensor, optional): Second batch of embeddings, shape (N, D).
                                         If None, z1 is used for both branches.

        Returns:
            torch.Tensor: The Barlow Twins loss value.
        """
        if z1.ndim != 2:
            raise ValueError(f"Input tensor z1 must be 2D (batch_size, feature_dim). Got ndim={z1.ndim}")

        if z2 is None:
            # Auto-correlation case, z1 is used for both branches.
            # Apply the batch size check for z1.
            if z1.shape[0] < 2:
                return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
            z2 = z1 # z2 now refers to z1
        else:
            # z2 is provided. Ensure it also has the same shape and is 2D.
            if z2.ndim != 2:
                raise ValueError(f"Input tensor z2 must be 2D (batch_size, feature_dim). Got ndim={z2.ndim}")
            if z1.shape != z2.shape:
                raise ValueError(f"Input tensors z1 and z2 must have the same shape. Got {z1.shape} and {z2.shape}")
            # If z1 and z2 are distinct, and *either* has batch_size < 2, std dev will be NaN.
            # This case is less directly related to the vjepa2 issue which uses calculate_reg_terms
            # or forward(z, None). For now, we focus on the single input case.
            # A more general solution for distinct z1, z2 with small batch sizes might be needed
            # if this loss is used in other contexts, but let's stick to the original problem's scope.
            # The primary concern is when z1.std or z2.std (which is z1.std if z2 is None) becomes NaN.
            if z1.shape[0] < 2: # This implies z2.shape[0] < 2 due to shape assertion
                # This path is taken if z2 was not None, but z1's batch size is < 2.
                # Both z1.std and z2.std would be NaN.
                return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)


        batch_size, feature_dim = z1.shape # batch_size from z1

        # Normalization (z1.std and z2.std will be NaN if their effective batch_size is 1)
        z1_norm = (z1 - z1.mean(dim=0, keepdim=True)) / \
            (z1.std(dim=0, keepdim=True) + self.eps)
        z2_norm = (z2 - z2.mean(dim=0, keepdim=True)) / \
            (z2.std(dim=0, keepdim=True) + self.eps)

        # 2. Calculate the cross-correlation matrix
        # c_ij = sum_k (z_norm_1_ki * z_norm_2_kj) / batch_size
        # (D, N) @ (N, D) -> (D, D)
        cross_corr_matrix = (z1_norm.T @ z2_norm) / batch_size

        # 3. Calculate loss terms
        # Invariance term: MSE loss for diagonal elements towards 1
        # (sum_i (c_ii - 1)^2)
        on_diag = torch.diagonal(cross_corr_matrix)
        invariance_loss = ((on_diag - 1)**2).sum()

        # Redundancy reduction term: sum of squared off-diagonal elements
        # lambda * sum_i sum_j!=i (c_ij^2)
        off_diag = self._off_diagonal(cross_corr_matrix)
        redundancy_loss = (off_diag**2).sum()

        total_loss = invariance_loss + self.lambda_param * redundancy_loss
        return total_loss * self.scale_loss

    def calculate_reg_terms(self, z: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to align with VICReg's usage in the current JEPA setup.
        Calculates Barlow Twins loss for a single batch of embeddings z.

        Args:
            z (torch.Tensor): Batch of embeddings, shape (N, D).

        Returns:
            torch.Tensor: The Barlow Twins loss value.
            torch.Tensor: The invariance component of the loss.
            torch.Tensor: The redundancy component of the loss.
        """
        assert z.ndim == 2, "Input tensor must be 2D (batch_size, feature_dim)."
        batch_size, feature_dim = z.shape

        if batch_size < 2:
            # If batch size is less than 2, std dev is undefined or unstable (NaN for batch_size=1).
            # Return zeros for all loss components.
            # Ensure the zeros are on the same device as the input tensor.
            zero_loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
            # BarlowTwins returns 3 values: total_loss, invariance_loss, weighted_redundancy_loss
            return zero_loss, zero_loss, zero_loss

        # Original calculations follow
        z_norm = (z - z.mean(dim=0, keepdim=True)) / \
            (z.std(dim=0, keepdim=True) + self.eps)

        # Auto-correlation matrix
        auto_corr_matrix = (z_norm.T @ z_norm) / batch_size

        on_diag = torch.diagonal(auto_corr_matrix)
        invariance_loss = ((on_diag - 1)**2).sum()

        off_diag = self._off_diagonal(auto_corr_matrix)
        redundancy_loss = (off_diag**2).sum()

        weighted_redundancy_loss = self.lambda_param * redundancy_loss
        total_loss = (invariance_loss +
                      weighted_redundancy_loss) * self.scale_loss

        # For compatibility with VICRegLoss output structure (total, term1, term2)
        # Here, term1 is invariance, term2 is weighted redundancy
        return total_loss, invariance_loss * self.scale_loss, weighted_redundancy_loss * self.scale_loss


if __name__ == '__main__':
    # Example Usage
    batch_size, embed_dim = 128, 256
    lambda_param = 5e-3  # Default from paper
    loss_fn = BarlowTwinsLoss(lambda_param=lambda_param)

    # Test with a single input z (for JEPA regularization style)
    z_embeddings = torch.randn(batch_size, embed_dim) * 5  # Some scale
    z_embeddings[:, :embed_dim//2] += 2  # Some mean shift

    # Using calculate_reg_terms
    total_loss_reg, inv_loss_reg, red_loss_reg = loss_fn.calculate_reg_terms(
        z_embeddings)
    print(f"Barlow Twins (Reg Terms for JEPA style on z1 only):")
    print(f"  Total Loss: {total_loss_reg.item():.4f}")
    print(f"  Invariance Loss (scaled): {inv_loss_reg.item():.4f}")
    print(f"  Redundancy Loss (scaled, weighted): {red_loss_reg.item():.4f}")

    # Test with two inputs z1, z2 (original Barlow Twins style)
    z1 = torch.randn(batch_size, embed_dim)
    z2 = z1 + 0.1 * torch.randn_like(z1)  # z2 is a slightly perturbed z1

    total_loss_two_inputs = loss_fn(z1, z2)
    print(f"\nBarlow Twins (Two inputs z1, z2):")
    print(f"  Total Loss: {total_loss_two_inputs.item():.4f}")

    # Test with perfectly correlated and normalized input (should yield low loss)
    # Create z_perfect where features are already normalized and orthogonal (identity correlation)
    # This is hard to achieve perfectly for a random matrix, but let's try to make it close to ideal.
    # If z_norm.T @ z_norm / N is identity, inv_loss and red_loss should be 0.

    # Ideal case: z_norm leads to identity correlation matrix
    # For calculate_reg_terms
    ideal_z = torch.randn(batch_size, embed_dim)
    ideal_z_norm = (ideal_z - ideal_z.mean(dim=0, keepdim=True)
                    ) / (ideal_z.std(dim=0, keepdim=True) + 1e-5)

    # If we make ideal_z_norm orthogonal, its auto-correlation will be diagonal.
    # For simplicity, let's test the components.
    # If correlation matrix is identity:
    # on_diag = 1, inv_loss = sum((1-1)^2) = 0
    # off_diag = 0, red_loss = sum(0^2) = 0
    # So total_loss should be 0.

    # Test with z_norm that would result in identity matrix if it were its own transpose (up to scaling)
    # More practically, let's test the normalization and calculation:
    # Create z where z_norm becomes easily predictable.
    # E.g., if all columns are identical after normalization, correlation matrix is all 1s.
    # Already mean-centered (mean=1), std=0. Normalization will handle this.
    z_all_ones_norm = torch.ones(batch_size, embed_dim)
    # Actually, std=0 will lead to NaN. Let's use randn.
    z_test_norm = torch.randn(batch_size, embed_dim)
    z_test_norm = (z_test_norm - z_test_norm.mean(dim=0, keepdim=True)
                   ) / (z_test_norm.std(dim=0, keepdim=True) + 1e-5)

    # If z_norm's columns are orthogonal, then z_norm.T @ z_norm is diagonal.
    # If z_norm's columns are orthonormal, then z_norm.T @ z_norm is identity (if N=D and z_norm is square orthogonal matrix).
    # This is not generally true for rectangular NxD matrices.

    # The goal is that (z_norm.T @ z_norm) / N should be identity.
    # Let's use a matrix that is already normalized and has identity correlation matrix
    # For example, take a batch of standard normal vectors. Their sample covariance won't be exactly identity.

    print(f"\nTesting with input that should give low loss:")
    # Create a z that is already normalized (features have mean 0, std 1)
    z_pre_normalized = torch.randn(batch_size, embed_dim)
    z_pre_normalized = (z_pre_normalized - z_pre_normalized.mean(dim=0,
                        keepdim=True)) / (z_pre_normalized.std(dim=0, keepdim=True) + 1e-5)

    # To make off-diagonal 0, features should be uncorrelated.
    # Using PCA and then whitening could achieve this for a given dataset.
    # For a random matrix, it's unlikely to be perfectly uncorrelated.
    # However, if input is already normalized, the first step in forward() does it again, but it's idempotent.

    loss_val_prenorm, inv_prenorm, red_prenorm = loss_fn.calculate_reg_terms(
        z_pre_normalized)
    print(f"  Loss with pre-normalized input (should be relatively low):")
    print(
        f"    Total: {loss_val_prenorm.item():.4f}, Inv: {inv_prenorm.item():.4f}, Red: {red_prenorm.item():.4f}")

    # What if z is such that z_norm.T @ z_norm / N is identity?
    # Example: if embed_dim = batch_size, and z_norm is an orthogonal matrix scaled by sqrt(batch_size)
    if batch_size == embed_dim:
        q, _ = torch.linalg.qr(torch.randn(
            batch_size, batch_size))  # q is orthogonal
        # This makes z_norm.T @ z_norm / batch_size = I
        z_orthogonal_norm = q * (batch_size**0.5)

        # We need to construct z such that its normalization becomes z_orthogonal_norm
        # This is tricky. Let's just test the core math with an identity correlation matrix.
        identity_corr = torch.eye(embed_dim, device=z_embeddings.device)
        on_diag_ideal = torch.diagonal(identity_corr)
        inv_loss_ideal = ((on_diag_ideal - 1)**2).sum()  # Should be 0

        off_diag_ideal = loss_fn._off_diagonal(identity_corr)
        red_loss_ideal = (off_diag_ideal**2).sum()  # Should be 0

        print(f"\nIdeal component losses (if correlation matrix is Identity):")
        print(f"  Ideal Invariance Loss: {inv_loss_ideal.item()}")
        print(f"  Ideal Redundancy Loss: {red_loss_ideal.item()}")
        print(
            f"  Total Ideal (unscaled by lambda): {inv_loss_ideal.item() + red_loss_ideal.item()}")

    # Test the _off_diagonal helper
    test_matrix = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    off_diag_elements = loss_fn._off_diagonal(test_matrix)
    expected_off_diag = torch.tensor(
        [2, 3, 4, 6, 7, 8], dtype=torch.float32)  # Corrected expected
    # flatten()[:-1] -> 1,2,3,4,5,6,7,8
    # view(2, 4) -> [[1,2,3,4],[5,6,7,8]]
    # [:, 1:] -> [[2,3,4],[6,7,8]]
    # flatten() -> [2,3,4,6,7,8] -> Correct!
    assert torch.allclose(
        off_diag_elements, expected_off_diag), f"Off-diagonal failed. Got {off_diag_elements}, expected {expected_off_diag}"
    print("\n_off_diagonal helper test passed.")

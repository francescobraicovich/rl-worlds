import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps  # Epsilon for numerical stability in variance calculation

    def forward(self, x, y):
        representation_loss = F.mse_loss(x, y)
        sim_loss = self.sim_coeff * representation_loss

        x_std = torch.sqrt(x.var(dim=0) + self.eps)  # (D,)
        # Hinge loss: max(0, 1 - std_dev)
        std_loss_x = torch.mean(F.relu(1 - x_std))

        y_std = torch.sqrt(y.var(dim=0) + self.eps)  # (D,)
        std_loss_y = torch.mean(F.relu(1 - y_std))  # Hinge loss

        std_loss = self.std_coeff * \
            (std_loss_x + std_loss_y) * 0.5  # Average over x and y

        x_centered = x - x.mean(dim=0)
        cov_x = (x_centered.T @ x_centered) / (x.size(0) - 1)
        cov_loss_x = (cov_x.fill_diagonal_(0).pow_(2).sum()) / \
            x.size(1)  # Division by D (num features)

        y_centered = y - y.mean(dim=0)
        cov_y = (y_centered.T @ y_centered) / (y.size(0) - 1)
        cov_loss_y = (cov_y.fill_diagonal_(0).pow_(2).sum()) / y.size(1)

        cov_loss = self.cov_coeff * \
            (cov_loss_x + cov_loss_y) * 0.5  # Average over x and y

        total_loss = sim_loss + std_loss + cov_loss
        return total_loss, sim_loss, std_loss, cov_loss

    def calculate_reg_terms(self, z):
        if z.size(0) < 2:
            # If batch size is less than 2, variance and covariance are undefined or unstable.
            # Return zeros for all loss components.
            # Ensure the zeros are on the same device as the input tensor.
            zero_loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
            return zero_loss, zero_loss, zero_loss

        # Original calculations follow
        z_std = torch.sqrt(z.var(dim=0) + self.eps)  # (D,)
        std_loss_val = torch.mean(F.relu(1 - z_std))  # Hinge loss

        z_centered = z - z.mean(dim=0)
        # Ensure denominator is not zero for covariance calculation.
        # The check z.size(0) < 2 already handles the z.size(0) - 1 == 0 case.
        # However, if z.size(0) was exactly 1 and not caught, this would be an issue.
        # The primary check is sufficient.
        cov_z = (z_centered.T @ z_centered) / (z.size(0) - 1)

        # Check for feature dimension being zero to prevent division by zero.
        # This is highly unlikely for embeddings but good for robustness.
        if z.size(1) == 0:
            cov_loss_val = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        else:
            cov_loss_val = (cov_z.fill_diagonal_(0).pow_(2).sum()) / z.size(1)

        weighted_std_loss = self.std_coeff * std_loss_val
        weighted_cov_loss = self.cov_coeff * cov_loss_val

        total_reg_loss = weighted_std_loss + weighted_cov_loss

        return total_reg_loss, weighted_std_loss, weighted_cov_loss


# Keep the original __main__ example for now, it's good for quick testing
if __name__ == '__main__':
    # Example Usage
    vicreg = VICRegLoss(sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0)

    # For full VICReg (e.g., with augmentations)
    batch_size, embed_dim = 128, 256
    x_embeddings = torch.randn(batch_size, embed_dim)
    y_embeddings = x_embeddings + 0.05 * \
        torch.randn_like(x_embeddings)  # y is a slightly perturbed x

    total_loss_full, sim_l, std_l, cov_l = vicreg(x_embeddings, y_embeddings)
    print(
        f"Full VICReg -> Total: {total_loss_full.item():.4f}, Sim: {sim_l.item():.4f}, Std: {std_l.item():.4f}, Cov: {cov_l.item():.4f}")

    # For JEPA-style regularization (only std and cov terms on a single set of embeddings)
    z_embeddings = torch.randn(batch_size, embed_dim)
    # Simulate low variance for some features
    z_embeddings[:, :embed_dim//2] *= 0.1
    # Simulate some covariance
    z_embeddings[:, 1] += 0.5 * z_embeddings[:, 0]

    total_reg, std_reg, cov_reg = vicreg.calculate_reg_terms(z_embeddings)
    print(
        f"Reg Terms Only (for JEPA) -> Total: {total_reg.item():.4f}, Std: {std_reg.item():.4f}, Cov: {cov_reg.item():.4f}")

    # Example with target std close to 1 and low covariance
    z_good_embeddings = torch.randn(batch_size, embed_dim)
    # z_good_embeddings = F.normalize(z_good_embeddings, p=2, dim=1) # Normalize to unit sphere
    # z_good_embeddings = z_good_embeddings * (embed_dim**0.5) # Scale to make std around 1 (Barlow Twins trick)
    # This normalization makes std of each sample 1. We need std of each feature over batch.
    # Let's try to make features have std=1 directly.
    z_good_embeddings = torch.randn(batch_size, embed_dim)
    z_good_embeddings = (z_good_embeddings - z_good_embeddings.mean(dim=0)
                         ) / (z_good_embeddings.std(dim=0) + 1e-5)

    total_reg_good, std_reg_good, cov_reg_good = vicreg.calculate_reg_terms(
        z_good_embeddings)
    print(
        f"Reg Terms (Good Embeddings) -> Total: {total_reg_good.item():.4f}, Std: {std_reg_good.item():.4f}, Cov: {cov_reg_good.item():.4f}")

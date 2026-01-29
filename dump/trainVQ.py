import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from anomalib.models.image.dsr.torch_model import DiscreteLatentModel

# ---------------Helper----------------------
def disable_inplace_ops(module: nn.Module):
    for name, child in module.named_children():
        # ReLU is the usual culprit
        if isinstance(child, nn.ReLU) and getattr(child, "inplace", False):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            disable_inplace_ops(child)

        # If you see other inplace activations, handle them similarly:
        # LeakyReLU, ELU, etc.
        if isinstance(child, nn.LeakyReLU) and getattr(child, "inplace", False):
            setattr(module, name, nn.LeakyReLU(negative_slope=child.negative_slope, inplace=False))

# -------------------------
# 1) Dataset (NO normalization)
# DSR docs emphasize it uses pretrained weights + its own pipeline;
# keep preprocessing simple: resize + toTensor only.
# -------------------------
def make_loaders(root: str, image_size: int, batch_size: int, num_workers: int):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # keep [0,1], no ImageNet normalization
    ])

    ds = ImageFolder(root=root, transform=tfm)   # expects subfolder per class; use "normal" as class
    # If your folder is train/normal/*.png, set root="DATA_ROOT/train" so class="normal"

    # Simple split
    n = len(ds)
    n_val = max(1, int(0.05 * n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# -------------------------
# 2) Lightning module wrapper
# -------------------------
class VQVAEPretrainModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 2e-4,
        num_hiddens: int = 128,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 64,
        num_embeddings: int = 4096,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vqvae = DiscreteLatentModel(
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        disable_inplace_ops(self.vqvae)  # avoid inplace ops for Lightning compatibility

    @staticmethod
    def _pick_recon(output: dict, x: torch.Tensor) -> torch.Tensor:
        """
        DiscreteLatentModel.forward returns a dict, but key names can vary by version.
        We pick the tensor that looks like an image reconstruction: same shape as input.
        """
        for v in output.values():
            if isinstance(v, torch.Tensor) and v.shape == x.shape:
                return v
        raise RuntimeError(f"Could not find recon in output keys: {list(output.keys())}")

    def training_step(self, batch, batch_idx):
        x, _ = batch  # ImageFolder returns (image, class_idx)
        out = self.vqvae(x)
        x_hat = self._pick_recon(out, x)

        # Reconstruction loss (start with L1; you can try L2 too)
        recon_loss = F.l1_loss(x_hat, x)

        # NOTE:
        # anomalib's VectorQuantizer docs show forward() returns only a Tensor
        # (no vq_loss returned), so this baseline trains the autoencoder path.
        # If you want a full VQ-VAE loss (commitment/codebook), see notes below.
        loss = recon_loss

        self.log("train/recon_l1", recon_loss, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out = self.vqvae(x)
        x_hat = self._pick_recon(out, x)
        recon_loss = F.l1_loss(x_hat, x)
        self.log("val/recon_l1", recon_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    DATA_ROOT = "data/configA/train"  # should contain subfolder "normal/"
    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    LR = 2e-4
    OUT_DIR = Path("weights_vqvae_pretrain")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(
        root=DATA_ROOT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = VQVAEPretrainModule(
        lr=LR,
        num_embeddings=4096,
        embedding_dim=128,
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
    )

    ckpt = ModelCheckpoint(
        dirpath=str(OUT_DIR),
        filename="vqvae-wafer-{epoch:03d}-{val_recon_l1:.4f}",
        monitor="val/recon_l1",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",   # <-- change from 16-mixed
        max_epochs=150,
        callbacks=[ckpt],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Saved checkpoints to:", OUT_DIR)


if __name__ == "__main__":
    main()

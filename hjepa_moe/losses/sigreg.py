"""SIGReg standalone — re-exports from vicreg module for clean imports."""
from hjepa_moe.losses.vicreg import SIGRegLoss, VICRegLoss, InfoNCELoss
__all__ = ["SIGRegLoss", "VICRegLoss", "InfoNCELoss"]

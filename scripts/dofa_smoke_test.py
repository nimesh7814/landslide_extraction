import torch

from dofa_model import DOFAUNet
from config import DOFA_WAVELENGTHS, DOFA_CHECKPOINT_DIR, DOFA_VARIANT

for dataset_name, wavelengths in DOFA_WAVELENGTHS.items():
    in_channels = len(wavelengths)

    print(f"\n=== {dataset_name} ({in_channels} channels, wavelengths={wavelengths}) ===")

    model = DOFAUNet(
        wavelengths=wavelengths,
        checkpoint_dir=DOFA_CHECKPOINT_DIR,
        variant=DOFA_VARIANT,
        out_channels=1,
        freeze_backbone=True
    )

    dummy_input = torch.zeros(1, in_channels, 512, 512)
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 1, 512, 512), "Unexpected output shape!"

print("\nAll 4 configs passed.")

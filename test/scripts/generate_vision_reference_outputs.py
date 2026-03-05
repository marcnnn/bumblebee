"""
Generate reference outputs for 4 vision models: SwinV2, Mask2Former, PVT, SAM.

This script runs forward passes through tiny-random HuggingFace models and prints
the output values needed for assert_all_close checks in the Bumblebee test suite.

Usage:
    pip install transformers torch numpy
    python test/scripts/generate_vision_reference_outputs.py

After running, copy the printed tensor values into the corresponding test files
under test/bumblebee/vision/.

Note: Bumblebee uses channels-last format (NHWC) while HuggingFace uses
channels-first (NCHW). The pixel_values input is created in NCHW for PyTorch,
but the output comparison values are printed as-is since hidden states and logits
don't depend on spatial layout conventions.
"""

import json
import sys

try:
    import torch
    import numpy as np
except ImportError:
    print("Please install: pip install transformers torch numpy")
    sys.exit(1)


def fmt_tensor(tensor, precision=4):
    """Format a tensor as an Elixir-compatible Nx.tensor literal."""
    arr = tensor.detach().cpu().numpy()

    def _fmt(a, depth=0):
        if a.ndim == 0:
            return f"{a.item():.{precision}f}"
        items = ", ".join(_fmt(a[i], depth + 1) for i in range(a.shape[0]))
        return f"[{items}]"

    return _fmt(arr)


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def generate_swin_v2():
    """Generate reference outputs for SwinV2."""
    from transformers import AutoModel, AutoModelForImageClassification

    print_section("SwinV2 :base")
    model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-Swinv2Model")
    model.eval()

    # NCHW input: batch=1, channels=3, height=32, width=32
    pixel_values = torch.full((1, 3, 32, 32), 0.5)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    hs = outputs.last_hidden_state
    ps = outputs.pooler_output
    print(f"hidden_state shape: {list(hs.shape)}")
    print(f"pooled_state shape: {list(ps.shape)}")
    print()
    print("# Copy into swin_v2_test.exs :base test:")
    print(f"# hidden_state[[.., 1..3, 1..3]]:")
    print(f"#   Nx.tensor({fmt_tensor(hs[0, 1:4, 1:4])})")
    print(f"# pooled_state[[.., 1..3]]:")
    print(f"#   Nx.tensor({fmt_tensor(ps[0, 1:4])})")

    print_section("SwinV2 :for_image_classification")
    model = AutoModelForImageClassification.from_pretrained(
        "hf-internal-testing/tiny-random-Swinv2ForImageClassification"
    )
    model.eval()

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    logits = outputs.logits
    print(f"logits shape: {list(logits.shape)}")
    print()
    print("# Copy into swin_v2_test.exs :for_image_classification test:")
    print(f"# logits:")
    print(f"#   Nx.tensor({fmt_tensor(logits)})")


def generate_pvt():
    """Generate reference outputs for PVT (PvtV2)."""
    from transformers import AutoModel, AutoModelForImageClassification

    print_section("PVT :base")
    model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-PvtV2Model")
    model.eval()

    pixel_values = torch.full((1, 3, 224, 224), 0.5)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    hs = outputs.last_hidden_state
    print(f"hidden_state shape: {list(hs.shape)}")
    print()
    print("# Copy into pvt_test.exs :base test:")
    print(f"# hidden_state[[.., 1..3, 1..3]]:")
    print(f"#   Nx.tensor({fmt_tensor(hs[0, 1:4, 1:4])})")

    print_section("PVT :for_image_classification")
    model = AutoModelForImageClassification.from_pretrained(
        "hf-internal-testing/tiny-random-PvtV2ForImageClassification"
    )
    model.eval()

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    logits = outputs.logits
    print(f"logits shape: {list(logits.shape)}")
    print()
    print("# Copy into pvt_test.exs :for_image_classification test:")
    print(f"# logits:")
    print(f"#   Nx.tensor({fmt_tensor(logits)})")


def generate_sam():
    """Generate reference outputs for SAM."""
    from transformers import SamModel

    print_section("SAM :base")
    model = SamModel.from_pretrained("hf-internal-testing/tiny-random-SamModel")
    model.eval()

    pixel_values = torch.full((1, 3, 64, 64), 0.5)

    with torch.no_grad():
        # SAM's forward normally expects prompts, but we can get image embeddings
        # by calling the vision encoder directly
        image_embeddings = model.get_image_embeddings(pixel_values)

    print(f"image_embeddings shape: {list(image_embeddings.shape)}")
    print()
    # SAM output is (batch, channels, h, w) in PyTorch but (batch, h, w, channels) in Bumblebee
    # For comparing values, we transpose to NHWC to match Bumblebee's output
    ie_nhwc = image_embeddings.permute(0, 2, 3, 1)
    print("# Copy into sam_test.exs :base test (values in NHWC layout):")
    print(f"# hidden_state[[.., 1..3, 1..3, 1..3]]:")
    print(f"#   Nx.tensor({fmt_tensor(ie_nhwc[0, 1:4, 1:4, 1:4])})")


def generate_mask2former():
    """Generate reference outputs for Mask2Former."""
    from transformers import AutoModel, Mask2FormerForUniversalSegmentation

    print_section("Mask2Former :base")
    model = AutoModel.from_pretrained(
        "hf-internal-testing/tiny-random-Mask2FormerModel"
    )
    model.eval()

    pixel_values = torch.full((1, 3, 224, 224), 0.5)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    hs = outputs.transformer_decoder_last_hidden_state
    print(f"hidden_state shape: {list(hs.shape)}")
    print()
    print("# Copy into mask2_former_test.exs :base test:")
    print(f"# hidden_state[[.., 0..2, 0..2]]:")
    print(f"#   Nx.tensor({fmt_tensor(hs[0, :3, :3])})")

    print_section("Mask2Former :for_instance_segmentation")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "hf-internal-testing/tiny-random-Mask2FormerForUniversalSegmentation"
    )
    model.eval()

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    logits = outputs.class_queries_logits
    print(f"class_queries_logits shape: {list(logits.shape)}")
    print()
    print("# Copy into mask2_former_test.exs :for_instance_segmentation test:")
    print(f"# logits[[.., 0..2, 0..2]]:")
    print(f"#   Nx.tensor({fmt_tensor(logits[0, :3, :3])})")


def main():
    print("=" * 60)
    print("  Vision Model Reference Output Generator")
    print("  for Bumblebee test suite assert_all_close checks")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")

    import transformers

    print(f"Transformers version: {transformers.__version__}")

    errors = []

    for name, fn in [
        ("SwinV2", generate_swin_v2),
        ("PVT", generate_pvt),
        ("SAM", generate_sam),
        ("Mask2Former", generate_mask2former),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"\nERROR generating {name}: {e}")
            import traceback

            traceback.print_exc()
            errors.append(name)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    if errors:
        print(f"Failed: {', '.join(errors)}")
    else:
        print("All 4 models generated successfully.")
    print(
        "\nCopy the printed Nx.tensor values into the corresponding test files"
    )
    print(
        "and replace the TODO comments with the actual assert_all_close calls."
    )


if __name__ == "__main__":
    main()

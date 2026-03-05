"""
Compare HuggingFace Transformers model outputs with Bumblebee implementations.

This script runs one forward pass through each model architecture using
tiny-random models from HuggingFace, captures the outputs, and saves them
as JSON files that can be loaded by the Bumblebee test suite for comparison.

Usage:
    pip install transformers torch numpy
    python test/scripts/compare_huggingface_outputs.py

The script generates JSON output files in test/fixtures/huggingface_outputs/
"""

import json
import os
import sys
import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
    from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
    from transformers import AutoModelForMaskedLM, AutoModelForMultipleChoice
    from transformers import AutoConfig
except ImportError:
    print("Please install: pip install transformers torch numpy")
    sys.exit(1)


def to_list(tensor):
    """Convert a torch tensor to a nested list for JSON serialization."""
    return tensor.detach().cpu().numpy().tolist()


def create_text_inputs(seq_len=10):
    """Create standard text inputs for testing."""
    input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def run_model(model_class, model_name, inputs, extra_key=None):
    """Run a model and capture outputs."""
    try:
        model = model_class.from_pretrained(model_name)
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs)

        result = {
            "model_name": model_name,
            "model_class": model_class.__name__,
            "input_ids": to_list(inputs.get("input_ids", torch.tensor([]))),
        }

        # Capture different output types based on the model
        if hasattr(outputs, "last_hidden_state"):
            hs = outputs.last_hidden_state
            result["hidden_state_shape"] = list(hs.shape)
            result["hidden_state_slice"] = to_list(hs[..., :3, :3])

        if hasattr(outputs, "logits"):
            logits = outputs.logits
            result["logits_shape"] = list(logits.shape)
            result["logits_slice"] = to_list(logits[..., :3, :3] if logits.dim() >= 3 else logits)

        if hasattr(outputs, "start_logits"):
            result["start_logits_shape"] = list(outputs.start_logits.shape)
            result["start_logits_slice"] = to_list(outputs.start_logits[..., :3])
            result["end_logits_shape"] = list(outputs.end_logits.shape)
            result["end_logits_slice"] = to_list(outputs.end_logits[..., :3])

        return result

    except Exception as e:
        print(f"  WARNING: Failed to run {model_name}: {e}")
        return None


def test_electra_models():
    """Test ELECTRA model architectures."""
    print("\n=== ELECTRA Models ===")
    results = []
    inputs = create_text_inputs()

    models = [
        (AutoModel, "hf-internal-testing/tiny-random-ElectraModel", "base"),
        (AutoModelForMaskedLM, "hf-internal-testing/tiny-random-ElectraForMaskedLM", "mlm"),
        (AutoModelForSequenceClassification, "hf-internal-testing/tiny-random-ElectraForSequenceClassification", "seq_cls"),
        (AutoModelForTokenClassification, "hf-internal-testing/tiny-random-ElectraForTokenClassification", "tok_cls"),
        (AutoModelForQuestionAnswering, "hf-internal-testing/tiny-random-ElectraForQuestionAnswering", "qa"),
    ]

    for model_class, name, arch in models:
        print(f"  Running {arch}...")
        result = run_model(model_class, name, inputs)
        if result:
            result["architecture"] = arch
            results.append(result)
            print(f"    OK - output shape: {result.get('hidden_state_shape') or result.get('logits_shape')}")

    return results


def test_deberta_v2_models():
    """Test DeBERTa-v2 model architectures."""
    print("\n=== DeBERTa-v2 Models ===")
    results = []
    inputs = create_text_inputs()

    models = [
        (AutoModel, "hf-internal-testing/tiny-random-DebertaV2Model", "base"),
        (AutoModelForMaskedLM, "hf-internal-testing/tiny-random-DebertaV2ForMaskedLM", "mlm"),
        (AutoModelForSequenceClassification, "hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification", "seq_cls"),
        (AutoModelForTokenClassification, "hf-internal-testing/tiny-random-DebertaV2ForTokenClassification", "tok_cls"),
        (AutoModelForQuestionAnswering, "hf-internal-testing/tiny-random-DebertaV2ForQuestionAnswering", "qa"),
    ]

    for model_class, name, arch in models:
        print(f"  Running {arch}...")
        result = run_model(model_class, name, inputs)
        if result:
            result["architecture"] = arch
            results.append(result)
            print(f"    OK - output shape: {result.get('hidden_state_shape') or result.get('logits_shape')}")

    return results


def test_falcon_models():
    """Test Falcon model architectures."""
    print("\n=== Falcon Models ===")
    results = []
    inputs = create_text_inputs()

    models = [
        (AutoModel, "hf-internal-testing/tiny-random-FalconModel", "base"),
        (AutoModelForCausalLM, "hf-internal-testing/tiny-random-FalconForCausalLM", "causal_lm"),
        (AutoModelForSequenceClassification, "hf-internal-testing/tiny-random-FalconForSequenceClassification", "seq_cls"),
        (AutoModelForTokenClassification, "hf-internal-testing/tiny-random-FalconForTokenClassification", "tok_cls"),
    ]

    for model_class, name, arch in models:
        print(f"  Running {arch}...")
        result = run_model(model_class, name, inputs)
        if result:
            result["architecture"] = arch
            results.append(result)
            print(f"    OK - output shape: {result.get('hidden_state_shape') or result.get('logits_shape')}")

    return results


def test_opt_models():
    """Test OPT model architectures."""
    print("\n=== OPT Models ===")
    results = []
    inputs = create_text_inputs()

    models = [
        (AutoModel, "hf-internal-testing/tiny-random-OPTModel", "base"),
        (AutoModelForCausalLM, "hf-internal-testing/tiny-random-OPTForCausalLM", "causal_lm"),
        (AutoModelForSequenceClassification, "hf-internal-testing/tiny-random-OPTForSequenceClassification", "seq_cls"),
        (AutoModelForQuestionAnswering, "hf-internal-testing/tiny-random-OPTForQuestionAnswering", "qa"),
    ]

    for model_class, name, arch in models:
        print(f"  Running {arch}...")
        result = run_model(model_class, name, inputs)
        if result:
            result["architecture"] = arch
            results.append(result)
            print(f"    OK - output shape: {result.get('hidden_state_shape') or result.get('logits_shape')}")

    return results


def test_qwen2_models():
    """Test Qwen2 model architectures."""
    print("\n=== Qwen2 Models ===")
    results = []
    inputs = create_text_inputs()

    models = [
        (AutoModel, "hf-internal-testing/tiny-random-Qwen2Model", "base"),
        (AutoModelForCausalLM, "hf-internal-testing/tiny-random-Qwen2ForCausalLM", "causal_lm"),
        (AutoModelForSequenceClassification, "hf-internal-testing/tiny-random-Qwen2ForSequenceClassification", "seq_cls"),
        (AutoModelForTokenClassification, "hf-internal-testing/tiny-random-Qwen2ForTokenClassification", "tok_cls"),
    ]

    for model_class, name, arch in models:
        print(f"  Running {arch}...")
        result = run_model(model_class, name, inputs)
        if result:
            result["architecture"] = arch
            results.append(result)
            print(f"    OK - output shape: {result.get('hidden_state_shape') or result.get('logits_shape')}")

    return results


def test_wav2vec2_models():
    """Test Wav2Vec2 model architectures."""
    print("\n=== Wav2Vec2 Models ===")
    results = []

    # Audio input - 1 second of silence at 16kHz
    audio_inputs = {"input_values": torch.zeros(1, 16000)}

    models = [
        (AutoModel, "hf-internal-testing/tiny-random-Wav2Vec2Model", "base"),
    ]

    for model_class, name, arch in models:
        print(f"  Running {arch}...")
        try:
            model = model_class.from_pretrained(name)
            model.eval()

            with torch.no_grad():
                outputs = model(**audio_inputs)

            result = {
                "model_name": name,
                "model_class": model_class.__name__,
                "architecture": arch,
            }

            if hasattr(outputs, "last_hidden_state"):
                hs = outputs.last_hidden_state
                result["hidden_state_shape"] = list(hs.shape)
                result["hidden_state_slice"] = to_list(hs[..., :3, :3])

            results.append(result)
            print(f"    OK - output shape: {result.get('hidden_state_shape')}")

        except Exception as e:
            print(f"    WARNING: Failed to run {name}: {e}")

    return results


def main():
    print("=" * 60)
    print("HuggingFace Transformers Output Comparison Script")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {__import__('transformers').__version__}")

    all_results = {}

    # Run all model tests
    all_results["electra"] = test_electra_models()
    all_results["deberta_v2"] = test_deberta_v2_models()
    all_results["falcon"] = test_falcon_models()
    all_results["opt"] = test_opt_models()
    all_results["qwen2"] = test_qwen2_models()
    all_results["wav2vec2"] = test_wav2vec2_models()

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures", "huggingface_outputs")
    os.makedirs(output_dir, exist_ok=True)

    for model_name, results in all_results.items():
        output_path = os.path.join(output_dir, f"{model_name}_outputs.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {model_name} results to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total = sum(len(r) for r in all_results.values())
    print(f"Total models tested: {total}")
    for model_name, results in all_results.items():
        print(f"  {model_name}: {len(results)} architectures")

    print("\nDone! Use these JSON files to verify Bumblebee model outputs.")


if __name__ == "__main__":
    main()

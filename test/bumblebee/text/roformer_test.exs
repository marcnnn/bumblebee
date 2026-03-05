defmodule Bumblebee.Text.RoformerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-RoFormerModel"})

    assert %Bumblebee.Text.Roformer{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, 10, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_masked_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-RoFormerForMaskedLM"})

    assert %Bumblebee.Text.Roformer{architecture: :for_masked_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, 10, _vocab_size} = Nx.shape(outputs.logits)
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-RoFormerForSequenceClassification"}
             )

    assert %Bumblebee.Text.Roformer{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Text.Roformer{
        architecture: :base,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 64,
        num_labels: 3
      }

      for arch <- Bumblebee.Text.Roformer.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

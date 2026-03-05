defmodule Bumblebee.Text.BloomTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-BloomModel"})

    assert %Bumblebee.Text.Bloom{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, 10, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-BloomForCausalLM"})

    assert %Bumblebee.Text.Bloom{architecture: :for_causal_language_modeling} = spec

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
               {:hf, "hf-internal-testing/tiny-random-BloomForSequenceClassification"}
             )

    assert %Bumblebee.Text.Bloom{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}
  end

  test ":for_token_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BloomForTokenClassification"}
             )

    assert %Bumblebee.Text.Bloom{architecture: :for_token_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 2}
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Text.Bloom{
        architecture: :base,
        vocab_size: 100,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        num_labels: 3
      }

      for arch <- Bumblebee.Text.Bloom.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Text.Bloom{}

      data = %{
        "vocab_size" => 250880,
        "hidden_size" => 2560,
        "n_layer" => 30,
        "n_head" => 32,
        "layer_norm_epsilon" => 1.0e-5,
        "initializer_range" => 0.02,
        "hidden_dropout" => 0.0,
        "attention_dropout" => 0.0
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 250880
      assert loaded_spec.hidden_size == 2560
      assert loaded_spec.num_blocks == 30
      assert loaded_spec.num_attention_heads == 32
    end
  end
end

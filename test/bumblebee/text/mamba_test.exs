defmodule Bumblebee.Text.MambaTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MambaModel"})

    assert %Bumblebee.Text.Mamba{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, 10, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MambaForCausalLM"})

    assert %Bumblebee.Text.Mamba{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, 10, _vocab_size} = Nx.shape(outputs.logits)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Text.Mamba{
        architecture: :base,
        vocab_size: 100,
        hidden_size: 16,
        num_blocks: 2,
        intermediate_size: 32
      }

      for arch <- Bumblebee.Text.Mamba.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Text.Mamba{}

      data = %{
        "vocab_size" => 50280,
        "d_model" => 768,
        "n_layer" => 24,
        "d_inner" => 1536,
        "d_state" => 16,
        "d_conv" => 4
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 50280
      assert loaded_spec.hidden_size == 768
      assert loaded_spec.num_blocks == 24
    end
  end
end

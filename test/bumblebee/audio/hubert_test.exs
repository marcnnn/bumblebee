defmodule Bumblebee.Audio.HubertTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-HubertModel"})

    assert %Bumblebee.Audio.Hubert{architecture: :base} = spec

    inputs = %{
      "input_values" => Nx.broadcast(0.0, {1, 16_000})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _seq_length, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_ctc" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-HubertForCTC"})

    assert %Bumblebee.Audio.Hubert{architecture: :for_ctc} = spec

    inputs = %{
      "input_values" => Nx.broadcast(0.0, {1, 16_000})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _seq_length, _vocab_size} = Nx.shape(outputs.logits)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Audio.Hubert{
        architecture: :base,
        vocab_size: 32,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 64,
        num_labels: 3
      }

      for arch <- Bumblebee.Audio.Hubert.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

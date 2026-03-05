defmodule Bumblebee.Text.Qwen2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-Qwen2Model"})

    assert %Bumblebee.Text.Qwen2{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, 10, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-Qwen2ForCausalLM"})

    assert %Bumblebee.Text.Qwen2{architecture: :for_causal_language_modeling} = spec

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
               {:hf, "hf-internal-testing/tiny-random-Qwen2ForSequenceClassification"}
             )

    assert %Bumblebee.Text.Qwen2{architecture: :for_sequence_classification} = spec

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
               {:hf, "hf-internal-testing/tiny-random-Qwen2ForTokenClassification"}
             )

    assert %Bumblebee.Text.Qwen2{architecture: :for_token_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 2}
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Text.Qwen2{
        architecture: :base,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        intermediate_size: 64,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        num_labels: 3
      }

      for arch <- Bumblebee.Text.Qwen2.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Text.Qwen2{}

      data = %{
        "vocab_size" => 151936,
        "max_position_embeddings" => 32768,
        "hidden_size" => 4096,
        "num_hidden_layers" => 32,
        "num_attention_heads" => 32,
        "num_key_value_heads" => 32,
        "intermediate_size" => 22016,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1.0e-6,
        "initializer_range" => 0.02,
        "rope_theta" => 1_000_000,
        "attention_bias" => true,
        "tie_word_embeddings" => false,
        "sliding_window" => 32768
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 151936
      assert loaded_spec.hidden_size == 4096
      assert loaded_spec.num_blocks == 32
      assert loaded_spec.num_key_value_heads == 32
      assert loaded_spec.use_attention_bias == true
      assert loaded_spec.attention_window_size == 32768
      assert loaded_spec.rotary_embedding_base == 1_000_000
    end
  end
end

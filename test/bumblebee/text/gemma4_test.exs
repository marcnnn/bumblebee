defmodule Bumblebee.Text.Gemma4Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Text.Gemma4{
        architecture: :base,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        attention_head_size: 8,
        intermediate_size: 64,
        num_blocks: 4,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        num_labels: 3
      }

      for arch <- Bumblebee.Text.Gemma4.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end

    test "builds with alternating layer_types" do
      spec = %Bumblebee.Text.Gemma4{
        architecture: :base,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        attention_head_size: 8,
        intermediate_size: 64,
        num_blocks: 4,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        layer_types: [:sliding_attention, :full_attention, :sliding_attention, :full_attention]
      }

      model = Bumblebee.build_model(spec)
      assert %Axon{} = model
    end

    test "produces correct output shapes for :base" do
      spec = %Bumblebee.Text.Gemma4{
        architecture: :base,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        attention_head_size: 8,
        intermediate_size: 64,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2
      }

      model = Bumblebee.build_model(spec)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1]])
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(inputs, Axon.ModelState.empty())
      outputs = predict_fn.(params, inputs)

      assert {1, 5, 16} = Nx.shape(outputs.hidden_state)
    end

    test "produces correct output shapes for :for_causal_language_modeling" do
      spec = %Bumblebee.Text.Gemma4{
        architecture: :for_causal_language_modeling,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        attention_head_size: 8,
        intermediate_size: 64,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2
      }

      model = Bumblebee.build_model(spec)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1]])
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(inputs, Axon.ModelState.empty())
      outputs = predict_fn.(params, inputs)

      assert {1, 5, 100} = Nx.shape(outputs.logits)
    end

    test "produces correct output shapes for :for_sequence_classification" do
      spec = %Bumblebee.Text.Gemma4{
        architecture: :for_sequence_classification,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        attention_head_size: 8,
        intermediate_size: 64,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        num_labels: 3,
        pad_token_id: 0
      }

      model = Bumblebee.build_model(spec)

      inputs = %{
        "input_ids" => Nx.tensor([[1, 2, 3, 4, 0]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 0]])
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(inputs, Axon.ModelState.empty())
      outputs = predict_fn.(params, inputs)

      assert {1, 3} = Nx.shape(outputs.logits)
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Text.Gemma4{}

      data = %{
        "vocab_size" => 262_144,
        "max_position_embeddings" => 131_072,
        "hidden_size" => 2304,
        "num_hidden_layers" => 30,
        "num_attention_heads" => 8,
        "num_key_value_heads" => 4,
        "head_dim" => 256,
        "intermediate_size" => 9216,
        "hidden_activation" => "gelu_pytorch_tanh",
        "attention_bias" => false,
        "sliding_window" => 512,
        "rope_theta" => 10_000,
        "rms_norm_eps" => 1.0e-6,
        "initializer_range" => 0.02
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 262_144
      assert loaded_spec.hidden_size == 2304
      assert loaded_spec.num_blocks == 30
      assert loaded_spec.num_attention_heads == 8
      assert loaded_spec.num_key_value_heads == 4
      assert loaded_spec.attention_head_size == 256
      assert loaded_spec.intermediate_size == 9216
      assert loaded_spec.use_attention_bias == false
      assert loaded_spec.sliding_window == 512
    end

    test "loads config with layer_types" do
      spec = %Bumblebee.Text.Gemma4{}

      data = %{
        "vocab_size" => 262_144,
        "max_position_embeddings" => 131_072,
        "hidden_size" => 2304,
        "num_hidden_layers" => 4,
        "num_attention_heads" => 8,
        "num_key_value_heads" => 4,
        "head_dim" => 256,
        "intermediate_size" => 9216,
        "hidden_activation" => "gelu_pytorch_tanh",
        "attention_bias" => false,
        "sliding_window" => 512,
        "layer_types" => [
          "sliding_attention",
          "full_attention",
          "sliding_attention",
          "full_attention"
        ],
        "rope_theta" => 10_000,
        "rms_norm_eps" => 1.0e-6,
        "initializer_range" => 0.02
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.layer_types == [
               :sliding_attention,
               :full_attention,
               :sliding_attention,
               :full_attention
             ]
    end

    test "loads config with rope scaling" do
      spec = %Bumblebee.Text.Gemma4{}

      data = %{
        "vocab_size" => 262_144,
        "max_position_embeddings" => 131_072,
        "hidden_size" => 2304,
        "num_hidden_layers" => 30,
        "num_attention_heads" => 8,
        "num_key_value_heads" => 4,
        "head_dim" => 256,
        "intermediate_size" => 9216,
        "hidden_activation" => "gelu_pytorch_tanh",
        "attention_bias" => false,
        "sliding_window" => 512,
        "rope_theta" => 10_000,
        "rope_scaling" => %{"type" => "linear", "factor" => 2.0},
        "rms_norm_eps" => 1.0e-6,
        "initializer_range" => 0.02
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.rotary_embedding_scaling_strategy == %{type: :linear, factor: 2.0}
    end
  end
end

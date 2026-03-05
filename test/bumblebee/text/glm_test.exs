defmodule Bumblebee.Text.GlmTest do
  use ExUnit.Case, async: true

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Text.Glm{
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

      for arch <- Bumblebee.Text.Glm.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Text.Glm{}

      data = %{
        "vocab_size" => 151552,
        "max_position_embeddings" => 131072,
        "hidden_size" => 4096,
        "num_hidden_layers" => 40,
        "num_attention_heads" => 32,
        "num_key_value_heads" => 8,
        "intermediate_size" => 13696,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1.0e-5,
        "initializer_range" => 0.02,
        "rope_theta" => 10_000,
        "attention_bias" => true,
        "tie_word_embeddings" => false
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 151552
      assert loaded_spec.hidden_size == 4096
      assert loaded_spec.num_blocks == 40
      assert loaded_spec.num_key_value_heads == 8
      assert loaded_spec.intermediate_size == 13696
      assert loaded_spec.use_qkv_bias == true
      assert loaded_spec.rotary_embedding_base == 10_000
      assert loaded_spec.layer_norm_epsilon == 1.0e-5
    end

    test "loads config with head_dim" do
      spec = %Bumblebee.Text.Glm{}

      data = %{
        "vocab_size" => 151552,
        "max_position_embeddings" => 131072,
        "hidden_size" => 4096,
        "num_hidden_layers" => 40,
        "num_attention_heads" => 32,
        "num_key_value_heads" => 8,
        "head_dim" => 128,
        "intermediate_size" => 13696,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1.0e-5,
        "initializer_range" => 0.02,
        "rope_theta" => 10_000,
        "tie_word_embeddings" => false
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.attention_head_size == 128
    end

    test "loads config with attention_bias set to false" do
      spec = %Bumblebee.Text.Glm{}

      data = %{
        "vocab_size" => 151552,
        "max_position_embeddings" => 131072,
        "hidden_size" => 4096,
        "num_hidden_layers" => 40,
        "num_attention_heads" => 32,
        "num_key_value_heads" => 8,
        "intermediate_size" => 13696,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1.0e-5,
        "initializer_range" => 0.02,
        "rope_theta" => 10_000,
        "attention_bias" => false,
        "tie_word_embeddings" => false
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.use_qkv_bias == false
    end
  end
end

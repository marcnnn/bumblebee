defmodule Bumblebee.Text.KimiK2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Text.KimiK2{
        architecture: :base,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        intermediate_size: 64,
        moe_intermediate_size: 32,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        num_labels: 3
      }

      for arch <- Bumblebee.Text.KimiK2.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Text.KimiK2{}

      data = %{
        "vocab_size" => 163_840,
        "max_position_embeddings" => 262_144,
        "hidden_size" => 7168,
        "num_hidden_layers" => 61,
        "num_attention_heads" => 64,
        "num_key_value_heads" => 64,
        "head_dim" => 64,
        "intermediate_size" => 18432,
        "moe_intermediate_size" => 2048,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1.0e-6,
        "rope_theta" => 10000,
        "kv_lora_rank" => 512,
        "q_lora_rank" => 1536,
        "n_routed_experts" => 384,
        "num_experts_per_tok" => 8,
        "n_shared_experts" => 1,
        "first_k_dense_replace" => 1
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 163_840
      assert loaded_spec.hidden_size == 7168
      assert loaded_spec.num_blocks == 61
      assert loaded_spec.num_attention_heads == 64
      assert loaded_spec.num_key_value_heads == 64
      assert loaded_spec.attention_head_size == 64
      assert loaded_spec.intermediate_size == 18432
      assert loaded_spec.moe_intermediate_size == 2048
      assert loaded_spec.kv_lora_rank == 512
      assert loaded_spec.q_lora_rank == 1536
      assert loaded_spec.num_experts == 384
      assert loaded_spec.num_experts_per_token == 8
      assert loaded_spec.num_shared_experts == 1
      assert loaded_spec.first_k_dense_replace == 1
    end

    test "loads config with yarn rope scaling" do
      spec = %Bumblebee.Text.KimiK2{}

      data = %{
        "vocab_size" => 163_840,
        "max_position_embeddings" => 262_144,
        "hidden_size" => 7168,
        "num_hidden_layers" => 61,
        "num_attention_heads" => 64,
        "num_key_value_heads" => 64,
        "head_dim" => 64,
        "intermediate_size" => 18432,
        "moe_intermediate_size" => 2048,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1.0e-6,
        "rope_theta" => 10000,
        "kv_lora_rank" => 512,
        "q_lora_rank" => 1536,
        "n_routed_experts" => 384,
        "num_experts_per_tok" => 8,
        "n_shared_experts" => 1,
        "first_k_dense_replace" => 1,
        "rope_scaling" => %{
          "type" => "yarn",
          "factor" => 40,
          "original_max_position_embeddings" => 4096
        }
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.rotary_embedding_scaling_strategy == %{
               type: :yarn,
               factor: 40,
               original_max_positions: 4096
             }
    end
  end
end

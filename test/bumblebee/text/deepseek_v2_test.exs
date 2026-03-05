defmodule Bumblebee.Text.DeepseekV2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Text.DeepseekV2{
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

      for arch <- Bumblebee.Text.DeepseekV2.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Text.DeepseekV2{}

      data = %{
        "vocab_size" => 102400,
        "max_position_embeddings" => 4096,
        "hidden_size" => 5120,
        "num_hidden_layers" => 60,
        "num_attention_heads" => 128,
        "num_key_value_heads" => 128,
        "intermediate_size" => 12288,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1.0e-6,
        "rope_theta" => 10000,
        "kv_lora_rank" => 512,
        "q_lora_rank" => 1536,
        "n_routed_experts" => 64,
        "num_experts_per_tok" => 6,
        "n_shared_experts" => 2,
        "first_k_dense_replace" => 1
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 102400
      assert loaded_spec.hidden_size == 5120
      assert loaded_spec.num_blocks == 60
      assert loaded_spec.num_attention_heads == 128
      assert loaded_spec.kv_lora_rank == 512
      assert loaded_spec.q_lora_rank == 1536
      assert loaded_spec.num_experts == 64
      assert loaded_spec.num_experts_per_token == 6
      assert loaded_spec.num_shared_experts == 2
      assert loaded_spec.first_k_dense_replace == 1
    end

    test "loads config with yarn rope scaling" do
      spec = %Bumblebee.Text.DeepseekV2{}

      data = %{
        "vocab_size" => 102400,
        "max_position_embeddings" => 4096,
        "hidden_size" => 5120,
        "num_hidden_layers" => 60,
        "num_attention_heads" => 128,
        "num_key_value_heads" => 128,
        "intermediate_size" => 12288,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1.0e-6,
        "rope_theta" => 10000,
        "kv_lora_rank" => 512,
        "q_lora_rank" => 1536,
        "n_routed_experts" => 64,
        "num_experts_per_tok" => 6,
        "n_shared_experts" => 2,
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

defmodule Bumblebee.Text.FalconH1Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Text.FalconH1{
        architecture: :base,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        intermediate_size: 64,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        mamba_d_state: 8,
        mamba_d_conv: 4,
        mamba_expand: 2,
        mamba_n_heads: 4,
        mamba_d_head: 4,
        num_labels: 3
      }

      for arch <- Bumblebee.Text.FalconH1.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end

    test "runs a forward pass for :base" do
      spec = %Bumblebee.Text.FalconH1{
        architecture: :base,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        intermediate_size: 64,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        mamba_d_state: 8,
        mamba_d_conv: 4,
        mamba_expand: 2,
        mamba_n_heads: 4,
        mamba_d_head: 4
      }

      model = Bumblebee.build_model(spec)
      {init_fn, predict_fn} = Axon.build(model)

      inputs = %{
        "input_ids" => Nx.tensor([[10, 20, 30, 40, 50]])
      }

      params = init_fn.(inputs, Axon.ModelState.empty())
      outputs = predict_fn.(params, inputs)

      assert {1, 5, 16} = Nx.shape(outputs.hidden_state)
    end

    test "runs a forward pass for :for_causal_language_modeling" do
      spec = %Bumblebee.Text.FalconH1{
        architecture: :for_causal_language_modeling,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        intermediate_size: 64,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        mamba_d_state: 8,
        mamba_d_conv: 4,
        mamba_expand: 2,
        mamba_n_heads: 4,
        mamba_d_head: 4
      }

      model = Bumblebee.build_model(spec)
      {init_fn, predict_fn} = Axon.build(model)

      inputs = %{
        "input_ids" => Nx.tensor([[10, 20, 30, 40, 50]])
      }

      params = init_fn.(inputs, Axon.ModelState.empty())
      outputs = predict_fn.(params, inputs)

      assert {1, 5, 100} = Nx.shape(outputs.logits)
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Text.FalconH1{}

      data = %{
        "vocab_size" => 65024,
        "hidden_size" => 4096,
        "num_hidden_layers" => 32,
        "num_attention_heads" => 32,
        "num_key_value_heads" => 8,
        "intermediate_size" => 11008,
        "rms_norm_eps" => 1.0e-5,
        "initializer_range" => 0.02,
        "max_position_embeddings" => 8192,
        "rope_theta" => 10_000,
        "mamba_d_state" => 128,
        "mamba_d_conv" => 4,
        "mamba_expand" => 2,
        "mamba_n_heads" => 128,
        "mamba_d_head" => 64,
        "tie_word_embeddings" => false
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 65024
      assert loaded_spec.hidden_size == 4096
      assert loaded_spec.num_blocks == 32
      assert loaded_spec.num_attention_heads == 32
      assert loaded_spec.num_key_value_heads == 8
      assert loaded_spec.intermediate_size == 11008
      assert loaded_spec.mamba_d_state == 128
      assert loaded_spec.mamba_d_conv == 4
      assert loaded_spec.mamba_expand == 2
      assert loaded_spec.mamba_n_heads == 128
      assert loaded_spec.mamba_d_head == 64
    end
  end
end

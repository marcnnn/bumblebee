defmodule Bumblebee.Audio.Wav2Vec2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-Wav2Vec2Model"})

    assert %Bumblebee.Audio.Wav2Vec2{architecture: :base} = spec

    inputs = %{
      "input_values" => Nx.broadcast(0.0, {1, 16000})
    }

    outputs = Axon.predict(model, params, inputs)

    # Output shape depends on CNN feature extraction
    assert {1, _time_steps, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_ctc" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-Wav2Vec2ForCTC"})

    assert %Bumblebee.Audio.Wav2Vec2{architecture: :for_ctc} = spec

    inputs = %{
      "input_values" => Nx.broadcast(0.0, {1, 16000})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _time_steps, _vocab_size} = Nx.shape(outputs.logits)
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-Wav2Vec2ForSequenceClassification"}
             )

    assert %Bumblebee.Audio.Wav2Vec2{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_values" => Nx.broadcast(0.0, {1, 16000})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_labels} = Nx.shape(outputs.logits)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Audio.Wav2Vec2{
        architecture: :base,
        vocab_size: 32,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 64,
        num_labels: 3,
        feature_extractor_num_channels: [32, 32],
        feature_extractor_kernel_sizes: [10, 3],
        feature_extractor_strides: [5, 2]
      }

      for arch <- Bumblebee.Audio.Wav2Vec2.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Audio.Wav2Vec2{}

      data = %{
        "vocab_size" => 32,
        "hidden_size" => 768,
        "num_hidden_layers" => 12,
        "num_attention_heads" => 12,
        "intermediate_size" => 3072,
        "hidden_act" => "gelu",
        "hidden_dropout" => 0.1,
        "attention_dropout" => 0.1,
        "layer_norm_eps" => 1.0e-5,
        "initializer_range" => 0.02,
        "conv_dim" => [512, 512, 512, 512, 512, 512, 512],
        "conv_kernel" => [10, 3, 3, 3, 3, 2, 2],
        "conv_stride" => [5, 2, 2, 2, 2, 2, 2],
        "feat_proj_dropout" => 0.0,
        "num_conv_pos_embedding_groups" => 16,
        "conv_pos_kernel_size" => 128
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 32
      assert loaded_spec.hidden_size == 768
      assert loaded_spec.num_blocks == 12
      assert loaded_spec.feature_extractor_num_channels == [512, 512, 512, 512, 512, 512, 512]
    end
  end
end

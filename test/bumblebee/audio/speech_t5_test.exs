defmodule Bumblebee.Audio.SpeechT5Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "model structure" do
    test "builds the Axon graph for :base architecture" do
      spec = %Bumblebee.Audio.SpeechT5{
        architecture: :base,
        vocab_size: 81,
        hidden_size: 16,
        encoder_num_blocks: 2,
        decoder_num_blocks: 2,
        encoder_num_attention_heads: 2,
        decoder_num_attention_heads: 2,
        encoder_intermediate_size: 64,
        decoder_intermediate_size: 64
      }

      model = Bumblebee.build_model(spec)
      assert %Axon{} = model
    end

    test "builds the Axon graph for :for_speech_to_text architecture" do
      spec = %Bumblebee.Audio.SpeechT5{
        architecture: :for_speech_to_text,
        vocab_size: 81,
        hidden_size: 16,
        encoder_num_blocks: 2,
        decoder_num_blocks: 2,
        encoder_num_attention_heads: 2,
        decoder_num_attention_heads: 2,
        encoder_intermediate_size: 64,
        decoder_intermediate_size: 64
      }

      model = Bumblebee.build_model(spec)
      assert %Axon{} = model
    end

    test "builds the Axon graph for :for_text_to_speech architecture" do
      spec = %Bumblebee.Audio.SpeechT5{
        architecture: :for_text_to_speech,
        vocab_size: 81,
        hidden_size: 16,
        encoder_num_blocks: 2,
        decoder_num_blocks: 2,
        encoder_num_attention_heads: 2,
        decoder_num_attention_heads: 2,
        encoder_intermediate_size: 64,
        decoder_intermediate_size: 64
      }

      model = Bumblebee.build_model(spec)
      assert %Axon{} = model
    end

    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Audio.SpeechT5{
        architecture: :base,
        vocab_size: 81,
        hidden_size: 16,
        encoder_num_blocks: 2,
        decoder_num_blocks: 2,
        encoder_num_attention_heads: 2,
        decoder_num_attention_heads: 2,
        encoder_intermediate_size: 64,
        decoder_intermediate_size: 64
      }

      for arch <- Bumblebee.Audio.SpeechT5.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Audio.SpeechT5{}

      data = %{
        "vocab_size" => 81,
        "hidden_size" => 768,
        "encoder_layers" => 12,
        "decoder_layers" => 6,
        "encoder_attention_heads" => 12,
        "decoder_attention_heads" => 12,
        "encoder_ffn_dim" => 3072,
        "decoder_ffn_dim" => 3072,
        "hidden_act" => "gelu",
        "dropout" => 0.1,
        "attention_dropout" => 0.1
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 81
      assert loaded_spec.hidden_size == 768
      assert loaded_spec.encoder_num_blocks == 12
      assert loaded_spec.decoder_num_blocks == 6
      assert loaded_spec.encoder_num_attention_heads == 12
      assert loaded_spec.decoder_num_attention_heads == 12
      assert loaded_spec.encoder_intermediate_size == 3072
      assert loaded_spec.decoder_intermediate_size == 3072
      assert loaded_spec.activation == :gelu
      assert loaded_spec.dropout_rate == 0.1
      assert loaded_spec.attention_dropout_rate == 0.1
    end
  end
end

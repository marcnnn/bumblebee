defmodule Bumblebee.Audio.EncodecTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "model structure" do
    test "builds the Axon graph for :base architecture" do
      spec = %Bumblebee.Audio.Encodec{
        architecture: :base,
        hidden_size: 32,
        num_filters: 8,
        num_residual_layers: 1,
        upsampling_ratios: [4, 2],
        codebook_size: 64,
        codebook_dim: 32
      }

      model = Bumblebee.build_model(spec)
      assert %Axon{} = model
    end

    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Audio.Encodec{
        architecture: :base,
        hidden_size: 32,
        num_filters: 8,
        num_residual_layers: 1,
        upsampling_ratios: [4, 2],
        codebook_size: 64,
        codebook_dim: 32
      }

      for arch <- Bumblebee.Audio.Encodec.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Audio.Encodec{}

      data = %{
        "hidden_size" => 128,
        "num_filters" => 32,
        "num_residual_layers" => 1,
        "upsampling_ratios" => [8, 5, 4, 2],
        "codebook_size" => 1024,
        "codebook_dim" => 128
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.hidden_size == 128
      assert loaded_spec.num_filters == 32
      assert loaded_spec.num_residual_layers == 1
      assert loaded_spec.upsampling_ratios == [8, 5, 4, 2]
      assert loaded_spec.codebook_size == 1024
      assert loaded_spec.codebook_dim == 128
    end
  end
end

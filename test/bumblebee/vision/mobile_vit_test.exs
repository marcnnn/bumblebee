defmodule Bumblebee.Vision.MobileVitTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MobileViTModel"})

    assert %Bumblebee.Vision.MobileVit{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 256, 256, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.MobileVit{
        architecture: :base,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 64,
        image_size: 32,
        patch_size: 2,
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.MobileVit.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

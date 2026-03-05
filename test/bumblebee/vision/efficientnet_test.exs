defmodule Bumblebee.Vision.EfficientNetTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-EfficientNetModel"})

    assert %Bumblebee.Vision.EfficientNet{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.EfficientNet{
        architecture: :base,
        hidden_size: 16,
        image_size: 32,
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.EfficientNet.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

defmodule Bumblebee.Vision.RegNetTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-RegNetModel"})

    assert %Bumblebee.Vision.RegNet{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _h, _w, _channels} = Nx.shape(outputs.hidden_state)
    assert {1, _pooled_channels} = Nx.shape(outputs.pooled_state)
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-RegNetForImageClassification"}
             )

    assert %Bumblebee.Vision.RegNet{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_labels} = Nx.shape(outputs.logits)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.RegNet{
        architecture: :base,
        embedding_size: 16,
        hidden_sizes: [32, 64],
        depths: [1, 1],
        groups_width: 16,
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.RegNet.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

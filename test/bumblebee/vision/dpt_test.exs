defmodule Bumblebee.Vision.DptTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-DPTModel"})

    assert %Bumblebee.Vision.Dpt{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, spec.image_size, spec.image_size, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    # Neck output is a spatial feature map
    assert {1, _h, _w, _channels} = Nx.shape(outputs.hidden_state)
  end

  test ":for_depth_estimation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DPTForDepthEstimation"}
             )

    assert %Bumblebee.Vision.Dpt{architecture: :for_depth_estimation} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, spec.image_size, spec.image_size, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    # Depth map should be spatial (batch, height, width)
    assert {1, _h, _w} = Nx.shape(outputs.predicted_depth)
  end

  test ":for_semantic_segmentation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DPTForSemanticSegmentation"}
             )

    assert %Bumblebee.Vision.Dpt{architecture: :for_semantic_segmentation} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, spec.image_size, spec.image_size, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, spec.image_size, spec.image_size, _num_labels} = Nx.shape(outputs.logits)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.Dpt{
        architecture: :base,
        image_size: 64,
        patch_size: 16,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 64,
        neck_hidden_sizes: [8, 16],
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.Dpt.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

defmodule Bumblebee.Vision.YolosTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-YolosModel"})

    assert %Bumblebee.Vision.Yolos{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, spec.image_size, spec.image_size, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    # Output should include CLS + patches + detection tokens
    num_patches = div(spec.image_size, spec.patch_size) ** 2
    expected_seq_len = 1 + num_patches + spec.num_detection_tokens

    assert {1, ^expected_seq_len, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_object_detection" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-YolosForObjectDetection"}
             )

    assert %Bumblebee.Vision.Yolos{architecture: :for_object_detection} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, spec.image_size, spec.image_size, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_detection_tokens, _num_classes_plus_one} = Nx.shape(outputs.logits)
    assert {1, _num_detection_tokens, 4} = Nx.shape(outputs.pred_boxes)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.Yolos{
        architecture: :base,
        image_size: 64,
        patch_size: 16,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 64,
        num_detection_tokens: 10,
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.Yolos.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

defmodule Bumblebee.Vision.DetrTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-DetrModel"})

    assert %Bumblebee.Vision.Detr{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_queries, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_object_detection" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DetrForObjectDetection"}
             )

    assert %Bumblebee.Vision.Detr{architecture: :for_object_detection} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_queries, _num_classes_plus_one} = Nx.shape(outputs.logits)
    assert {1, _num_queries, 4} = Nx.shape(outputs.pred_boxes)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.Detr{
        architecture: :base,
        hidden_size: 16,
        encoder_num_blocks: 2,
        encoder_num_attention_heads: 2,
        encoder_intermediate_size: 64,
        decoder_num_blocks: 2,
        decoder_num_attention_heads: 2,
        decoder_intermediate_size: 64,
        num_queries: 10,
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.Detr.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

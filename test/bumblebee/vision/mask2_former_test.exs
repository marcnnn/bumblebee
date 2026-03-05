defmodule Bumblebee.Vision.Mask2FormerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-Mask2FormerModel"})

    assert %Bumblebee.Vision.Mask2Former{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_queries, _hidden_size} = Nx.shape(outputs.hidden_state)

    # TODO: Fill in reference values from generate_vision_reference_outputs.py
    # assert_all_close(
    #   outputs.hidden_state[[.., 0..2, 0..2]],
    #   Nx.tensor([[...]])
    # )
  end

  test ":for_instance_segmentation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-Mask2FormerForUniversalSegmentation"}
             )

    assert %Bumblebee.Vision.Mask2Former{architecture: :for_instance_segmentation} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_queries, _num_classes_plus_one} = Nx.shape(outputs.logits)

    # TODO: Fill in reference values from generate_vision_reference_outputs.py
    # assert_all_close(
    #   outputs.logits[[.., 0..2, 0..2]],
    #   Nx.tensor([[...]])
    # )
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.Mask2Former{
        architecture: :base,
        image_size: 64,
        hidden_size: 16,
        num_queries: 10,
        encoder_hidden_size: 16,
        encoder_num_blocks: 1,
        encoder_num_attention_heads: 2,
        encoder_intermediate_size: 32,
        decoder_hidden_size: 16,
        decoder_num_blocks: 1,
        decoder_num_attention_heads: 2,
        decoder_intermediate_size: 32,
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.Mask2Former.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

defmodule Bumblebee.Vision.PvtTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-PvtV2Model"})

    assert %Bumblebee.Vision.Pvt{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _seq_length, _hidden_size} = Nx.shape(outputs.hidden_state)

    # TODO: Fill in reference values from generate_vision_reference_outputs.py
    # assert_all_close(
    #   outputs.hidden_state[[.., 1..3, 1..3]],
    #   Nx.tensor([[...]])
    # )
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-PvtV2ForImageClassification"}
             )

    assert %Bumblebee.Vision.Pvt{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_labels} = Nx.shape(outputs.logits)

    # TODO: Fill in reference values from generate_vision_reference_outputs.py
    # assert_all_close(
    #   outputs.logits,
    #   Nx.tensor([[...]])
    # )
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.Pvt{
        architecture: :base,
        image_size: 32,
        hidden_sizes: [16, 32],
        num_blocks: [1, 1],
        num_attention_heads: [1, 2],
        patch_sizes: [7, 3],
        strides: [4, 2],
        intermediate_ratios: [4, 4],
        sr_ratios: [4, 2],
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.Pvt.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

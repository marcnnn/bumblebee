defmodule Bumblebee.Vision.SegformerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-SegformerModel"})

    assert %Bumblebee.Vision.Segformer{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 512, 512, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _seq_length, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-SegformerForImageClassification"}
             )

    assert %Bumblebee.Vision.Segformer{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 512, 512, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_classes} = Nx.shape(outputs.logits)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.Segformer{
        architecture: :base,
        hidden_sizes: [16, 32],
        num_blocks: [1, 1],
        num_attention_heads: [1, 2],
        image_size: 32,
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.Segformer.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

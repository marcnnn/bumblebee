defmodule Bumblebee.Vision.BeitTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-BeitModel"})

    assert %Bumblebee.Vision.Beit{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _seq_length, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BeitForImageClassification"}
             )

    assert %Bumblebee.Vision.Beit{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_classes} = Nx.shape(outputs.logits)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.Beit{
        architecture: :base,
        vocab_size: 100,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 64,
        image_size: 32,
        patch_size: 8,
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.Beit.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

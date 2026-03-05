defmodule Bumblebee.Vision.SwinV2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-Swinv2Model"})

    assert %Bumblebee.Vision.SwinV2{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 32, 32, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 16, 64}
    assert Nx.shape(outputs.pooled_state) == {1, 64}
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-Swinv2ForImageClassification"}
             )

    assert %Bumblebee.Vision.SwinV2{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 32, 32, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _num_labels} = Nx.shape(outputs.logits)
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.SwinV2{
        architecture: :base,
        image_size: 32,
        embedding_size: 16,
        num_blocks: [1, 1],
        num_attention_heads: [2, 4],
        window_size: 4,
        num_labels: 3
      }

      for arch <- Bumblebee.Vision.SwinV2.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

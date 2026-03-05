defmodule Bumblebee.Vision.SamTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-SamModel"})

    assert %Bumblebee.Vision.Sam{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, 64, 64, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _height, _width, _hidden_size} = Nx.shape(outputs.hidden_state)

    # TODO: Fill in reference values from generate_vision_reference_outputs.py
    # assert_all_close(
    #   outputs.hidden_state[[.., 1..3, 1..3, 1..3]],
    #   Nx.tensor([[...]])
    # )
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Vision.Sam{
        architecture: :base,
        image_size: 64,
        patch_size: 16,
        hidden_size: 32,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 64,
        neck_hidden_size: 16
      }

      for arch <- Bumblebee.Vision.Sam.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end
end

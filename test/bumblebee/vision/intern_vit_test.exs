defmodule Bumblebee.Vision.InternVitTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-InternVLChatV1_5"},
               module: Bumblebee.Vision.InternVit,
               architecture: :base
             )

    assert %Bumblebee.Vision.InternVit{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, spec.image_size, spec.image_size, spec.num_channels})
    }

    outputs = Axon.predict(model, params, inputs)

    num_patches = div(spec.image_size, spec.patch_size) ** 2
    # hidden_state includes class token + patch tokens
    assert {1, sequence_length, _hidden_size} = Nx.shape(outputs.hidden_state)
    assert sequence_length == num_patches + 1
  end
end

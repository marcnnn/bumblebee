defmodule Bumblebee.Vision.SiglipVisionTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-SiglipModel"},
               module: Bumblebee.Vision.SiglipVision,
               architecture: :base
             )

    assert %Bumblebee.Vision.SiglipVision{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, spec.image_size, spec.image_size, spec.num_channels})
    }

    outputs = Axon.predict(model, params, inputs)

    num_patches = div(spec.image_size, spec.patch_size) ** 2

    assert Nx.shape(outputs.hidden_state) == {1, num_patches, spec.hidden_size}
    assert Nx.shape(outputs.pooled_state) == {1, spec.hidden_size}
  end
end

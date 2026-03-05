defmodule Bumblebee.Vision.Qwen2VlVisionTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, spec} =
             Bumblebee.load_spec({:hf, "Qwen/Qwen2-VL-2B-Instruct"},
               module: Bumblebee.Vision.Qwen2VlVision,
               architecture: :base
             )

    assert %Bumblebee.Vision.Qwen2VlVision{architecture: :base} = spec
  end
end

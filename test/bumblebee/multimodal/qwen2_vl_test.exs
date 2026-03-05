defmodule Bumblebee.Multimodal.Qwen2VlTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":for_conditional_generation" do
    assert {:ok, spec} =
             Bumblebee.load_spec({:hf, "Qwen/Qwen2-VL-2B-Instruct"},
               module: Bumblebee.Multimodal.Qwen2Vl,
               architecture: :for_conditional_generation
             )

    assert %Bumblebee.Multimodal.Qwen2Vl{architecture: :for_conditional_generation} = spec
    assert %Bumblebee.Vision.Qwen2VlVision{} = spec.vision_spec
    assert %Bumblebee.Text.Qwen2{architecture: :for_causal_language_modeling} = spec.text_spec
  end
end

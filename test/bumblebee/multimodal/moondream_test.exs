defmodule Bumblebee.Multimodal.MoondreamTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-MoondreamForConditionalGeneration"},
               module: Bumblebee.Multimodal.Moondream,
               architecture: :for_conditional_generation
             )

    assert %Bumblebee.Multimodal.Moondream{architecture: :for_conditional_generation} = spec

    %{vision_spec: vision_spec} = spec

    inputs = %{
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
      "pixel_values" =>
        Nx.broadcast(
          0.5,
          {1, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}
        )
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, _seq_len, _vocab_size} = Nx.shape(outputs.logits)
  end
end

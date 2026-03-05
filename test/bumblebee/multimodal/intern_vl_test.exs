defmodule Bumblebee.Multimodal.InternVlTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-InternVLChatV1_5"}
             )

    assert %Bumblebee.Multimodal.InternVl{architecture: :for_conditional_generation} = spec

    %{vision_spec: vision_spec, text_spec: text_spec} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
      "pixel_values" =>
        Nx.broadcast(
          0.5,
          {1, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}
        )
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, 8, _vocab_size} = Nx.shape(outputs.logits)
  end
end

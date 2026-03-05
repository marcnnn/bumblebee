defmodule Bumblebee.Multimodal.Blip2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-Blip2ForConditionalGeneration"}
             )

    assert %Bumblebee.Multimodal.Blip2{architecture: :for_conditional_generation} = spec

    inputs = %{
      "decoder_input_ids" => Nx.tensor([[15, 25, 35, 45, 55, 65, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
      "pixel_values" => Nx.broadcast(0.5, {1, 30, 30, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) |> elem(0) == 1
    assert Nx.shape(outputs.logits) |> elem(1) == 8
  end
end

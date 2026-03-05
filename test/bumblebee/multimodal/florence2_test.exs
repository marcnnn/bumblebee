defmodule Bumblebee.Multimodal.Florence2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":for_conditional_generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-Florence2ForConditionalGeneration"},
               log_params_diff: false
             )

    assert %Bumblebee.Multimodal.Florence2{architecture: :for_conditional_generation} = spec

    inputs = %{
      "pixel_values" =>
        Nx.broadcast(
          0.5,
          {1, spec.vision_spec.image_size, spec.vision_spec.image_size,
           spec.vision_spec.num_channels}
        ),
      "decoder_input_ids" => Nx.tensor([[2, 10, 20, 30, 0, 0]]),
      "decoder_attention_mask" => Nx.tensor([[1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, 6, _vocab_size} = Nx.shape(outputs.logits)
  end

  describe "model/1 builds without errors" do
    test ":for_conditional_generation architecture" do
      vision_spec =
        Bumblebee.configure(%Bumblebee.Vision.Davit{},
          image_size: 32,
          patch_size: 4,
          hidden_sizes: [16, 32],
          depths: [1, 1],
          num_attention_heads: [2, 4]
        )

      text_spec =
        Bumblebee.configure(
          %Bumblebee.Text.Bart{architecture: :for_conditional_generation},
          vocab_size: 100,
          max_positions: 64,
          hidden_size: 32,
          encoder_num_blocks: 1,
          decoder_num_blocks: 1,
          encoder_num_attention_heads: 2,
          decoder_num_attention_heads: 2,
          encoder_intermediate_size: 64,
          decoder_intermediate_size: 64
        )

      spec =
        Bumblebee.configure(
          %Bumblebee.Multimodal.Florence2{architecture: :for_conditional_generation},
          vision_spec: vision_spec,
          text_spec: text_spec,
          projection_size: 32
        )

      model = Bumblebee.build_model(spec)
      assert %Axon{} = model

      inputs = %{
        "pixel_values" => Nx.broadcast(0.5, {1, 32, 32, 3}),
        "decoder_input_ids" => Nx.tensor([[2, 10, 20]]),
        "decoder_attention_mask" => Nx.tensor([[1, 1, 1]])
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(inputs, Axon.None.template())
      outputs = predict_fn.(params, inputs)

      assert {1, 3, 100} = Nx.shape(outputs.logits)
    end
  end
end

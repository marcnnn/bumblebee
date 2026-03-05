defmodule Bumblebee.Multimodal.LlavaTest do
  use ExUnit.Case, async: true

  describe "Bumblebee.Multimodal.Llava" do
    test ":for_conditional_generation model builds successfully" do
      vision_spec = Bumblebee.configure(Bumblebee.Vision.ClipVision, image_size: 32, patch_size: 8, hidden_size: 16, num_blocks: 1, num_attention_heads: 2, intermediate_size: 32)
      text_spec = Bumblebee.configure(Bumblebee.Text.Llama, architecture: :for_causal_language_modeling, vocab_size: 100, hidden_size: 32, num_blocks: 1, num_attention_heads: 2, intermediate_size: 64, num_key_value_heads: 2)

      spec =
        Bumblebee.configure(Bumblebee.Multimodal.Llava,
          vision_spec: vision_spec,
          text_spec: text_spec
        )

      assert %Bumblebee.Multimodal.Llava{architecture: :for_conditional_generation} = spec

      model = Bumblebee.build_model(spec)
      assert %Axon{} = model
    end

    test "spec configuration" do
      spec = %Bumblebee.Multimodal.Llava{}
      assert spec.architecture == :for_conditional_generation
      assert spec.vision_spec == nil
      assert spec.text_spec == nil
      assert spec.projection_size == nil
    end

    test "architectures" do
      assert Bumblebee.Multimodal.Llava.architectures() == [:for_conditional_generation]
    end

    test "input_template returns correct shapes" do
      vision_spec = Bumblebee.configure(Bumblebee.Vision.ClipVision, image_size: 224, num_channels: 3)

      spec =
        Bumblebee.configure(Bumblebee.Multimodal.Llava,
          vision_spec: vision_spec,
          text_spec: Bumblebee.configure(Bumblebee.Text.Llama, architecture: :for_causal_language_modeling)
        )

      template = Bumblebee.Multimodal.Llava.input_template(spec)

      assert %Nx.Tensor{} = template["pixel_values"]
      assert Nx.shape(template["pixel_values"]) == {1, 224, 224, 3}
      assert template["input_ids"] == Nx.template({1, 1}, :u32)
    end
  end
end

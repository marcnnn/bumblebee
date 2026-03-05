defmodule Bumblebee.Vision.DavitTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-DaViTModel"},
               log_params_diff: false
             )

    assert %Bumblebee.Vision.Davit{architecture: :base} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, spec.image_size, spec.image_size, spec.num_channels})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.pooled_state) ==
             {1, List.last(spec.hidden_sizes)}
  end

  test ":for_image_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DaViTForImageClassification"},
               log_params_diff: false
             )

    assert %Bumblebee.Vision.Davit{architecture: :for_image_classification} = spec

    inputs = %{
      "pixel_values" => Nx.broadcast(0.5, {1, spec.image_size, spec.image_size, spec.num_channels})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, spec.num_labels}
  end

  describe "model/1 builds without errors" do
    test ":base architecture" do
      spec =
        Bumblebee.configure(%Bumblebee.Vision.Davit{},
          image_size: 32,
          patch_size: 4,
          hidden_sizes: [16, 32],
          depths: [1, 1],
          num_attention_heads: [2, 4]
        )

      assert %Axon{} = Bumblebee.build_model(spec)
    end

    test ":for_image_classification architecture" do
      spec =
        Bumblebee.configure(%Bumblebee.Vision.Davit{architecture: :for_image_classification},
          image_size: 32,
          patch_size: 4,
          hidden_sizes: [16, 32],
          depths: [1, 1],
          num_attention_heads: [2, 4],
          num_labels: 5
        )

      model = Bumblebee.build_model(spec)
      assert %Axon{} = model

      inputs = %{
        "pixel_values" => Nx.broadcast(0.5, {1, 32, 32, 3})
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(inputs, Axon.None.template())
      outputs = predict_fn.(params, inputs)

      assert Nx.shape(outputs.logits) == {1, 5}
    end
  end
end

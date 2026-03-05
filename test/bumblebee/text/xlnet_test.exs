defmodule Bumblebee.Text.XlnetTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-XLNetModel"})

    assert %Bumblebee.Text.Xlnet{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, 10, _hidden_size} = Nx.shape(outputs.hidden_state)
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-XLNetForSequenceClassification"}
             )

    assert %Bumblebee.Text.Xlnet{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}
  end

  test ":for_token_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-XLNetForTokenClassification"}
             )

    assert %Bumblebee.Text.Xlnet{architecture: :for_token_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 2}
  end

  test ":for_question_answering" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-XLNetForQuestionAnswering"}
             )

    assert %Bumblebee.Text.Xlnet{architecture: :for_question_answering} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.start_logits) == {1, 10}
    assert Nx.shape(outputs.end_logits) == {1, 10}
  end

  test ":for_multiple_choice" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-XLNetForMultipleChoice"}
             )

    assert %Bumblebee.Text.Xlnet{architecture: :for_multiple_choice} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]]),
      "attention_mask" => Nx.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]]),
      "token_type_ids" => Nx.tensor([[[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 1}
  end

  describe "model structure" do
    test "builds the Axon graph for each architecture" do
      spec = %Bumblebee.Text.Xlnet{
        architecture: :base,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 64,
        num_labels: 3
      }

      for arch <- Bumblebee.Text.Xlnet.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Text.Xlnet{}

      data = %{
        "vocab_size" => 32000,
        "max_position_embeddings" => 512,
        "d_model" => 1024,
        "n_layer" => 24,
        "n_head" => 16,
        "d_inner" => 4096,
        "hidden_act" => "gelu",
        "hidden_dropout_prob" => 0.1,
        "attention_probs_dropout_prob" => 0.1,
        "layer_norm_eps" => 1.0e-12,
        "initializer_range" => 0.02
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 32000
      assert loaded_spec.hidden_size == 1024
      assert loaded_spec.num_blocks == 24
      assert loaded_spec.num_attention_heads == 16
      assert loaded_spec.intermediate_size == 4096
    end
  end
end

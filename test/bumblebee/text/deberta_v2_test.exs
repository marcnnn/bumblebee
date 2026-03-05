defmodule Bumblebee.Text.DebertaV2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-DebertaV2Model"})

    assert %Bumblebee.Text.DebertaV2{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}
  end

  test ":for_masked_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DebertaV2ForMaskedLM"}
             )

    assert %Bumblebee.Text.DebertaV2{architecture: :for_masked_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert {1, 10, _vocab_size} = Nx.shape(outputs.logits)
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification"}
             )

    assert %Bumblebee.Text.DebertaV2{architecture: :for_sequence_classification} = spec

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
               {:hf, "hf-internal-testing/tiny-random-DebertaV2ForTokenClassification"}
             )

    assert %Bumblebee.Text.DebertaV2{architecture: :for_token_classification} = spec

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
               {:hf, "hf-internal-testing/tiny-random-DebertaV2ForQuestionAnswering"}
             )

    assert %Bumblebee.Text.DebertaV2{architecture: :for_question_answering} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "token_type_ids" => Nx.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.start_logits) == {1, 10}
    assert Nx.shape(outputs.end_logits) == {1, 10}
  end

  test ":for_multiple_choice" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DebertaV2ForMultipleChoice"}
             )

    assert %Bumblebee.Text.DebertaV2{architecture: :for_multiple_choice} = spec

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
      spec = %Bumblebee.Text.DebertaV2{
        architecture: :base,
        vocab_size: 100,
        max_positions: 32,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 64,
        num_labels: 3
      }

      for arch <- Bumblebee.Text.DebertaV2.architectures() do
        spec = %{spec | architecture: arch}
        model = Bumblebee.build_model(spec)
        assert %Axon{} = model
      end
    end
  end

  describe "config loading" do
    test "loads config from HuggingFace format" do
      spec = %Bumblebee.Text.DebertaV2{}

      data = %{
        "vocab_size" => 128100,
        "max_position_embeddings" => 512,
        "type_vocab_size" => 0,
        "hidden_size" => 1536,
        "num_hidden_layers" => 24,
        "num_attention_heads" => 24,
        "intermediate_size" => 6144,
        "hidden_act" => "gelu",
        "hidden_dropout_prob" => 0.1,
        "attention_probs_dropout_prob" => 0.1,
        "layer_norm_eps" => 1.0e-7,
        "initializer_range" => 0.02,
        "relative_attention" => true
      }

      loaded_spec = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      assert loaded_spec.vocab_size == 128100
      assert loaded_spec.hidden_size == 1536
      assert loaded_spec.num_blocks == 24
      assert loaded_spec.relative_attention == true
    end
  end
end

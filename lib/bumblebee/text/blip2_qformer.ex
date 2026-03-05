defmodule Bumblebee.Text.Blip2Qformer do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 30522,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 512,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      encoder_hidden_size: [
        default: 1408,
        doc: "the dimensionality of the encoder (vision model) hidden state"
      ],
      num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the Q-Former"
      ],
      num_attention_heads: [
        default: 12,
        doc: "the number of attention heads for each attention layer in the Q-Former"
      ],
      intermediate_size: [
        default: 3072,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN)"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for embedding and encoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      layer_norm_epsilon: [
        default: 1.0e-12,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      num_query_tokens: [
        default: 32,
        doc: "the number of learnable query tokens for the Q-Former"
      ],
      cross_attention_frequency: [
        default: 2,
        doc: "the frequency of cross-attention layers (every Nth block has cross-attention)"
      ]
    ]

  @moduledoc """
  The BLIP-2 Q-Former model.

  The Q-Former is a BERT-like encoder with cross-attention to vision encoder
  features. It uses learnable query embeddings that cross-attend to the
  vision encoder output, bridging the modality gap between vision and language.

  ## Architectures

    * `:base` - the base Q-Former model

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary. Optional when
      using only query embeddings.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"encoder_hidden_state"` - `{batch_size, encoder_sequence_length, encoder_hidden_size}`

      Last hidden state output from the vision encoder. This hidden state is
      used in cross-attention blocks in the Q-Former.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(_spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :u32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}
    encoder_hidden_shape = {nil, nil, spec.encoder_hidden_size}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("encoder_hidden_state", optional: true, shape: encoder_hidden_shape)
    ])
  end

  defp core(inputs, spec) do
    query_embeddings =
      Layers.learned_embeddings(spec.num_query_tokens, spec.hidden_size,
        name: "query_embeddings",
        initializer: Axon.Initializers.normal(scale: spec.initializer_scale)
      )

    embeddings =
      embedder(inputs["input_ids"], inputs["position_ids"], inputs["input_embeddings"], spec,
        name: "embedder"
      )

    # Prepend learnable query tokens to the text embeddings
    hidden_state =
      Layers.concatenate_embeddings([query_embeddings, embeddings])

    encoder_outputs =
      encoder(
        hidden_state,
        inputs["attention_mask"],
        inputs["encoder_hidden_state"],
        spec,
        name: "encoder"
      )

    # Extract only the query token outputs (first num_query_tokens positions)
    query_output =
      Axon.nx(encoder_outputs.hidden_state, fn hidden_state ->
        Nx.slice_along_axis(hidden_state, 0, spec.num_query_tokens, axis: 1)
      end)

    %{
      hidden_state: query_output,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions,
      cross_attentions: encoder_outputs.cross_attentions
    }
  end

  defp embedder(input_ids, position_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    input_embeddings =
      Layers.default input_embeddings do
        Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "token_embedding")
        )
      end

    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(input_embeddings)
      end

    position_embeddings =
      Axon.embedding(position_ids, spec.max_positions, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "position_embedding")
      )

    input_embeddings
    |> Axon.add(position_embeddings)
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dropout(rate: spec.dropout_rate)
  end

  defp encoder(hidden_state, attention_mask, encoder_hidden_state, spec, opts) do
    name = opts[:name]

    Layers.Transformer.blocks(
      hidden_state,
      [
        attention_mask: attention_mask,
        num_blocks: spec.num_blocks,
        num_attention_heads: spec.num_attention_heads,
        hidden_size: spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: spec.dropout_rate,
        attention_dropout_rate: spec.attention_dropout_rate,
        layer_norm: [
          epsilon: spec.layer_norm_epsilon
        ],
        ffn: [
          intermediate_size: spec.intermediate_size,
          activation: spec.activation
        ],
        name: join(name, "blocks"),
        cross_hidden_state: encoder_hidden_state,
        cross_attention_mask: Layers.none(),
        cross_attention_head_mask: Layers.none()
      ]
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, %{"model_type" => "blip-2", "qformer_config" => data}) do
      load(spec, data)
    end

    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          encoder_hidden_size: {"encoder_hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()},
          cross_attention_frequency: {"cross_attention_frequency", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      prefix = "qformer.bert."

      %{
        "query_embeddings" => %{
          "embeddings" =>
            {[{"qformer.query_tokens", "query_tokens"}],
             fn [value] -> Nx.squeeze(value, axes: [0]) end}
        },
        "embedder.token_embedding" => prefix <> "embeddings.word_embeddings",
        "embedder.position_embedding" => prefix <> "embeddings.position_embeddings",
        "embedder.token_type_embedding" => prefix <> "embeddings.token_type_embeddings",
        "embedder.norm" => prefix <> "embeddings.LayerNorm",
        "encoder.blocks.{n}.self_attention.query" =>
          prefix <> "encoder.layer.{n}.attention.self.query",
        "encoder.blocks.{n}.self_attention.key" =>
          prefix <> "encoder.layer.{n}.attention.self.key",
        "encoder.blocks.{n}.self_attention.value" =>
          prefix <> "encoder.layer.{n}.attention.self.value",
        "encoder.blocks.{n}.self_attention.output" =>
          prefix <> "encoder.layer.{n}.attention.output.dense",
        "encoder.blocks.{n}.self_attention_norm" =>
          prefix <> "encoder.layer.{n}.attention.output.LayerNorm",
        "encoder.blocks.{n}.cross_attention.query" =>
          prefix <> "encoder.layer.{n}.crossattention.self.query",
        "encoder.blocks.{n}.cross_attention.key" =>
          prefix <> "encoder.layer.{n}.crossattention.self.key",
        "encoder.blocks.{n}.cross_attention.value" =>
          prefix <> "encoder.layer.{n}.crossattention.self.value",
        "encoder.blocks.{n}.cross_attention.output" =>
          prefix <> "encoder.layer.{n}.crossattention.output.dense",
        "encoder.blocks.{n}.cross_attention_norm" =>
          prefix <> "encoder.layer.{n}.crossattention.output.LayerNorm",
        "encoder.blocks.{n}.ffn.intermediate" =>
          prefix <> "encoder.layer.{n}.intermediate.dense",
        "encoder.blocks.{n}.ffn.output" => prefix <> "encoder.layer.{n}.output.dense",
        "encoder.blocks.{n}.output_norm" => prefix <> "encoder.layer.{n}.output.LayerNorm"
      }
    end
  end
end

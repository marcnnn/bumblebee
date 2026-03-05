defmodule Bumblebee.Text.Bloom do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 250_880,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 2048,
        doc: """
        the maximum sequence length that this model can process. Typically this is set to a
        large value just in case, such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 2560,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 30,
        doc: "the number of Transformer blocks in the decoder"
      ],
      num_attention_heads: [
        default: 32,
        doc: "the number of attention heads for each attention layer in the decoder"
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
      classifier_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for the classification head"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 3)

  @moduledoc """
  BLOOM model family.

  ## Architectures

    * `:base` - plain BLOOM without any head on top

    * `:for_causal_language_modeling` - BLOOM with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - BLOOM with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - BLOOM with a token classification
      head. The head returns logits for each token in the original
      sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"attention_head_mask"` - `{num_blocks, num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the decoder.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression). With cache, certain hidden states are
      taken from the cache, rather than recomputed on every decoding
      pass. The cache should be treated as opaque and initialized with
      `Bumblebee.Text.Generation.init_cache/4`.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(),
    do: [
      :base,
      :for_causal_language_modeling,
      :for_sequence_classification,
      :for_token_classification
    ]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(_spec) do
    %{"input_ids" => Nx.template({1, 1}, :u32)}
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      Axon.dense(outputs.hidden_state, spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.output",
        use_bias: false
      )

    pooled_logits =
      Layers.if_present inputs["input_ids"] do
        Axon.layer(
          fn logits, input_ids, _opts ->
            indices =
              input_ids
              |> Nx.not_equal(spec.pad_token_id)
              |> Nx.sum(axes: [-1])
              |> Nx.subtract(1)
              |> Nx.as_type({:s, 64})

            Bumblebee.Utils.Nx.batched_take(logits, indices)
          end,
          [logits, inputs["input_ids"]]
        )
      else
        Layers.take_token(logits, axis: 1, index: -1)
      end

    Layers.output(%{
      logits: pooled_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :for_token_classification} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Axon.dropout(
        rate: classifier_dropout_rate(spec),
        name: "token_classification_head.dropout"
      )
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "token_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      decoder_num_blocks: spec.num_blocks
    )
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}
    decoder_attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: decoder_attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(inputs["input_ids"], inputs["input_embeddings"], spec,
        name: "embedder"
      )

    decoder_outputs =
      decoder(
        embeddings,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    hidden_state =
      Axon.layer_norm(decoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: "norm"
      )

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.replace(decoder_outputs.hidden_states, -1, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  defp embedder(input_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    input_embeddings =
      Layers.default input_embeddings do
        Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "token_embedding")
        )
      end

    Axon.layer_norm(input_embeddings,
      epsilon: spec.layer_norm_epsilon,
      name: join(name, "token_embedding_norm")
    )
  end

  defp decoder(
         hidden_state,
         attention_mask,
         attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]

    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      cache: cache,
      causal: true,
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
        intermediate_size: 4 * spec.hidden_size,
        activation: spec.activation
      ],
      block_type: :norm_first,
      query_use_bias: true,
      key_use_bias: true,
      value_use_bias: true,
      output_use_bias: true,
      name: join(name, "blocks")
    )
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    Layers.dense_transposed(hidden_state, spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
  end

  defp classifier_dropout_rate(spec) do
    spec.classifier_dropout_rate || spec.dropout_rate
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          hidden_size:
            {"n_embed",
             fn n_embed, data ->
               cond do
                 n_embed -> {:ok, n_embed}
                 data["hidden_size"] -> {:ok, data["hidden_size"]}
                 true -> :error
               end
             end},
          num_blocks: {"n_layer", number()},
          num_attention_heads: {"n_head", number()},
          activation: {"hidden_act", optional(activation())},
          dropout_rate: {"hidden_dropout", number()},
          attention_dropout_rate: {"attention_dropout", number()},
          layer_norm_epsilon: {"layer_norm_epsilon", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => "transformer.word_embeddings",
        "embedder.token_embedding_norm" => "transformer.word_embeddings_layernorm",
        "decoder.blocks.{n}.self_attention.query" =>
          Shared.sliced_dense_params_source(
            "transformer.h.{n}.self_attention.query_key_value",
            {spec.num_attention_heads, [1, 1, 1], :auto},
            0
          ),
        "decoder.blocks.{n}.self_attention.key" =>
          Shared.sliced_dense_params_source(
            "transformer.h.{n}.self_attention.query_key_value",
            {spec.num_attention_heads, [1, 1, 1], :auto},
            1
          ),
        "decoder.blocks.{n}.self_attention.value" =>
          Shared.sliced_dense_params_source(
            "transformer.h.{n}.self_attention.query_key_value",
            {spec.num_attention_heads, [1, 1, 1], :auto},
            2
          ),
        "decoder.blocks.{n}.self_attention.output" =>
          "transformer.h.{n}.self_attention.dense",
        "decoder.blocks.{n}.self_attention_norm" =>
          "transformer.h.{n}.input_layernorm",
        "decoder.blocks.{n}.ffn.intermediate" =>
          "transformer.h.{n}.mlp.dense_h_to_4h",
        "decoder.blocks.{n}.ffn.output" =>
          "transformer.h.{n}.mlp.dense_4h_to_h",
        "decoder.blocks.{n}.output_norm" =>
          "transformer.h.{n}.post_attention_layernorm",
        "norm" => "transformer.ln_f",
        "language_modeling_head.output" => "lm_head",
        "sequence_classification_head.output" => "score",
        "token_classification_head.output" => "classifier"
      }
    end
  end
end

defmodule Bumblebee.Text.GptNeo do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 50257,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 2048,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 2048,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 24,
        doc: "the number of Transformer blocks in the decoder"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the decoder"
      ],
      intermediate_size: [
        default: 8192,
        doc: """
        the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the decoder
        """
      ],
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      attention_window_size: [
        default: 256,
        doc: "the window size for local attention layers"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for embedding and encoder"
      ],
      embeddings_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for embeddings"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      classifier_dropout_rate: [
        default: 0.1,
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
      Shared.token_options(pad_token_id: 50256)

  @moduledoc """
  GPT-Neo model family.

  ## Architectures

    * `:base` - plain GPT-Neo without any head on top

    * `:for_causal_language_modeling` - GPT-Neo with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - GPT-Neo with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_token_classification` - GPT-Neo with a token classification
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
      Layers.dense_transposed(outputs.hidden_state, spec.vocab_size,
        kernel_initializer: kernel_initializer(spec),
        name: "language_modeling_head.output"
      )

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
        name: "sequence_classification_head.output"
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
      attentions: outputs.attentions
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
      embedder(inputs["input_ids"], inputs["position_ids"], inputs["input_embeddings"], spec,
        name: "embedder"
      )

    outputs =
      decoder(
        embeddings,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    hidden_state =
      Axon.layer_norm(outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: "norm"
      )

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.replace(outputs.hidden_states, -1, hidden_state),
      attentions: outputs.attentions,
      cache: outputs.cache
    }
  end

  defp embedder(input_ids, position_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    input_embeddings =
      Layers.default input_embeddings do
        Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
          name: join(name, "token_embedding")
        )
      end

    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(input_embeddings)
      end

    position_embeddings =
      Axon.embedding(position_ids, spec.max_positions, spec.hidden_size,
        name: join(name, "position_embedding")
      )

    input_embeddings
    |> Axon.add(position_embeddings)
    |> Axon.dropout(rate: spec.embeddings_dropout_rate)
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
        intermediate_size: spec.intermediate_size,
        activation: spec.activation
      ],
      block_type: :norm_first,
      query_use_bias: false,
      key_use_bias: false,
      value_use_bias: false,
      output_use_bias: true,
      name: join(name, "blocks")
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
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_layers", number()},
          num_attention_heads: {"num_heads", number()},
          intermediate_size: {"intermediate_size", optional(number())},
          activation: {"activation_function", activation()},
          attention_window_size: {"window_size", number()},
          dropout_rate: {"resid_dropout", number()},
          embeddings_dropout_rate: {"embed_dropout", number()},
          attention_dropout_rate: {"attention_dropout", number()},
          classifier_dropout_rate: {"classifier_dropout", optional(number())},
          layer_norm_epsilon: {"layer_norm_epsilon", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.token_embedding" => "transformer.wte",
        "embedder.position_embedding" => "transformer.wpe",
        "decoder.blocks.{n}.self_attention.query" => "transformer.h.{n}.attn.attention.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "transformer.h.{n}.attn.attention.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "transformer.h.{n}.attn.attention.v_proj",
        "decoder.blocks.{n}.self_attention.output" =>
          "transformer.h.{n}.attn.attention.out_proj",
        "decoder.blocks.{n}.self_attention_norm" => "transformer.h.{n}.ln_1",
        "decoder.blocks.{n}.ffn.intermediate" => "transformer.h.{n}.mlp.c_fc",
        "decoder.blocks.{n}.ffn.output" => "transformer.h.{n}.mlp.c_proj",
        "decoder.blocks.{n}.output_norm" => "transformer.h.{n}.ln_2",
        "norm" => "transformer.ln_f",
        "language_modeling_head.output" => "transformer.wte",
        "sequence_classification_head.output" => "score",
        "token_classification_head.output" => "classifier"
      }
    end
  end
end

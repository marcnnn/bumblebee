defmodule Bumblebee.Text.Opt do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 50272,
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
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 12,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      intermediate_size: [
        default: 3072,
        doc: "the dimensionality of intermediate layers"
      ],
      activation: [
        default: :relu,
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
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      do_layer_norm_before: [
        default: true,
        doc: "whether to apply layer norm before (pre-norm) or after (post-norm) attention"
      ],
      word_embedding_projection: [
        default: nil,
        doc: """
        word embedding projection size. If specified and different from hidden_size,
        a projection layer is added after the token embedding
        """
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 1, bos_token_id: 2, eos_token_id: 2)

  @moduledoc """
  OPT model family.

  ## Architectures

    * `:base` - plain OPT without any head on top

    * `:for_causal_language_modeling` - OPT with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - OPT with a sequence
      classification head. The head returns logits corresponding to
      possible classes

    * `:for_question_answering` - OPT with a span classification head.
      The head returns logits for the span start and end positions

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
      the encoder.

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

  ## References

    * [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)

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
      :for_question_answering
    ]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(_spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
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
    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

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

  def model(%__MODULE__{architecture: :for_question_answering} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      Axon.dense(outputs.hidden_state, 2,
        kernel_initializer: kernel_initializer(spec),
        name: "question_answering_head.output"
      )

    {start_logits, end_logits} = Layers.split_pair(logits)

    Layers.output(%{
      start_logits: start_logits,
      end_logits: end_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    })
  end

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}

    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(
        inputs["input_ids"],
        inputs["position_ids"],
        inputs["input_embeddings"],
        spec,
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
      if spec.do_layer_norm_before do
        Axon.layer_norm(decoder_outputs.hidden_state,
          epsilon: spec.layer_norm_epsilon,
          name: "output_norm"
        )
      else
        decoder_outputs.hidden_state
      end

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  defp embedder(input_ids, position_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    embed_size = spec.word_embedding_projection || spec.hidden_size

    token_embeddings =
      Layers.default input_embeddings do
        Axon.embedding(input_ids, spec.vocab_size, embed_size,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "token_embedding")
        )
      end

    # OPT uses offset=2 for position embeddings
    position_ids =
      Layers.default position_ids do
        Layers.default_position_ids(token_embeddings)
      end

    offset_position_ids =
      Axon.nx(position_ids, fn pos -> Nx.add(pos, 2) end)

    position_embeddings =
      Axon.embedding(offset_position_ids, spec.max_positions + 2, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "position_embedding")
      )

    token_embeddings =
      if embed_size != spec.hidden_size do
        Axon.dense(token_embeddings, spec.hidden_size,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "projection")
        )
      else
        token_embeddings
      end

    Axon.add(token_embeddings, position_embeddings)
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
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

    block_type = if spec.do_layer_norm_before, do: :norm_first, else: :norm_last

    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      cache: cache,
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
      block_type: block_type,
      causal: true,
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
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"ffn_dim", number()},
          activation: {"activation_function", activation()},
          dropout_rate: {"dropout", number()},
          attention_dropout_rate: {"attention_dropout", number()},
          layer_norm_epsilon: {"layer_norm_elementwise_affine", optional(number())},
          initializer_scale: {"init_std", optional(number())},
          do_layer_norm_before: {"do_layer_norm_before", boolean()},
          word_embedding_projection: {"word_embed_proj_dim", optional(number())}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.token_embedding" => "model.decoder.embed_tokens",
        "embedder.position_embedding" => "model.decoder.embed_positions",
        "embedder.projection" => "model.decoder.project_in",
        "decoder.blocks.{n}.self_attention.query" =>
          "model.decoder.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" =>
          "model.decoder.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" =>
          "model.decoder.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" =>
          "model.decoder.layers.{n}.self_attn.out_proj",
        "decoder.blocks.{n}.self_attention_norm" =>
          "model.decoder.layers.{n}.self_attn_layer_norm",
        "decoder.blocks.{n}.ffn.intermediate" => "model.decoder.layers.{n}.fc1",
        "decoder.blocks.{n}.ffn.output" => "model.decoder.layers.{n}.fc2",
        "decoder.blocks.{n}.output_norm" => "model.decoder.layers.{n}.final_layer_norm",
        "output_norm" => "model.decoder.final_layer_norm",
        "language_modeling_head.output" => "model.decoder.embed_tokens",
        "sequence_classification_head.output" => "score",
        "question_answering_head.output" => "qa_outputs"
      }
    end
  end
end

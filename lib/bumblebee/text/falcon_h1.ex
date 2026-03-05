defmodule Bumblebee.Text.FalconH1 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 65024,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 8192,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 4096,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 11008,
        doc: "the dimensionality of intermediate layers in the feed-forward network"
      ],
      num_blocks: [
        default: 32,
        doc: "the number of hybrid Transformer+Mamba blocks in the model"
      ],
      num_attention_heads: [
        default: 32,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: nil,
        doc: """
        the number of key-value heads for grouped query attention.
        Defaults to num_attention_heads if not set
        """
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      mamba_d_state: [
        default: 128,
        doc: "the dimensionality of the SSM state space"
      ],
      mamba_d_conv: [
        default: 4,
        doc: "the kernel size of the 1D convolution in the Mamba block"
      ],
      mamba_expand: [
        default: 2,
        doc: "the expansion factor for the Mamba inner dimension"
      ],
      mamba_n_heads: [
        default: 128,
        doc: "the number of heads in the Mamba SSM block"
      ],
      mamba_d_head: [
        default: 64,
        doc: "the dimensionality per head in the Mamba SSM block"
      ],
      head_size: [
        default: nil,
        doc: """
        the size of the key, value, and query projection per attention head.
        Defaults to `div(hidden_size, num_attention_heads)`
        """
      ],
      rotary_embedding_base: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by RMS normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      tie_word_embeddings: [
        default: false,
        doc: "whether to tie input and output embedding weights"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 11)

  @moduledoc """
  Falcon-H1 hybrid Transformer+Mamba model family.

  ## Architectures

    * `:base` - plain Falcon-H1 without any head on top

    * `:for_causal_language_modeling` - Falcon-H1 with a language modeling
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

  ## Notes

  Falcon-H1 is a parallel hybrid architecture where each block contains both
  a Transformer attention component and a Mamba-2 SSM component running in
  parallel, with their outputs added together. The SSM component is approximated
  using dense layers (same approach as the Mamba model in Bumblebee).

  ## References

    * [Falcon-H1 models](https://huggingface.co/tiiuae)

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
      :for_causal_language_modeling
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
    num_key_value_heads = spec.num_key_value_heads || spec.num_attention_heads

    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.head_size,
      decoder_num_attention_heads: num_key_value_heads,
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
        inputs["input_embeddings"],
        spec,
        name: "embedder"
      )

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    decoder_outputs =
      hybrid_blocks(
        embeddings,
        position_ids,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    hidden_state =
      Layers.rms_norm(decoder_outputs.hidden_state,
        name: "output_norm",
        epsilon: spec.layer_norm_epsilon
      )

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  defp embedder(input_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    Layers.default input_embeddings do
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
    end
  end

  defp hybrid_blocks(
         hidden_state,
         position_ids,
         attention_mask,
         attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]
    num_key_value_heads = spec.num_key_value_heads || spec.num_attention_heads
    mamba_inner_size = spec.hidden_size * spec.mamba_expand

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({}),
      cache: cache
    }

    outputs =
      for idx <- 0..(spec.num_blocks - 1), reduce: state do
        state ->
          block_name = join(name, "blocks.#{idx}")
          block_attention_head_mask = Axon.nx(attention_head_mask, & &1[idx])
          block_cache = Layers.Decoder.get_block_cache(state.cache, idx)
          {self_attention_cache, _cross_attention_cache} =
            Layers.Decoder.get_attention_caches(block_cache)

          hidden_state = state.hidden_state

          # Pre-norm (RMS norm) before the parallel attention + SSM
          normed =
            Layers.rms_norm(hidden_state,
              name: join(block_name, "input_norm"),
              epsilon: spec.layer_norm_epsilon
            )

          # === Attention branch ===
          {attention_output, attention_weights, self_attention_cache, _attention_relative_bias} =
            Layers.Transformer.multi_head_attention(normed, normed, normed,
              attention_mask: attention_mask,
              attention_head_mask: block_attention_head_mask,
              attention_cache: self_attention_cache,
              offset: offset,
              causal: true,
              num_heads: spec.num_attention_heads,
              num_key_value_heads: num_key_value_heads,
              hidden_size: spec.hidden_size,
              kernel_initializer: kernel_initializer(spec),
              attention_head_size: spec.head_size,
              query_use_bias: false,
              key_use_bias: false,
              value_use_bias: false,
              output_use_bias: false,
              rotary_embedding: [
                position_ids: position_ids,
                max_positions: spec.max_positions,
                base: spec.rotary_embedding_base
              ],
              name: join(block_name, "self_attention")
            )

          # === SSM (Mamba) branch - dense approximation ===
          # In projection: projects to 2 * mamba_inner_size (split into x and gate)
          in_projected =
            Axon.dense(normed, 2 * mamba_inner_size,
              name: join(block_name, "mamba.in_proj"),
              use_bias: false
            )

          # Split into two paths
          x_branch =
            Axon.nx(in_projected, fn t ->
              Nx.slice_along_axis(t, 0, mamba_inner_size, axis: -1)
            end)

          gate_branch =
            Axon.nx(in_projected, fn t ->
              Nx.slice_along_axis(t, mamba_inner_size, mamba_inner_size, axis: -1)
            end)

          # Apply convolution approximation via dense layer
          x_branch =
            Axon.dense(x_branch, mamba_inner_size,
              name: join(block_name, "mamba.conv_proj"),
              use_bias: true
            )

          x_branch = Layers.activation(x_branch, :silu)

          # Apply gate with silu activation
          gate_branch = Layers.activation(gate_branch, :silu)

          # Combine branches
          mixed = Axon.multiply(x_branch, gate_branch)

          # Out projection
          ssm_output =
            Axon.dense(mixed, spec.hidden_size,
              name: join(block_name, "mamba.out_proj"),
              use_bias: false
            )

          # === Combine attention + SSM outputs with residual ===
          combined = Axon.add(attention_output, ssm_output)
          hidden_state_after_parallel = Axon.add(hidden_state, combined)

          # === Post-attention FFN with its own norm ===
          ffn_normed =
            Layers.rms_norm(hidden_state_after_parallel,
              name: join(block_name, "pre_ff_norm"),
              epsilon: spec.layer_norm_epsilon
            )

          ffn_output =
            gated_ffn(ffn_normed, spec.intermediate_size, spec.hidden_size,
              name: join(block_name, "ffn"),
              activation: spec.activation
            )

          new_hidden_state = Axon.add(hidden_state_after_parallel, ffn_output)

          # Update caches
          cross_attention_cache = Layers.none()

          block_cache =
            Layers.Decoder.put_attention_caches(
              block_cache,
              self_attention_cache,
              cross_attention_cache
            )

          cache = Layers.Decoder.put_block_cache(state.cache, idx, block_cache)

          %{
            hidden_state: new_hidden_state,
            hidden_states: Layers.append(state.hidden_states, new_hidden_state),
            attentions: Layers.append(state.attentions, attention_weights),
            cache: cache
          }
      end

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, hidden_state))
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]
    activation = opts[:activation]

    gate =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, "gate"),
        use_bias: false
      )

    up =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, "up"),
        use_bias: false
      )

    hidden_state = Axon.multiply(Axon.activation(gate, activation), up)

    Axon.dense(hidden_state, output_size,
      name: join(name, "down"),
      use_bias: false
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
          max_positions: {"max_position_embeddings", optional(number())},
          hidden_size: {"hidden_size", number()},
          intermediate_size: {"intermediate_size", optional(number())},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", optional(number())},
          activation: {"hidden_act", optional(activation())},
          head_size: {"head_dim", optional(number())},
          mamba_d_state: {"mamba_d_state", optional(number())},
          mamba_d_conv: {"mamba_d_conv", optional(number())},
          mamba_expand: {"mamba_expand", optional(number())},
          mamba_n_heads: {"mamba_n_heads", optional(number())},
          mamba_d_head: {"mamba_d_head", optional(number())},
          rotary_embedding_base: {"rope_theta", optional(number())},
          layer_norm_epsilon: {"rms_norm_eps", optional(number())},
          initializer_scale: {"initializer_range", optional(number())},
          tie_word_embeddings: {"tie_word_embeddings", optional(boolean())}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => "model.embed_tokens",
        "decoder.blocks.{n}.input_norm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.self_attention.query" => "model.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.mamba.in_proj" => "model.layers.{n}.mamba.in_proj",
        "decoder.blocks.{n}.mamba.conv_proj" => "model.layers.{n}.mamba.conv1d",
        "decoder.blocks.{n}.mamba.out_proj" => "model.layers.{n}.mamba.out_proj",
        "decoder.blocks.{n}.pre_ff_norm" => "model.layers.{n}.pre_ff_layernorm",
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.feed_forward.gate_proj",
        "decoder.blocks.{n}.ffn.up" => "model.layers.{n}.feed_forward.up_proj",
        "decoder.blocks.{n}.ffn.down" => "model.layers.{n}.feed_forward.down_proj",
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head")
      }
    end
  end
end

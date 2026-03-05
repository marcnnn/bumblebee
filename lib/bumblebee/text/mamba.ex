defmodule Bumblebee.Text.Mamba do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 50280,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 24,
        doc: "the number of Mamba blocks in the model"
      ],
      intermediate_size: [
        default: 1536,
        doc: "the dimensionality of the inner projection in the mixer"
      ],
      state_size: [
        default: 16,
        doc: "the dimensionality of the state space latent"
      ],
      conv_kernel_size: [
        default: 4,
        doc: "the size of the 1D convolution kernel in the mixer"
      ],
      expand_factor: [
        default: 2,
        doc: "the expansion factor applied to determine the inner dimension from hidden_size"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by RMS normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 0)

  @moduledoc """
  Mamba model family.

  ## Architectures

    * `:base` - plain Mamba without any head on top

    * `:for_causal_language_modeling` - Mamba with a language modeling
      head. The head returns logits for each token in the original
      sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## Notes

  The selective state space mechanism (SSM) is approximated using dense layers,
  since Axon is primarily designed for transformer-based architectures. This
  implementation allows loading pretrained Mamba weights but the forward pass
  uses a simplified dense approximation rather than the true selective scan.
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
  def init_cache(_spec, _batch_size, _max_length, _inputs) do
    # Mamba uses recurrent state rather than KV cache, but we
    # provide an empty cache for compatibility with the generation interface
    %{}
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

    logits =
      language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states
    })
  end

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(inputs["input_ids"], inputs["input_embeddings"], spec,
        name: "embedder"
      )

    block_outputs =
      mamba_blocks(embeddings, spec, name: "decoder")

    hidden_state =
      Layers.rms_norm(block_outputs.hidden_state,
        name: "norm",
        epsilon: spec.layer_norm_epsilon
      )

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.replace(block_outputs.hidden_states, -1, hidden_state)
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

  defp mamba_blocks(hidden_state, spec, opts) do
    name = opts[:name]
    inner_size = spec.intermediate_size || spec.hidden_size * spec.expand_factor

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state})
    }

    for i <- 0..(spec.num_blocks - 1), reduce: state do
      state ->
        block_name = join(name, "blocks.#{i}")

        hidden_state = state.hidden_state

        # RMS norm before mixer
        normed =
          Layers.rms_norm(hidden_state,
            name: join(block_name, "norm"),
            epsilon: spec.layer_norm_epsilon
          )

        # Simplified Mamba mixer block using dense layers
        # In projection: projects to 2 * inner_size (split into x and residual gate)
        in_projected =
          Axon.dense(normed, 2 * inner_size,
            name: join(block_name, "mixer.in_proj"),
            use_bias: false
          )

        # Split into two paths
        x_branch =
          Axon.nx(in_projected, fn t ->
            Nx.slice_along_axis(t, 0, inner_size, axis: -1)
          end)

        gate_branch =
          Axon.nx(in_projected, fn t ->
            Nx.slice_along_axis(t, inner_size, inner_size, axis: -1)
          end)

        # Apply convolution approximation via dense layer
        x_branch =
          Axon.dense(x_branch, inner_size,
            name: join(block_name, "mixer.conv_proj"),
            use_bias: true
          )

        x_branch = Layers.activation(x_branch, :silu)

        # Apply gate with silu activation
        gate_branch = Layers.activation(gate_branch, :silu)

        # Combine branches
        mixed = Axon.multiply(x_branch, gate_branch)

        # Out projection
        mixer_output =
          Axon.dense(mixed, spec.hidden_size,
            name: join(block_name, "mixer.out_proj"),
            use_bias: false
          )

        # Residual connection
        new_hidden_state = Axon.add(hidden_state, mixer_output)

        hidden_states =
          Layers.append(state.hidden_states, new_hidden_state)

        %{
          hidden_state: new_hidden_state,
          hidden_states: hidden_states
        }
    end
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    # Tie lm-head to word embedding
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
          hidden_size: {"d_model", number()},
          num_blocks: {"n_layer", number()},
          intermediate_size: {"d_inner", optional(number())},
          state_size: {"d_state", number()},
          conv_kernel_size: {"d_conv", number()},
          expand_factor: {"expand", number()},
          layer_norm_epsilon: {"layer_norm_epsilon", optional(number())}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.token_embedding" => "backbone.embeddings",
        "decoder.blocks.{n}.norm" => "backbone.layers.{n}.norm",
        "decoder.blocks.{n}.mixer.in_proj" => "backbone.layers.{n}.mixer.in_proj",
        "decoder.blocks.{n}.mixer.conv_proj" => "backbone.layers.{n}.mixer.conv1d",
        "decoder.blocks.{n}.mixer.out_proj" => "backbone.layers.{n}.mixer.out_proj",
        "norm" => "backbone.norm_f",
        "language_modeling_head.output" => "backbone.embeddings"
      }
    end
  end
end

defmodule Bumblebee.Vision.RegNet do
  alias Bumblebee.Shared

  options =
    [
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      embedding_size: [
        default: 32,
        doc: "the dimensionality of the embedding layer (stem)"
      ],
      hidden_sizes: [
        default: [128, 192, 512, 1088],
        doc: "the dimensionality of hidden layers at each stage"
      ],
      depths: [
        default: [2, 6, 12, 2],
        doc: "the depth (number of residual blocks) at each stage"
      ],
      groups_width: [
        default: 64,
        doc: "the group width for grouped convolutions"
      ],
      layer_type: [
        default: :y,
        doc: """
        the type of layer to use, either `:x` for RegNetX (no squeeze-and-excitation)
        or `:y` for RegNetY (with squeeze-and-excitation)
        """
      ],
      activation: [
        default: :relu,
        doc: "the activation function"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  RegNet model family.

  ## Architectures

    * `:base` - plain RegNet without any head on top

    * `:for_image_classification` - RegNet with a classification head.
      The head consists of a single dense layer on top of the pooled
      features and it returns logits corresponding to possible classes

  ## Inputs

    * `"pixel_values"` - `{batch_size, height, width, num_channels}`

      Featurized image pixel values (224x224).

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_image_classification]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(spec) do
    %{
      "pixel_values" => Nx.template({1, 224, 224, spec.num_channels}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    spec
    |> core()
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_image_classification} = spec) do
    outputs = core(spec)

    logits =
      outputs.pooled_state
      |> Axon.dense(spec.num_labels, name: "image_classification_head.output")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states
    })
  end

  defp core(spec, opts \\ []) do
    name = opts[:name]

    input = Axon.input("pixel_values", shape: {nil, 224, 224, spec.num_channels})

    encoder_outputs =
      input
      |> embedder(spec, name: join(name, "embedder"))
      |> encoder(spec, name: join(name, "encoder"))

    pooled_output =
      encoder_outputs.hidden_state
      |> Axon.adaptive_avg_pool(
        output_size: {1, 1},
        name: join(name, "pooler")
      )
      |> Axon.flatten()

    %{
      hidden_state: encoder_outputs.hidden_state,
      pooled_state: pooled_output,
      hidden_states: encoder_outputs.hidden_states
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]

    pixel_values
    |> Axon.conv(spec.embedding_size,
      kernel_size: 3,
      strides: 2,
      padding: [{1, 1}, {1, 1}],
      use_bias: false,
      kernel_initializer: conv_kernel_initializer(),
      name: join(name, "conv")
    )
    |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "norm"))
    |> Axon.activation(spec.activation, name: join(name, "activation"))
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

    stages = spec.hidden_sizes |> Enum.zip(spec.depths) |> Enum.with_index()

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      in_channels: spec.embedding_size
    }

    for {{size, depth}, idx} <- stages, reduce: state do
      state ->
        strides = if idx == 0, do: 1, else: 2

        hidden_state =
          stage(state.hidden_state, state.in_channels, size, spec,
            depth: depth,
            strides: strides,
            name: join(name, "stages.#{idx}")
          )

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          in_channels: size
        }
    end
  end

  defp stage(hidden_state, in_channels, out_channels, spec, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 2, depth: 2])
    name = opts[:name]
    strides = opts[:strides]
    depth = opts[:depth]

    # First block handles channel change and optional downsampling
    hidden_state =
      regnet_block(hidden_state, in_channels, out_channels, spec,
        strides: strides,
        name: join(name, "blocks.0")
      )

    for idx <- 1..(depth - 1)//1, reduce: hidden_state do
      hidden_state ->
        regnet_block(hidden_state, out_channels, out_channels, spec,
          name: join(name, "blocks.#{idx}")
        )
    end
  end

  defp regnet_block(hidden_state, in_channels, out_channels, spec, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1])
    name = opts[:name]
    strides = opts[:strides]

    groups = max(1, div(out_channels, spec.groups_width))

    shortcut =
      shortcut(hidden_state, in_channels, out_channels,
        strides: strides,
        name: join(name, "shortcut")
      )

    # 1x1 conv -> group conv -> optional SE -> 1x1 conv
    result =
      hidden_state
      |> conv_block(out_channels,
        kernel_size: 1,
        activation: spec.activation,
        name: join(name, "conv_blocks.0")
      )
      |> conv_block(out_channels,
        kernel_size: 3,
        strides: strides,
        groups: groups,
        activation: spec.activation,
        name: join(name, "conv_blocks.1")
      )

    result =
      if spec.layer_type == :y do
        squeeze_and_excitation(result, out_channels, spec,
          name: join(name, "squeeze_and_excitation")
        )
      else
        result
      end

    result
    |> conv_block(out_channels,
      kernel_size: 1,
      activation: :linear,
      name: join(name, "conv_blocks.2")
    )
    |> Axon.add(shortcut)
    |> Axon.activation(spec.activation, name: join(name, "activation"))
  end

  defp squeeze_and_excitation(hidden_state, channels, spec, opts) do
    name = opts[:name]
    reduced_channels = max(1, div(channels, 4))

    se =
      hidden_state
      |> Axon.adaptive_avg_pool(output_size: {1, 1}, name: join(name, "pool"))
      |> Axon.conv(reduced_channels,
        kernel_size: 1,
        name: join(name, "reduce")
      )
      |> Axon.activation(spec.activation, name: join(name, "reduce_activation"))
      |> Axon.conv(channels,
        kernel_size: 1,
        name: join(name, "expand")
      )
      |> Axon.sigmoid(name: join(name, "sigmoid"))

    Axon.multiply(hidden_state, se)
  end

  defp shortcut(hidden_state, in_channels, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, strides: 1])
    name = opts[:name]
    strides = opts[:strides]

    project_shortcut? = in_channels != out_channels or strides != 1

    if project_shortcut? do
      hidden_state
      |> Axon.conv(out_channels,
        kernel_size: 1,
        strides: strides,
        use_bias: false,
        kernel_initializer: conv_kernel_initializer(),
        name: join(name, "projection")
      )
      |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "norm"))
    else
      hidden_state
    end
  end

  defp conv_block(hidden_state, out_channels, opts) do
    opts = Keyword.validate!(opts, [:name, kernel_size: 3, strides: 1, groups: 1, activation: :relu])
    name = opts[:name]
    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    groups = opts[:groups]
    activation = opts[:activation]

    edge_padding = div(kernel_size, 2)
    padding_spec = [{edge_padding, edge_padding}, {edge_padding, edge_padding}]

    hidden_state
    |> Axon.conv(out_channels,
      kernel_size: kernel_size,
      strides: strides,
      padding: padding_spec,
      feature_group_size: groups,
      use_bias: false,
      kernel_initializer: conv_kernel_initializer(),
      name: join(name, "conv")
    )
    |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "norm"))
    |> Axon.activation(activation, name: join(name, "activation"))
  end

  defp conv_kernel_initializer() do
    Axon.Initializers.variance_scaling(scale: 2.0, mode: :fan_out, distribution: :normal)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          num_channels: {"num_channels", number()},
          embedding_size: {"embedding_size", number()},
          hidden_sizes: {"hidden_sizes", list(number())},
          depths: {"depths", list(number())},
          groups_width: {"groups_width", number()},
          layer_type: {"layer_type", atom()},
          activation: {"hidden_act", activation()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.conv" => "regnet.embedder.embedder.convolution",
        "embedder.norm" => "regnet.embedder.embedder.normalization",
        "encoder.stages.{n}.blocks.{m}.conv_blocks.{l}.conv" =>
          "regnet.encoder.stages.{n}.layers.{m}.layer.{l}.convolution",
        "encoder.stages.{n}.blocks.{m}.conv_blocks.{l}.norm" =>
          "regnet.encoder.stages.{n}.layers.{m}.layer.{l}.normalization",
        "encoder.stages.{n}.blocks.{m}.squeeze_and_excitation.reduce" =>
          "regnet.encoder.stages.{n}.layers.{m}.attention.0",
        "encoder.stages.{n}.blocks.{m}.squeeze_and_excitation.expand" =>
          "regnet.encoder.stages.{n}.layers.{m}.attention.2",
        "encoder.stages.{n}.blocks.{m}.shortcut.projection" =>
          "regnet.encoder.stages.{n}.layers.{m}.shortcut.convolution",
        "encoder.stages.{n}.blocks.{m}.shortcut.norm" =>
          "regnet.encoder.stages.{n}.layers.{m}.shortcut.normalization",
        "image_classification_head.output" => "classifier.1"
      }
    end
  end
end

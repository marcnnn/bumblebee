defmodule Bumblebee.Vision.EfficientNet do
  alias Bumblebee.Shared

  options =
    [
      hidden_size: [
        default: 1280,
        doc: "the dimensionality of the final hidden layer"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      image_size: [
        default: 224,
        doc: "the size of the input spatial dimensions"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.2,
        doc: "the dropout rate for the classification head"
      ],
      embedding_size: [
        default: 32,
        doc: "the dimensionality of the first convolutional layer"
      ],
      hidden_sizes: [
        default: [16, 24, 40, 80, 112, 192, 320],
        doc: "the dimensionality of hidden layers at each stage"
      ],
      depths: [
        default: [1, 2, 2, 3, 3, 4, 1],
        doc: "the depth (number of blocks) at each stage"
      ],
      kernel_sizes: [
        default: [3, 3, 5, 3, 5, 5, 3],
        doc: "the kernel sizes at each stage"
      ],
      strides: [
        default: [1, 2, 2, 2, 1, 2, 1],
        doc: "the strides at each stage"
      ],
      expand_ratios: [
        default: [1, 6, 6, 6, 6, 6, 6],
        doc: "the expansion ratios at each stage"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  EfficientNet model family.

  ## Architectures

    * `:base` - plain EfficientNet without any head on top

    * `:for_image_classification` - EfficientNet with a classification head.
      The head consists of a single dense layer on top of the pooled
      features and it returns logits corresponding to possible classes

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

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
      "pixel_values" =>
        Nx.template({1, spec.image_size, spec.image_size, spec.num_channels}, :f32)
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
      |> Axon.dropout(rate: spec.dropout_rate)
      |> Axon.dense(spec.num_labels, name: "image_classification_head.output")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states
    })
  end

  defp core(spec, opts \\ []) do
    name = opts[:name]

    input = Axon.input("pixel_values", shape: {nil, spec.image_size, spec.image_size, spec.num_channels})

    encoder_outputs =
      input
      |> embedder(spec, name: join(name, "embedder"))
      |> encoder(spec, name: join(name, "encoder"))

    # Final top conv layer
    last_hidden_size = List.last(spec.hidden_sizes)

    hidden_state =
      encoder_outputs.hidden_state
      |> Axon.conv(spec.hidden_size,
        kernel_size: 1,
        use_bias: false,
        kernel_initializer: conv_kernel_initializer(),
        name: join(name, "top_conv")
      )
      |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "top_norm"))
      |> Axon.activation(spec.activation)

    pooled_output =
      hidden_state
      |> Axon.adaptive_avg_pool(output_size: {1, 1}, name: join(name, "pooler"))
      |> Axon.flatten()

    %{
      hidden_state: hidden_state,
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
    |> Axon.activation(spec.activation)
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

    stages =
      spec.hidden_sizes
      |> Enum.zip(spec.depths)
      |> Enum.zip(spec.kernel_sizes)
      |> Enum.zip(spec.strides)
      |> Enum.zip(spec.expand_ratios)
      |> Enum.with_index()

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      in_channels: spec.embedding_size
    }

    for {{{{{{out_channels, depth}, kernel_size}, stride}, expand_ratio}, idx} <- stages,
        reduce: state do
      state ->
        hidden_state =
          mbconv_stage(state.hidden_state, state.in_channels, out_channels, spec,
            depth: depth,
            kernel_size: kernel_size,
            stride: stride,
            expand_ratio: expand_ratio,
            name: join(name, "stages.#{idx}")
          )

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state),
          in_channels: out_channels
        }
    end
  end

  defp mbconv_stage(hidden_state, in_channels, out_channels, spec, opts) do
    name = opts[:name]
    depth = opts[:depth]
    kernel_size = opts[:kernel_size]
    stride = opts[:stride]
    expand_ratio = opts[:expand_ratio]

    # First block may have stride > 1
    hidden_state =
      mbconv_block(hidden_state, in_channels, out_channels, spec,
        kernel_size: kernel_size,
        stride: stride,
        expand_ratio: expand_ratio,
        name: join(name, "blocks.0")
      )

    for idx <- 1..(depth - 1)//1, reduce: hidden_state do
      hidden_state ->
        mbconv_block(hidden_state, out_channels, out_channels, spec,
          kernel_size: kernel_size,
          stride: 1,
          expand_ratio: expand_ratio,
          name: join(name, "blocks.#{idx}")
        )
    end
  end

  defp mbconv_block(hidden_state, in_channels, out_channels, spec, opts) do
    name = opts[:name]
    kernel_size = opts[:kernel_size]
    stride = opts[:stride]
    expand_ratio = opts[:expand_ratio]

    expanded_channels = in_channels * expand_ratio
    edge_padding = div(kernel_size, 2)

    residual = hidden_state

    # Expansion phase
    hidden_state =
      if expand_ratio != 1 do
        hidden_state
        |> Axon.conv(expanded_channels,
          kernel_size: 1,
          use_bias: false,
          kernel_initializer: conv_kernel_initializer(),
          name: join(name, "expand_conv")
        )
        |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "expand_norm"))
        |> Axon.activation(spec.activation)
      else
        hidden_state
      end

    # Depthwise convolution
    hidden_state =
      hidden_state
      |> Axon.depthwise_conv(1,
        kernel_size: kernel_size,
        strides: stride,
        padding: [{edge_padding, edge_padding}, {edge_padding, edge_padding}],
        use_bias: false,
        kernel_initializer: conv_kernel_initializer(),
        name: join(name, "depthwise_conv")
      )
      |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "depthwise_norm"))
      |> Axon.activation(spec.activation)

    # Squeeze and Excitation
    hidden_state = squeeze_excite(hidden_state, expanded_channels, in_channels, spec, name: join(name, "se"))

    # Output phase
    hidden_state =
      hidden_state
      |> Axon.conv(out_channels,
        kernel_size: 1,
        use_bias: false,
        kernel_initializer: conv_kernel_initializer(),
        name: join(name, "project_conv")
      )
      |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "project_norm"))

    # Skip connection
    if stride == 1 and in_channels == out_channels do
      Axon.add(hidden_state, residual)
    else
      hidden_state
    end
  end

  defp squeeze_excite(hidden_state, channels, reduced_channels, _spec, opts) do
    name = opts[:name]
    se_channels = max(1, div(reduced_channels, 4))

    se =
      hidden_state
      |> Axon.adaptive_avg_pool(output_size: {1, 1}, name: join(name, "pool"))
      |> Axon.conv(se_channels,
        kernel_size: 1,
        name: join(name, "reduce")
      )
      |> Axon.activation(:silu)
      |> Axon.conv(channels,
        kernel_size: 1,
        name: join(name, "expand")
      )
      |> Axon.sigmoid()

    Axon.multiply(hidden_state, se)
  end

  defp conv_kernel_initializer() do
    Axon.Initializers.variance_scaling(scale: 2.0, mode: :fan_out, distribution: :normal)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          hidden_size: {"hidden_dim", number()},
          num_channels: {"num_channels", number()},
          image_size: {"image_size", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"dropout_rate", number()},
          embedding_size: {"embedding_size", number()},
          hidden_sizes: {"hidden_sizes", list(number())},
          depths: {"depths", list(number())},
          kernel_sizes: {"kernel_sizes", list(number())},
          strides: {"strides", list(number())},
          expand_ratios: {"expand_ratios", list(number())}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.conv" => "efficientnet.embeddings.convolution",
        "embedder.norm" => "efficientnet.embeddings.batchnorm",
        "encoder.stages.{n}.blocks.{m}.expand_conv" =>
          "efficientnet.encoder.blocks.{n}.{m}.expansion.expand_conv",
        "encoder.stages.{n}.blocks.{m}.expand_norm" =>
          "efficientnet.encoder.blocks.{n}.{m}.expansion.expand_bn",
        "encoder.stages.{n}.blocks.{m}.depthwise_conv" =>
          "efficientnet.encoder.blocks.{n}.{m}.depthwise_conv.depthwise_conv",
        "encoder.stages.{n}.blocks.{m}.depthwise_norm" =>
          "efficientnet.encoder.blocks.{n}.{m}.depthwise_conv.depthwise_bn",
        "encoder.stages.{n}.blocks.{m}.se.reduce" =>
          "efficientnet.encoder.blocks.{n}.{m}.squeeze_excite.reduce",
        "encoder.stages.{n}.blocks.{m}.se.expand" =>
          "efficientnet.encoder.blocks.{n}.{m}.squeeze_excite.expand",
        "encoder.stages.{n}.blocks.{m}.project_conv" =>
          "efficientnet.encoder.blocks.{n}.{m}.projection.project_conv",
        "encoder.stages.{n}.blocks.{m}.project_norm" =>
          "efficientnet.encoder.blocks.{n}.{m}.projection.project_bn",
        "top_conv" => "efficientnet.top_conv",
        "top_norm" => "efficientnet.top_bn",
        "image_classification_head.output" => "classifier"
      }
    end
  end
end

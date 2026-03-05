defmodule Bumblebee.Vision.MobileVit do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: 256,
        doc: "the size of the input spatial dimensions"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_size: [
        default: 2,
        doc: "the size of the patch spatial dimensions"
      ],
      hidden_size: [
        default: 640,
        doc: "the dimensionality of the final hidden layer"
      ],
      num_blocks: [
        default: 3,
        doc: "the number of Transformer blocks in each MobileViT layer"
      ],
      num_attention_heads: [
        default: 4,
        doc: "the number of attention heads for each attention layer"
      ],
      intermediate_size: [
        default: 2560,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN)"
      ],
      neck_hidden_sizes: [
        default: [16, 32, 64, 96, 128, 160, 640],
        doc: "the hidden sizes for each convolutional stage in the network"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate"
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
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  MobileViT model family.

  ## Architectures

    * `:base` - plain MobileViT without any head on top

    * `:for_image_classification` - MobileViT with a classification head.
      The head consists of a single dense layer on top of the pooled
      features and it returns logits corresponding to possible classes

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

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
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "image_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp core(spec, opts \\ []) do
    name = opts[:name]

    input =
      Axon.input("pixel_values",
        shape: {nil, spec.image_size, spec.image_size, spec.num_channels}
      )

    neck_sizes = spec.neck_hidden_sizes

    # Initial conv stages (MobileNetV2-like)
    hidden_state =
      input
      |> conv_block(Enum.at(neck_sizes, 0),
        kernel_size: 3,
        strides: 2,
        activation: spec.activation,
        name: join(name, "conv_stem")
      )
      |> inverted_residual(Enum.at(neck_sizes, 0), Enum.at(neck_sizes, 1),
        stride: 1,
        activation: spec.activation,
        name: join(name, "encoder.layer.0")
      )
      |> inverted_residual(Enum.at(neck_sizes, 1), Enum.at(neck_sizes, 2),
        stride: 2,
        activation: spec.activation,
        name: join(name, "encoder.layer.1")
      )
      |> inverted_residual(Enum.at(neck_sizes, 2), Enum.at(neck_sizes, 3),
        stride: 2,
        activation: spec.activation,
        name: join(name, "encoder.layer.2")
      )

    # MobileViT blocks with transformer
    hidden_state =
      hidden_state
      |> mobilevit_layer(Enum.at(neck_sizes, 3), Enum.at(neck_sizes, 4), spec,
        stride: 2,
        name: join(name, "encoder.layer.3")
      )
      |> mobilevit_layer(Enum.at(neck_sizes, 4), Enum.at(neck_sizes, 5), spec,
        stride: 2,
        name: join(name, "encoder.layer.4")
      )

    # Final conv
    hidden_state =
      conv_block(hidden_state, spec.hidden_size,
        kernel_size: 1,
        activation: spec.activation,
        name: join(name, "conv_1x1_exp")
      )

    pooled_output =
      hidden_state
      |> Axon.adaptive_avg_pool(output_size: {1, 1}, name: join(name, "pooler"))
      |> Axon.flatten()

    %{
      hidden_state: hidden_state,
      pooled_state: pooled_output,
      hidden_states: Layers.none(),
      attentions: Layers.none()
    }
  end

  defp conv_block(hidden_state, out_channels, opts) do
    name = opts[:name]
    kernel_size = opts[:kernel_size] || 3
    strides = opts[:strides] || 1
    activation = opts[:activation] || :silu

    edge_padding = div(kernel_size, 2)

    hidden_state
    |> Axon.conv(out_channels,
      kernel_size: kernel_size,
      strides: strides,
      padding: [{edge_padding, edge_padding}, {edge_padding, edge_padding}],
      use_bias: false,
      kernel_initializer: conv_kernel_initializer(),
      name: join(name, "conv")
    )
    |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "norm"))
    |> Axon.activation(activation, name: join(name, "activation"))
  end

  defp inverted_residual(hidden_state, in_channels, out_channels, opts) do
    name = opts[:name]
    stride = opts[:stride] || 1
    activation = opts[:activation] || :silu
    expand_ratio = opts[:expand_ratio] || 1

    expanded_channels = in_channels * expand_ratio

    residual = hidden_state

    # Expand
    hidden_state =
      if expand_ratio != 1 do
        conv_block(hidden_state, expanded_channels,
          kernel_size: 1,
          activation: activation,
          name: join(name, "expand")
        )
      else
        hidden_state
      end

    # Depthwise
    edge_padding = 1

    hidden_state =
      hidden_state
      |> Axon.depthwise_conv(1,
        kernel_size: 3,
        strides: stride,
        padding: [{edge_padding, edge_padding}, {edge_padding, edge_padding}],
        use_bias: false,
        kernel_initializer: conv_kernel_initializer(),
        name: join(name, "depthwise.conv")
      )
      |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "depthwise.norm"))
      |> Axon.activation(activation)

    # Project
    hidden_state =
      hidden_state
      |> Axon.conv(out_channels,
        kernel_size: 1,
        use_bias: false,
        kernel_initializer: conv_kernel_initializer(),
        name: join(name, "project.conv")
      )
      |> Axon.batch_norm(gamma_initializer: :ones, name: join(name, "project.norm"))

    if stride == 1 and in_channels == out_channels do
      Axon.add(hidden_state, residual)
    else
      hidden_state
    end
  end

  defp mobilevit_layer(hidden_state, in_channels, out_channels, spec, opts) do
    name = opts[:name]
    stride = opts[:stride] || 1

    # Local representation via convolution
    hidden_state =
      if stride > 1 do
        inverted_residual(hidden_state, in_channels, in_channels,
          stride: stride,
          activation: spec.activation,
          name: join(name, "downsampling")
        )
      else
        hidden_state
      end

    hidden_state =
      conv_block(hidden_state, out_channels,
        kernel_size: 1,
        activation: spec.activation,
        name: join(name, "conv_kxk")
      )

    # Unfold to patches, apply transformer, fold back
    hidden_state =
      hidden_state
      |> Axon.reshape({:batch, :auto, out_channels}, name: join(name, "unfold"))

    transformer_outputs =
      Layers.Transformer.blocks(hidden_state,
        num_blocks: spec.num_blocks,
        num_attention_heads: spec.num_attention_heads,
        hidden_size: out_channels,
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
        name: join(name, "transformer")
      )

    transformer_outputs.hidden_state
    |> Axon.layer_norm(
      epsilon: spec.layer_norm_epsilon,
      name: join(name, "layernorm")
    )
    |> Axon.layer(
      fn hidden_state, original, _opts ->
        {batch, h, w, _c} = Nx.shape(original)
        Nx.reshape(hidden_state, {batch, h, w, out_channels})
      end,
      [hidden_state],
      name: join(name, "fold")
    )
    |> conv_block(out_channels,
      kernel_size: 1,
      activation: spec.activation,
      name: join(name, "conv_projection")
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defp conv_kernel_initializer() do
    Axon.Initializers.variance_scaling(scale: 2.0, mode: :fan_out, distribution: :normal)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          image_size: {"image_size", number()},
          num_channels: {"num_channels", number()},
          patch_size: {"patch_size", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_transformer_blocks", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"mlp_ratio", number()},
          neck_hidden_sizes: {"neck_hidden_sizes", list(number())},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "conv_stem.conv" => "mobilevit.conv_stem.convolution",
        "conv_stem.norm" => "mobilevit.conv_stem.normalization",
        "encoder.layer.{n}.expand.conv" =>
          "mobilevit.encoder.layer.{n}.expand_1x1.convolution",
        "encoder.layer.{n}.expand.norm" =>
          "mobilevit.encoder.layer.{n}.expand_1x1.normalization",
        "encoder.layer.{n}.depthwise.conv" =>
          "mobilevit.encoder.layer.{n}.conv_kxk.convolution",
        "encoder.layer.{n}.depthwise.norm" =>
          "mobilevit.encoder.layer.{n}.conv_kxk.normalization",
        "encoder.layer.{n}.project.conv" =>
          "mobilevit.encoder.layer.{n}.reduce_1x1.convolution",
        "encoder.layer.{n}.project.norm" =>
          "mobilevit.encoder.layer.{n}.reduce_1x1.normalization",
        "encoder.layer.{n}.downsampling.depthwise.conv" =>
          "mobilevit.encoder.layer.{n}.downsampling_layer.conv_kxk.convolution",
        "encoder.layer.{n}.downsampling.depthwise.norm" =>
          "mobilevit.encoder.layer.{n}.downsampling_layer.conv_kxk.normalization",
        "encoder.layer.{n}.downsampling.project.conv" =>
          "mobilevit.encoder.layer.{n}.downsampling_layer.conv_1x1.convolution",
        "encoder.layer.{n}.downsampling.project.norm" =>
          "mobilevit.encoder.layer.{n}.downsampling_layer.conv_1x1.normalization",
        "encoder.layer.{n}.conv_kxk.conv" =>
          "mobilevit.encoder.layer.{n}.pre_conv_kxk.convolution",
        "encoder.layer.{n}.conv_kxk.norm" =>
          "mobilevit.encoder.layer.{n}.pre_conv_kxk.normalization",
        "encoder.layer.{n}.transformer.blocks.{m}.self_attention_norm" =>
          "mobilevit.encoder.layer.{n}.transformer.layer.{m}.layernorm_before",
        "encoder.layer.{n}.transformer.blocks.{m}.self_attention.query" =>
          "mobilevit.encoder.layer.{n}.transformer.layer.{m}.attention.attention.query",
        "encoder.layer.{n}.transformer.blocks.{m}.self_attention.key" =>
          "mobilevit.encoder.layer.{n}.transformer.layer.{m}.attention.attention.key",
        "encoder.layer.{n}.transformer.blocks.{m}.self_attention.value" =>
          "mobilevit.encoder.layer.{n}.transformer.layer.{m}.attention.attention.value",
        "encoder.layer.{n}.transformer.blocks.{m}.self_attention.output" =>
          "mobilevit.encoder.layer.{n}.transformer.layer.{m}.attention.output.dense",
        "encoder.layer.{n}.transformer.blocks.{m}.output_norm" =>
          "mobilevit.encoder.layer.{n}.transformer.layer.{m}.layernorm_after",
        "encoder.layer.{n}.transformer.blocks.{m}.ffn.intermediate" =>
          "mobilevit.encoder.layer.{n}.transformer.layer.{m}.intermediate.dense",
        "encoder.layer.{n}.transformer.blocks.{m}.ffn.output" =>
          "mobilevit.encoder.layer.{n}.transformer.layer.{m}.output.dense",
        "encoder.layer.{n}.layernorm" =>
          "mobilevit.encoder.layer.{n}.layernorm",
        "encoder.layer.{n}.conv_projection.conv" =>
          "mobilevit.encoder.layer.{n}.conv_projection.convolution",
        "encoder.layer.{n}.conv_projection.norm" =>
          "mobilevit.encoder.layer.{n}.conv_projection.normalization",
        "conv_1x1_exp.conv" => "mobilevit.conv_1x1_exp.convolution",
        "conv_1x1_exp.norm" => "mobilevit.conv_1x1_exp.normalization",
        "image_classification_head.output" => "classifier"
      }
    end
  end
end

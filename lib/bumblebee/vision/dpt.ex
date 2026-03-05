defmodule Bumblebee.Vision.Dpt do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: 384,
        doc: "the size of the input spatial dimensions"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_size: [
        default: 16,
        doc: "the size of the patch spatial dimensions"
      ],
      hidden_size: [
        default: 768,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 12,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      intermediate_size: [
        default: 3072,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      neck_hidden_sizes: [
        default: [96, 192, 384, 768],
        doc: "the hidden sizes of the reassemble layers in the neck"
      ],
      use_attention_bias: [
        default: true,
        doc: "whether to use bias in query, key, and value projections"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for encoder and decoder"
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
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  DPT model family.

  ## Architectures

    * `:base` - plain DPT without any head on top

    * `:for_depth_estimation` - DPT with a depth estimation head on
      top for monocular depth prediction

    * `:for_semantic_segmentation` - DPT with a semantic segmentation
      head on top

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_depth_estimation, :for_semantic_segmentation]

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
    |> inputs()
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_depth_estimation} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    half_size = div(spec.image_size, 2)

    # DPT depth estimation head: conv -> upsample -> conv -> relu -> conv
    predicted_depth =
      outputs.hidden_state
      |> Axon.conv(div(spec.hidden_size, 2),
        kernel_size: 3,
        padding: [{1, 1}, {1, 1}],
        name: "depth_estimation_head.conv1"
      )
      |> Axon.layer(
        fn x, _opts ->
          Axon.Layers.resize(x,
            size: {half_size, half_size},
            method: :bilinear,
            channels: :last
          )
        end,
        [],
        name: "depth_estimation_head.upsample"
      )
      |> Axon.conv(32,
        kernel_size: 3,
        padding: [{1, 1}, {1, 1}],
        name: "depth_estimation_head.conv2"
      )
      |> Axon.activation(:relu)
      |> Axon.conv(1,
        kernel_size: 1,
        name: "depth_estimation_head.output"
      )
      |> Axon.nx(fn x -> Nx.squeeze(x, axes: [-1]) end)

    Layers.output(%{
      predicted_depth: predicted_depth,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_semantic_segmentation} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Axon.conv(spec.num_labels,
        kernel_size: 1,
        name: "semantic_segmentation_head.output"
      )
      |> Axon.layer(
        fn x, _opts ->
          Axon.Layers.resize(x,
            size: {spec.image_size, spec.image_size},
            method: :bilinear,
            channels: :last
          )
        end,
        [],
        name: "semantic_segmentation_head.upsample"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape)
    ])
  end

  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    embeddings =
      embedder(inputs["pixel_values"], spec, name: join(name, "embedder"))

    encoder_outputs = encoder(embeddings, spec, name: join(name, "encoder"))

    hidden_state =
      Axon.layer_norm(encoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "norm")
      )

    # DPT neck: reassemble features from encoder and fuse them
    neck_output = neck(hidden_state, spec, name: join(name, "neck"))

    %{
      hidden_state: neck_output,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]

    patch_embeddings =
      pixel_values
      |> patch_embedding(spec, name: join(name, "patch_embedding"))

    class_embedding =
      Layers.learned_embeddings(1, spec.hidden_size, name: join(name, "class_embedding"))

    input_embeddings = Layers.concatenate_embeddings([class_embedding, patch_embeddings])

    num_patches = div(spec.image_size, spec.patch_size) ** 2

    position_embeddings =
      Layers.learned_embeddings(num_patches + 1, spec.hidden_size,
        initializer: :zeros,
        name: join(name, "position_embedding")
      )

    Axon.add(input_embeddings, position_embeddings)
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]

    pixel_values
    |> Axon.conv(spec.hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "projection")
    )
    |> Axon.reshape({:batch, :auto, spec.hidden_size}, name: join(name, "reshape"))
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

    Layers.Transformer.blocks(hidden_state,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      dropout_rate: spec.dropout_rate,
      attention_dropout_rate: spec.attention_dropout_rate,
      query_use_bias: spec.use_attention_bias,
      key_use_bias: spec.use_attention_bias,
      value_use_bias: spec.use_attention_bias,
      layer_norm: [
        epsilon: spec.layer_norm_epsilon
      ],
      ffn: [
        intermediate_size: spec.intermediate_size,
        activation: spec.activation
      ],
      block_type: :norm_first,
      name: join(name, "blocks")
    )
  end

  defp neck(hidden_state, spec, opts) do
    name = opts[:name]

    # Remove CLS token and reshape to spatial
    spatial_size = div(spec.image_size, spec.patch_size)

    features =
      Axon.nx(hidden_state, fn x ->
        # Remove CLS token
        x = x[[.., 1..-1//1, ..]]
        {batch, _seq, channels} = Nx.shape(x)
        Nx.reshape(x, {batch, spatial_size, spatial_size, channels})
      end)

    # Reassemble: project to neck hidden size and progressively upsample
    [neck_size | _] = Enum.reverse(spec.neck_hidden_sizes)

    features
    |> Axon.conv(neck_size,
      kernel_size: 3,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "reassemble_conv")
    )
    |> Axon.activation(:relu)
    |> Axon.conv(neck_size,
      kernel_size: 3,
      padding: [{1, 1}, {1, 1}],
      name: join(name, "fusion_conv")
    )
    |> Axon.activation(:relu)
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
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
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          neck_hidden_sizes: {"neck_hidden_sizes", list(number())},
          activation: {"hidden_act", activation()},
          use_attention_bias: {"qkv_bias", boolean()},
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
        "embedder.patch_embedding.projection" =>
          "dpt.embeddings.patch_embeddings.projection",
        "embedder.class_embedding" => %{
          "embeddings" => {
            [{"dpt.embeddings", "cls_token"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "embedder.position_embedding" => %{
          "embeddings" => {
            [{"dpt.embeddings", "position_embeddings"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "encoder.blocks.{n}.self_attention_norm" =>
          "dpt.encoder.layer.{n}.layernorm_before",
        "encoder.blocks.{n}.self_attention.key" =>
          "dpt.encoder.layer.{n}.attention.attention.key",
        "encoder.blocks.{n}.self_attention.query" =>
          "dpt.encoder.layer.{n}.attention.attention.query",
        "encoder.blocks.{n}.self_attention.value" =>
          "dpt.encoder.layer.{n}.attention.attention.value",
        "encoder.blocks.{n}.self_attention.output" =>
          "dpt.encoder.layer.{n}.attention.output.dense",
        "encoder.blocks.{n}.ffn.intermediate" =>
          "dpt.encoder.layer.{n}.intermediate.dense",
        "encoder.blocks.{n}.ffn.output" =>
          "dpt.encoder.layer.{n}.output.dense",
        "encoder.blocks.{n}.output_norm" =>
          "dpt.encoder.layer.{n}.layernorm_after",
        "norm" => "dpt.layernorm",
        "neck.reassemble_conv" => "neck.reassemble_stage.layers.0.projection",
        "neck.fusion_conv" => "neck.fusion_stage.layers.0.residual_layer1.convolution",
        "depth_estimation_head.conv1" => "head.projection",
        "depth_estimation_head.conv2" => "head.head.0",
        "depth_estimation_head.output" => "head.head.2",
        "semantic_segmentation_head.output" => "head"
      }
    end
  end
end

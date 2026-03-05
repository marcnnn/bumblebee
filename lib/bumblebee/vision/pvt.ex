defmodule Bumblebee.Vision.Pvt do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: 224,
        doc: "the size of the input spatial dimensions"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_sizes: [
        default: [7, 3, 3, 3],
        doc: "the patch sizes for the embedding layer at each stage"
      ],
      strides: [
        default: [4, 2, 2, 2],
        doc: "the strides for the embedding layer at each stage"
      ],
      hidden_sizes: [
        default: [64, 128, 320, 512],
        doc: "the dimensionality of hidden layers at each stage"
      ],
      num_blocks: [
        default: [3, 4, 6, 3],
        doc: "the number of Transformer blocks at each stage"
      ],
      num_attention_heads: [
        default: [1, 2, 5, 8],
        doc: "the number of attention heads for each attention layer at each stage"
      ],
      intermediate_ratios: [
        default: [8, 8, 4, 4],
        doc: "the expansion ratio for the intermediate layer in the FFN at each stage"
      ],
      sr_ratios: [
        default: [8, 4, 2, 1],
        doc: "the spatial reduction ratios for each stage"
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
        default: 1.0e-6,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  PVT (Pyramid Vision Transformer) model family.

  ## Architectures

    * `:base` - plain PVT without any head on top

    * `:for_image_classification` - PVT with a classification head.
      The head consists of a single dense layer on top of the average-pooled
      features

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)

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
    |> inputs()
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_image_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Axon.adaptive_avg_pool(output_size: {1, 1}, name: "image_classification_head.pool")
      |> Axon.flatten(name: "image_classification_head.flatten")
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

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape)
    ])
  end

  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    encoder_outputs = encoder(inputs["pixel_values"], spec, name: join(name, "encoder"))

    %{
      hidden_state: encoder_outputs.hidden_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp encoder(pixel_values, spec, opts) do
    name = opts[:name]

    num_stages = length(spec.hidden_sizes)

    state = %{
      hidden_state: pixel_values,
      hidden_states: Layers.none(),
      attentions: Layers.none()
    }

    result =
      for stage_idx <- 0..(num_stages - 1), reduce: state do
        state ->
          hidden_size = Enum.at(spec.hidden_sizes, stage_idx)
          num_blocks = Enum.at(spec.num_blocks, stage_idx)
          num_heads = Enum.at(spec.num_attention_heads, stage_idx)
          patch_size = Enum.at(spec.patch_sizes, stage_idx)
          stride = Enum.at(spec.strides, stage_idx)
          intermediate_ratio = Enum.at(spec.intermediate_ratios, stage_idx)
          intermediate_size = hidden_size * intermediate_ratio

          # Patch embedding for this stage
          embedded =
            patch_embedding(state.hidden_state, hidden_size,
              patch_size: patch_size,
              stride: stride,
              kernel_initializer: kernel_initializer(spec),
              layer_norm_epsilon: spec.layer_norm_epsilon,
              name: join(name, "patch_embeddings.#{stage_idx}")
            )

          # Transformer blocks for this stage
          block_outputs =
            Layers.Transformer.blocks(embedded,
              num_blocks: num_blocks,
              num_attention_heads: num_heads,
              hidden_size: hidden_size,
              kernel_initializer: kernel_initializer(spec),
              dropout_rate: spec.dropout_rate,
              attention_dropout_rate: spec.attention_dropout_rate,
              layer_norm: [
                epsilon: spec.layer_norm_epsilon
              ],
              ffn: [
                intermediate_size: intermediate_size,
                activation: :gelu
              ],
              block_type: :norm_first,
              name: join(name, "block.#{stage_idx}")
            )

          # Reshape back to spatial for next stage
          stage_hidden_state =
            block_outputs.hidden_state
            |> Axon.layer_norm(
              epsilon: spec.layer_norm_epsilon,
              name: join(name, "layer_norms.#{stage_idx}")
            )
            |> reshape_to_spatial(state.hidden_state, stride,
              hidden_size: hidden_size,
              name: join(name, "reshape.#{stage_idx}")
            )

          %{
            hidden_state: stage_hidden_state,
            hidden_states: block_outputs.hidden_states,
            attentions: block_outputs.attentions
          }
      end

    %{
      hidden_state: result.hidden_state,
      hidden_states: result.hidden_states,
      attentions: result.attentions
    }
  end

  defp patch_embedding(hidden_state, hidden_size, opts) do
    name = opts[:name]
    patch_size = opts[:patch_size]
    stride = opts[:stride]
    kernel_initializer = opts[:kernel_initializer]
    layer_norm_epsilon = opts[:layer_norm_epsilon]

    edge_padding = div(patch_size, 2)
    padding_spec = [{edge_padding, edge_padding}, {edge_padding, edge_padding}]

    hidden_state
    |> Axon.conv(hidden_size,
      kernel_size: patch_size,
      strides: stride,
      padding: padding_spec,
      kernel_initializer: kernel_initializer,
      name: join(name, "projection")
    )
    |> Axon.reshape({:batch, :auto, hidden_size}, name: join(name, "reshape"))
    |> Axon.layer_norm(epsilon: layer_norm_epsilon, name: join(name, "norm"))
  end

  defp reshape_to_spatial(hidden_state, input_before_stage, stride, opts) do
    name = opts[:name]
    hidden_size = opts[:hidden_size]

    Axon.layer(
      fn hidden_state, input_before_stage, _opts ->
        {_batch, h, w, _c} = Nx.shape(input_before_stage)
        new_h = div(h, stride)
        new_w = div(w, stride)
        Nx.reshape(hidden_state, {:auto, new_h, new_w, hidden_size})
      end,
      [hidden_state, input_before_stage],
      name: name
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
          image_size: {"image_size", number()},
          num_channels: {"num_channels", number()},
          hidden_sizes: {"hidden_sizes", list(number())},
          num_blocks: {"depths", list(number())},
          num_attention_heads: {"num_attention_heads", list(number())},
          patch_sizes: {"patch_sizes", list(number())},
          strides: {"strides", list(number())},
          sr_ratios: {"sr_ratios", list(number())},
          intermediate_ratios: {"mlp_ratios", list(number())},
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
        "encoder.patch_embeddings.{s}.projection" =>
          "pvt.encoder.patch_embeddings.{s}.projection",
        "encoder.patch_embeddings.{s}.norm" =>
          "pvt.encoder.patch_embeddings.{s}.layer_norm",
        "encoder.block.{s}.blocks.{n}.self_attention_norm" =>
          "pvt.encoder.block.{s}.{n}.layer_norm_1",
        "encoder.block.{s}.blocks.{n}.self_attention.query" =>
          "pvt.encoder.block.{s}.{n}.attention.self.query",
        "encoder.block.{s}.blocks.{n}.self_attention.key" =>
          "pvt.encoder.block.{s}.{n}.attention.self.key",
        "encoder.block.{s}.blocks.{n}.self_attention.value" =>
          "pvt.encoder.block.{s}.{n}.attention.self.value",
        "encoder.block.{s}.blocks.{n}.self_attention.output" =>
          "pvt.encoder.block.{s}.{n}.attention.output.dense",
        "encoder.block.{s}.blocks.{n}.output_norm" =>
          "pvt.encoder.block.{s}.{n}.layer_norm_2",
        "encoder.block.{s}.blocks.{n}.ffn.intermediate" =>
          "pvt.encoder.block.{s}.{n}.mlp.fc1",
        "encoder.block.{s}.blocks.{n}.ffn.output" =>
          "pvt.encoder.block.{s}.{n}.mlp.fc2",
        "encoder.layer_norms.{s}" => "pvt.encoder.layer_norm.{s}",
        "image_classification_head.output" => "classifier"
      }
    end
  end
end

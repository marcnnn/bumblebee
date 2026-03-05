defmodule Bumblebee.Vision.Segformer do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: 512,
        doc: "the size of the input spatial dimensions"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      hidden_sizes: [
        default: [32, 64, 160, 256],
        doc: "the dimensionality of hidden layers at each stage"
      ],
      num_blocks: [
        default: [2, 2, 2, 2],
        doc: "the number of Transformer blocks at each stage"
      ],
      num_attention_heads: [
        default: [1, 2, 5, 8],
        doc: "the number of attention heads for each attention layer at each stage"
      ],
      patch_sizes: [
        default: [7, 3, 3, 3],
        doc: "the patch sizes for the embedding layer at each stage"
      ],
      strides: [
        default: [4, 2, 2, 2],
        doc: "the strides for the embedding layer at each stage"
      ],
      sr_ratios: [
        default: [8, 4, 2, 1],
        doc: "the spatial reduction ratios for each stage"
      ],
      intermediate_ratios: [
        default: [4, 4, 4, 4],
        doc: "the expansion ratio for the intermediate layer in the FFN at each stage"
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
        default: 1.0e-6,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      decoder_hidden_size: [
        default: 256,
        doc: "the hidden size for the decoder head used in semantic segmentation"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  SegFormer model family.

  ## Architectures

    * `:base` - plain SegFormer without any head on top

    * `:for_image_classification` - SegFormer with a classification head.
      The head consists of a single dense layer on top of the average-pooled
      features

    * `:for_semantic_segmentation` - SegFormer with a semantic segmentation
      head on top

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_image_classification, :for_semantic_segmentation]

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

  def model(%__MODULE__{architecture: :for_semantic_segmentation} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      segmentation_head(outputs.stage_hidden_states, spec, name: "segmentation_head")

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
      attentions: encoder_outputs.attentions,
      stage_hidden_states: encoder_outputs.stage_hidden_states
    }
  end

  defp encoder(pixel_values, spec, opts) do
    name = opts[:name]

    num_stages = length(spec.hidden_sizes)

    state = %{
      hidden_state: pixel_values,
      hidden_states: Layers.none(),
      attentions: Layers.none(),
      stage_hidden_states: []
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

          # Overlapping patch embedding for this stage
          embedded =
            overlap_patch_embedding(state.hidden_state, hidden_size,
              patch_size: patch_size,
              stride: stride,
              kernel_initializer: kernel_initializer(spec),
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
                activation: spec.activation
              ],
              block_type: :norm_first,
              name: join(name, "block.#{stage_idx}")
            )

          # Reshape back to spatial for next stage and for output
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
            attentions: block_outputs.attentions,
            stage_hidden_states: state.stage_hidden_states ++ [stage_hidden_state]
          }
      end

    %{
      hidden_state: result.hidden_state,
      hidden_states: result.hidden_states,
      attentions: result.attentions,
      stage_hidden_states: result.stage_hidden_states
    }
  end

  defp overlap_patch_embedding(hidden_state, hidden_size, opts) do
    name = opts[:name]
    patch_size = opts[:patch_size]
    stride = opts[:stride]
    kernel_initializer = opts[:kernel_initializer]

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
    |> Axon.layer_norm(epsilon: 1.0e-6, name: join(name, "norm"))
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

  defp segmentation_head(stage_hidden_states, spec, opts) do
    name = opts[:name]

    # Project each stage output to decoder_hidden_size
    projected =
      stage_hidden_states
      |> Enum.with_index()
      |> Enum.map(fn {stage_output, idx} ->
        stage_output
        |> Axon.conv(spec.decoder_hidden_size,
          kernel_size: 1,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "linear_c.#{idx}")
        )
        |> Axon.layer(
          fn x, _opts ->
            {batch, _h, _w, c} = Nx.shape(x)
            target_h = div(spec.image_size, 4)
            target_w = div(spec.image_size, 4)

            Nx.new_axis(x, 3)
            |> Nx.tile([1, 1, 1, 1, 1])
            |> then(fn x ->
              Axon.Layers.resize(x,
                size: {target_h, target_w},
                method: :bilinear,
                channels: :last
              )
            end)
            |> Nx.reshape({batch, target_h, target_w, c})
          end,
          [],
          name: join(name, "resize.#{idx}")
        )
      end)

    # Concatenate all projected stage outputs
    Enum.reduce(projected, fn x, acc -> Axon.concatenate([acc, x], axis: 3) end)
    |> Axon.conv(spec.decoder_hidden_size,
      kernel_size: 1,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "linear_fuse")
    )
    |> Axon.batch_norm(name: join(name, "batch_norm"))
    |> Axon.activation(spec.activation)
    |> Axon.dropout(rate: spec.dropout_rate)
    |> Axon.conv(spec.num_labels,
      kernel_size: 1,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "classifier")
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
          hidden_sizes: {"hidden_sizes", list(number())},
          num_blocks: {"depths", list(number())},
          num_attention_heads: {"num_attention_heads", list(number())},
          patch_sizes: {"patch_sizes", list(number())},
          strides: {"strides", list(number())},
          sr_ratios: {"sr_ratios", list(number())},
          intermediate_ratios: {"mlp_ratios", list(number())},
          image_size: {"image_size", number()},
          num_channels: {"num_channels", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()},
          decoder_hidden_size: {"decoder_hidden_size", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "encoder.patch_embeddings.{s}.projection" =>
          "segformer.encoder.patch_embeddings.{s}.proj",
        "encoder.patch_embeddings.{s}.norm" =>
          "segformer.encoder.patch_embeddings.{s}.layer_norm",
        "encoder.block.{s}.blocks.{n}.self_attention_norm" =>
          "segformer.encoder.block.{s}.{n}.layer_norm_1",
        "encoder.block.{s}.blocks.{n}.self_attention.query" =>
          "segformer.encoder.block.{s}.{n}.attention.self.query",
        "encoder.block.{s}.blocks.{n}.self_attention.key" =>
          "segformer.encoder.block.{s}.{n}.attention.self.key",
        "encoder.block.{s}.blocks.{n}.self_attention.value" =>
          "segformer.encoder.block.{s}.{n}.attention.self.value",
        "encoder.block.{s}.blocks.{n}.self_attention.output" =>
          "segformer.encoder.block.{s}.{n}.attention.output.dense",
        "encoder.block.{s}.blocks.{n}.output_norm" =>
          "segformer.encoder.block.{s}.{n}.layer_norm_2",
        "encoder.block.{s}.blocks.{n}.ffn.intermediate" =>
          "segformer.encoder.block.{s}.{n}.mlp.dense1",
        "encoder.block.{s}.blocks.{n}.ffn.output" =>
          "segformer.encoder.block.{s}.{n}.mlp.dense2",
        "encoder.layer_norms.{s}" => "segformer.encoder.layer_norm.{s}",
        "segmentation_head.linear_c.{s}" =>
          "decode_head.linear_c.{s}",
        "segmentation_head.linear_fuse" => "decode_head.linear_fuse",
        "segmentation_head.batch_norm" => "decode_head.batch_norm",
        "segmentation_head.classifier" => "decode_head.classifier",
        "image_classification_head.output" => "classifier"
      }
    end
  end
end

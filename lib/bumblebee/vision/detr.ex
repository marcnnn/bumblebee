defmodule Bumblebee.Vision.Detr do
  alias Bumblebee.Shared

  options =
    [
      hidden_size: [
        default: 256,
        doc: "the dimensionality of hidden layers"
      ],
      encoder_num_blocks: [
        default: 6,
        doc: "the number of Transformer blocks in the encoder"
      ],
      decoder_num_blocks: [
        default: 6,
        doc: "the number of Transformer blocks in the decoder"
      ],
      encoder_num_attention_heads: [
        default: 8,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      decoder_num_attention_heads: [
        default: 8,
        doc: "the number of attention heads for each attention layer in the decoder"
      ],
      encoder_intermediate_size: [
        default: 2048,
        docs:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      decoder_intermediate_size: [
        default: 2048,
        docs:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the decoder"
      ],
      num_queries: [
        default: 100,
        doc: "the number of object queries, i.e. detection slots"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      image_size: [
        default: 800,
        doc: "the size of the input spatial dimensions"
      ],
      activation: [
        default: :relu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for encoder and decoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  DETR model family.

  ## Architectures

    * `:base` - plain DETR without any head on top

    * `:for_object_detection` - DETR with object detection heads on top
      for predicting bounding boxes and class labels

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"attention_head_mask"` - `{encoder_num_blocks, encoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"decoder_attention_head_mask"` - `{decoder_num_blocks, decoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the decoder.

    * `"cross_attention_head_mask"` - `{decoder_num_blocks, decoder_num_attention_heads}`

      Mask to nullify selected heads of the cross-attention blocks in
      the decoder with shape.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_object_detection]

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

  def model(%__MODULE__{architecture: :for_object_detection} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    # Class prediction head
    logits =
      outputs.hidden_state
      |> Axon.dense(spec.num_labels + 1,
        kernel_initializer: kernel_initializer(spec),
        name: "object_detection_head.class_output"
      )

    # Bounding box prediction head (MLP)
    pred_boxes =
      outputs.hidden_state
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: "object_detection_head.bbox_dense1"
      )
      |> Axon.activation(:relu)
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: "object_detection_head.bbox_dense2"
      )
      |> Axon.activation(:relu)
      |> Axon.dense(4,
        kernel_initializer: kernel_initializer(spec),
        name: "object_detection_head.bbox_output"
      )
      |> Axon.sigmoid()

    Layers.output(%{
      logits: logits,
      pred_boxes: pred_boxes,
      decoder_hidden_states: outputs.decoder_hidden_states,
      decoder_attentions: outputs.decoder_attentions,
      cross_attentions: outputs.cross_attentions,
      encoder_hidden_state: outputs.encoder_hidden_state,
      encoder_hidden_states: outputs.encoder_hidden_states,
      encoder_attentions: outputs.encoder_attentions
    })
  end

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    encoder_attention_head_mask_shape =
      {spec.encoder_num_blocks, spec.encoder_num_attention_heads}

    decoder_attention_head_mask_shape =
      {spec.decoder_num_blocks, spec.decoder_num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape),
      Axon.input("attention_mask", optional: true, shape: {nil, nil}),
      Axon.input("attention_head_mask", optional: true, shape: encoder_attention_head_mask_shape),
      Axon.input("decoder_attention_head_mask",
        optional: true,
        shape: decoder_attention_head_mask_shape
      ),
      Axon.input("cross_attention_head_mask",
        optional: true,
        shape: decoder_attention_head_mask_shape
      )
    ])
  end

  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    # Use a simple CNN to extract features and project to hidden_size
    backbone_features =
      inputs["pixel_values"]
      |> backbone(spec, name: join(name, "backbone"))

    # Flatten spatial dimensions and project to hidden_size
    encoder_input =
      backbone_features
      |> Axon.reshape({:batch, :auto, spec.hidden_size}, name: join(name, "flatten"))

    # Encoder
    encoder_outputs =
      encoder(
        encoder_input,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        spec,
        name: join(name, "encoder")
      )

    # Object queries (learned embeddings)
    query_embeddings =
      Layers.learned_embeddings(spec.num_queries, spec.hidden_size,
        name: join(name, "query_position_embeddings")
      )

    # Expand query embeddings to batch size
    query_input =
      Axon.layer(
        fn query_embeds, encoder_hidden, _opts ->
          batch_size = Nx.axis_size(encoder_hidden, 0)
          Nx.broadcast(query_embeds, {batch_size, spec.num_queries, spec.hidden_size})
        end,
        [query_embeddings, encoder_outputs.hidden_state],
        name: join(name, "query_broadcast")
      )

    # Decoder with cross-attention to encoder outputs
    decoder_outputs =
      decoder(
        query_input,
        nil,
        inputs["decoder_attention_head_mask"],
        encoder_outputs.hidden_state,
        inputs["attention_mask"],
        inputs["cross_attention_head_mask"],
        spec,
        name: join(name, "decoder")
      )

    %{
      hidden_state: decoder_outputs.hidden_state,
      decoder_hidden_states: decoder_outputs.hidden_states,
      decoder_attentions: decoder_outputs.attentions,
      cross_attentions: decoder_outputs.cross_attentions,
      encoder_hidden_state: encoder_outputs.hidden_state,
      encoder_hidden_states: encoder_outputs.hidden_states,
      encoder_attentions: encoder_outputs.attentions
    }
  end

  defp backbone(pixel_values, spec, opts) do
    name = opts[:name]

    # Simplified CNN backbone that extracts features and projects to hidden_size
    pixel_values
    |> Axon.conv(64,
      kernel_size: 7,
      strides: 2,
      padding: [{3, 3}, {3, 3}],
      use_bias: false,
      kernel_initializer: conv_kernel_initializer(),
      name: join(name, "conv1")
    )
    |> Axon.batch_norm(name: join(name, "bn1"))
    |> Axon.activation(:relu)
    |> Axon.max_pool(kernel_size: 3, strides: 2, padding: [{1, 1}, {1, 1}])
    |> Axon.conv(128,
      kernel_size: 3,
      strides: 2,
      padding: [{1, 1}, {1, 1}],
      use_bias: false,
      kernel_initializer: conv_kernel_initializer(),
      name: join(name, "conv2")
    )
    |> Axon.batch_norm(name: join(name, "bn2"))
    |> Axon.activation(:relu)
    |> Axon.conv(spec.hidden_size,
      kernel_size: 1,
      kernel_initializer: conv_kernel_initializer(),
      name: join(name, "input_projection")
    )
  end

  defp encoder(hidden_state, attention_mask, attention_head_mask, spec, opts) do
    name = opts[:name]

    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      num_blocks: spec.encoder_num_blocks,
      num_attention_heads: spec.encoder_num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      dropout_rate: spec.dropout_rate,
      attention_dropout_rate: spec.attention_dropout_rate,
      layer_norm: [
        epsilon: 1.0e-5
      ],
      ffn: [
        intermediate_size: spec.encoder_intermediate_size,
        activation: spec.activation
      ],
      name: join(name, "blocks")
    )
  end

  defp decoder(
         hidden_state,
         attention_mask,
         attention_head_mask,
         encoder_hidden_state,
         encoder_attention_mask,
         cross_attention_head_mask,
         spec,
         opts
       ) do
    name = opts[:name]

    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      cross_hidden_state: encoder_hidden_state,
      cross_attention_mask: encoder_attention_mask,
      cross_attention_head_mask: cross_attention_head_mask,
      num_blocks: spec.decoder_num_blocks,
      num_attention_heads: spec.decoder_num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      dropout_rate: spec.dropout_rate,
      attention_dropout_rate: spec.attention_dropout_rate,
      layer_norm: [
        epsilon: 1.0e-5
      ],
      ffn: [
        intermediate_size: spec.decoder_intermediate_size,
        activation: spec.activation
      ],
      name: join(name, "blocks")
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
          hidden_size: {"d_model", number()},
          encoder_num_blocks: {"encoder_layers", number()},
          decoder_num_blocks: {"decoder_layers", number()},
          encoder_num_attention_heads: {"encoder_attention_heads", number()},
          decoder_num_attention_heads: {"decoder_attention_heads", number()},
          encoder_intermediate_size: {"encoder_ffn_dim", number()},
          decoder_intermediate_size: {"decoder_ffn_dim", number()},
          num_queries: {"num_queries", number()},
          num_channels: {"num_channels", number()},
          activation: {"activation_function", activation()},
          dropout_rate: {"dropout", number()},
          attention_dropout_rate: {"attention_dropout", number()},
          initializer_scale: {"init_std", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "backbone.conv1" => "model.backbone.conv_encoder.model.embedder.embedder.convolution",
        "backbone.bn1" => "model.backbone.conv_encoder.model.embedder.embedder.normalization",
        "backbone.conv2" => "model.backbone.conv_encoder.model.encoder.stages.0.layers.0.layer.0.convolution",
        "backbone.bn2" => "model.backbone.conv_encoder.model.encoder.stages.0.layers.0.layer.0.normalization",
        "backbone.input_projection" => "model.input_projection",
        "query_position_embeddings" => %{
          "embeddings" => {
            [{"model", "query_position_embeddings.weight"}],
            fn [value] -> value end
          }
        },
        "encoder.blocks.{n}.self_attention.query" =>
          "model.encoder.layers.{n}.self_attn.q_proj",
        "encoder.blocks.{n}.self_attention.key" =>
          "model.encoder.layers.{n}.self_attn.k_proj",
        "encoder.blocks.{n}.self_attention.value" =>
          "model.encoder.layers.{n}.self_attn.v_proj",
        "encoder.blocks.{n}.self_attention.output" =>
          "model.encoder.layers.{n}.self_attn.out_proj",
        "encoder.blocks.{n}.self_attention_norm" =>
          "model.encoder.layers.{n}.self_attn_layer_norm",
        "encoder.blocks.{n}.ffn.intermediate" => "model.encoder.layers.{n}.fc1",
        "encoder.blocks.{n}.ffn.output" => "model.encoder.layers.{n}.fc2",
        "encoder.blocks.{n}.output_norm" => "model.encoder.layers.{n}.final_layer_norm",
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
        "decoder.blocks.{n}.cross_attention.query" =>
          "model.decoder.layers.{n}.encoder_attn.q_proj",
        "decoder.blocks.{n}.cross_attention.key" =>
          "model.decoder.layers.{n}.encoder_attn.k_proj",
        "decoder.blocks.{n}.cross_attention.value" =>
          "model.decoder.layers.{n}.encoder_attn.v_proj",
        "decoder.blocks.{n}.cross_attention.output" =>
          "model.decoder.layers.{n}.encoder_attn.out_proj",
        "decoder.blocks.{n}.cross_attention_norm" =>
          "model.decoder.layers.{n}.encoder_attn_layer_norm",
        "decoder.blocks.{n}.ffn.intermediate" => "model.decoder.layers.{n}.fc1",
        "decoder.blocks.{n}.ffn.output" => "model.decoder.layers.{n}.fc2",
        "decoder.blocks.{n}.output_norm" => "model.decoder.layers.{n}.final_layer_norm",
        "object_detection_head.class_output" => "class_labels_classifier",
        "object_detection_head.bbox_dense1" => "bbox_predictor.layers.0",
        "object_detection_head.bbox_dense2" => "bbox_predictor.layers.1",
        "object_detection_head.bbox_output" => "bbox_predictor.layers.2"
      }
    end
  end
end

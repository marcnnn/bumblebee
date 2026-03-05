defmodule Bumblebee.Vision.Mask2Former do
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
      hidden_size: [
        default: 256,
        doc: "the dimensionality of hidden layers"
      ],
      num_queries: [
        default: 100,
        doc: "the number of object queries"
      ],
      encoder_hidden_size: [
        default: 256,
        doc: "the hidden size for the pixel decoder encoder"
      ],
      encoder_num_blocks: [
        default: 6,
        doc: "the number of Transformer blocks in the pixel decoder encoder"
      ],
      encoder_num_attention_heads: [
        default: 8,
        doc: "the number of attention heads in the pixel decoder encoder"
      ],
      encoder_intermediate_size: [
        default: 1024,
        doc:
          "the dimensionality of the intermediate layer in the pixel decoder encoder FFN"
      ],
      decoder_hidden_size: [
        default: 256,
        doc: "the hidden size for the transformer decoder"
      ],
      decoder_num_blocks: [
        default: 6,
        doc: "the number of Transformer blocks in the decoder"
      ],
      decoder_num_attention_heads: [
        default: 8,
        doc: "the number of attention heads in the decoder"
      ],
      decoder_intermediate_size: [
        default: 1024,
        doc:
          "the dimensionality of the intermediate layer in the decoder FFN"
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
  Mask2Former model family.

  ## Architectures

    * `:base` - plain Mask2Former without any head on top

    * `:for_instance_segmentation` - Mask2Former with class and mask
      prediction heads for instance segmentation

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_instance_segmentation]

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

  def model(%__MODULE__{architecture: :for_instance_segmentation} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    # Class prediction head
    logits =
      outputs.hidden_state
      |> Axon.dense(spec.num_labels + 1,
        kernel_initializer: kernel_initializer(spec),
        name: "class_predictor"
      )

    # Mask prediction head (project queries then dot with pixel features)
    mask_embeddings =
      outputs.hidden_state
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: "mask_predictor.projection"
      )

    pred_masks =
      Axon.layer(
        fn mask_embeds, pixel_features, _opts ->
          # mask_embeds: {batch, num_queries, hidden}
          # pixel_features: {batch, h, w, hidden}
          {batch, h, w, _c} = Nx.shape(pixel_features)
          flat_features = Nx.reshape(pixel_features, {batch, h * w, :auto})
          # {batch, num_queries, h*w}
          masks = Nx.dot(mask_embeds, [2], flat_features, [2])
          Nx.reshape(masks, {batch, :auto, h, w})
        end,
        [mask_embeddings, outputs.pixel_decoder_hidden_state],
        name: "mask_predictor.predict"
      )

    Layers.output(%{
      logits: logits,
      pred_masks: pred_masks,
      encoder_hidden_state: outputs.encoder_hidden_state,
      decoder_hidden_states: outputs.decoder_hidden_states,
      decoder_attentions: outputs.decoder_attentions
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

    # Backbone: simple CNN to extract features
    backbone_features =
      inputs["pixel_values"]
      |> backbone(spec, name: join(name, "backbone"))

    # Pixel decoder (encoder transformer on flattened features)
    pixel_decoder_input =
      backbone_features
      |> Axon.reshape({:batch, :auto, spec.encoder_hidden_size},
        name: join(name, "pixel_decoder_flatten")
      )

    pixel_encoder_outputs =
      pixel_decoder_encoder(pixel_decoder_input, spec, name: join(name, "pixel_decoder"))

    # Reshape pixel decoder output back to spatial
    pixel_decoder_hidden_state =
      Axon.layer(
        fn hidden_state, backbone_feat, _opts ->
          {_batch, h, w, _c} = Nx.shape(backbone_feat)
          Nx.reshape(hidden_state, {:auto, h, w, spec.encoder_hidden_size})
        end,
        [pixel_encoder_outputs.hidden_state, backbone_features],
        name: join(name, "pixel_decoder_reshape")
      )

    # Object queries (learned embeddings)
    query_embeddings =
      Layers.learned_embeddings(spec.num_queries, spec.decoder_hidden_size,
        name: join(name, "query_embeddings")
      )

    # Expand query embeddings to batch size
    query_input =
      Axon.layer(
        fn query_embeds, encoder_hidden, _opts ->
          batch_size = Nx.axis_size(encoder_hidden, 0)
          Nx.broadcast(query_embeds, {batch_size, spec.num_queries, spec.decoder_hidden_size})
        end,
        [query_embeddings, pixel_encoder_outputs.hidden_state],
        name: join(name, "query_broadcast")
      )

    # Transformer decoder with cross-attention to pixel decoder features
    decoder_outputs =
      transformer_decoder(
        query_input,
        pixel_encoder_outputs.hidden_state,
        spec,
        name: join(name, "transformer_decoder")
      )

    %{
      hidden_state: decoder_outputs.hidden_state,
      pixel_decoder_hidden_state: pixel_decoder_hidden_state,
      encoder_hidden_state: pixel_encoder_outputs.hidden_state,
      decoder_hidden_states: decoder_outputs.hidden_states,
      decoder_attentions: decoder_outputs.attentions
    }
  end

  defp backbone(pixel_values, spec, opts) do
    name = opts[:name]

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
    |> Axon.conv(spec.encoder_hidden_size,
      kernel_size: 1,
      kernel_initializer: conv_kernel_initializer(),
      name: join(name, "input_projection")
    )
  end

  defp pixel_decoder_encoder(hidden_state, spec, opts) do
    name = opts[:name]

    Layers.Transformer.blocks(hidden_state,
      num_blocks: spec.encoder_num_blocks,
      num_attention_heads: spec.encoder_num_attention_heads,
      hidden_size: spec.encoder_hidden_size,
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

  defp transformer_decoder(hidden_state, encoder_hidden_state, spec, opts) do
    name = opts[:name]

    Layers.Transformer.blocks(hidden_state,
      cross_hidden_state: encoder_hidden_state,
      num_blocks: spec.decoder_num_blocks,
      num_attention_heads: spec.decoder_num_attention_heads,
      hidden_size: spec.decoder_hidden_size,
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
          hidden_size: {"hidden_dimension", number()},
          num_queries: {"num_queries", number()},
          encoder_hidden_size: {"encoder_feedforward_dim", optional(number())},
          encoder_num_blocks: {"encoder_layers", number()},
          encoder_num_attention_heads: {"num_attention_heads", number()},
          decoder_hidden_size: {"hidden_dim", optional(number())},
          decoder_num_blocks: {"decoder_layers", number()},
          decoder_num_attention_heads: {"num_attention_heads", number()},
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
        "backbone.conv1" =>
          "model.pixel_level_module.encoder.model.embedder.embedder.convolution",
        "backbone.bn1" =>
          "model.pixel_level_module.encoder.model.embedder.embedder.normalization",
        "backbone.conv2" =>
          "model.pixel_level_module.encoder.model.encoder.stages.0.layers.0.layer.0.convolution",
        "backbone.bn2" =>
          "model.pixel_level_module.encoder.model.encoder.stages.0.layers.0.layer.0.normalization",
        "backbone.input_projection" =>
          "model.pixel_level_module.input_projection",
        "pixel_decoder.blocks.{n}.self_attention.query" =>
          "model.pixel_level_module.decoder.layers.{n}.self_attn.q_proj",
        "pixel_decoder.blocks.{n}.self_attention.key" =>
          "model.pixel_level_module.decoder.layers.{n}.self_attn.k_proj",
        "pixel_decoder.blocks.{n}.self_attention.value" =>
          "model.pixel_level_module.decoder.layers.{n}.self_attn.v_proj",
        "pixel_decoder.blocks.{n}.self_attention.output" =>
          "model.pixel_level_module.decoder.layers.{n}.self_attn.out_proj",
        "pixel_decoder.blocks.{n}.self_attention_norm" =>
          "model.pixel_level_module.decoder.layers.{n}.self_attn_layer_norm",
        "pixel_decoder.blocks.{n}.ffn.intermediate" =>
          "model.pixel_level_module.decoder.layers.{n}.fc1",
        "pixel_decoder.blocks.{n}.ffn.output" =>
          "model.pixel_level_module.decoder.layers.{n}.fc2",
        "pixel_decoder.blocks.{n}.output_norm" =>
          "model.pixel_level_module.decoder.layers.{n}.final_layer_norm",
        "query_embeddings" => %{
          "embeddings" => {
            [{"model.transformer_module", "query_embeddings.weight"}],
            fn [value] -> value end
          }
        },
        "transformer_decoder.blocks.{n}.self_attention.query" =>
          "model.transformer_module.decoder.layers.{n}.self_attn.q_proj",
        "transformer_decoder.blocks.{n}.self_attention.key" =>
          "model.transformer_module.decoder.layers.{n}.self_attn.k_proj",
        "transformer_decoder.blocks.{n}.self_attention.value" =>
          "model.transformer_module.decoder.layers.{n}.self_attn.v_proj",
        "transformer_decoder.blocks.{n}.self_attention.output" =>
          "model.transformer_module.decoder.layers.{n}.self_attn.out_proj",
        "transformer_decoder.blocks.{n}.self_attention_norm" =>
          "model.transformer_module.decoder.layers.{n}.self_attn_layer_norm",
        "transformer_decoder.blocks.{n}.cross_attention.query" =>
          "model.transformer_module.decoder.layers.{n}.cross_attn.q_proj",
        "transformer_decoder.blocks.{n}.cross_attention.key" =>
          "model.transformer_module.decoder.layers.{n}.cross_attn.k_proj",
        "transformer_decoder.blocks.{n}.cross_attention.value" =>
          "model.transformer_module.decoder.layers.{n}.cross_attn.v_proj",
        "transformer_decoder.blocks.{n}.cross_attention.output" =>
          "model.transformer_module.decoder.layers.{n}.cross_attn.out_proj",
        "transformer_decoder.blocks.{n}.cross_attention_norm" =>
          "model.transformer_module.decoder.layers.{n}.encoder_attn_layer_norm",
        "transformer_decoder.blocks.{n}.ffn.intermediate" =>
          "model.transformer_module.decoder.layers.{n}.fc1",
        "transformer_decoder.blocks.{n}.ffn.output" =>
          "model.transformer_module.decoder.layers.{n}.fc2",
        "transformer_decoder.blocks.{n}.output_norm" =>
          "model.transformer_module.decoder.layers.{n}.final_layer_norm",
        "class_predictor" => "model.class_predictor",
        "mask_predictor.projection" => "model.mask_predictor"
      }
    end
  end
end

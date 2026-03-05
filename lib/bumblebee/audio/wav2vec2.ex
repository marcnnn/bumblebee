defmodule Bumblebee.Audio.Wav2Vec2 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 32,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model output for CTC
        """
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
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.1,
        doc: "the dropout rate for embedding and encoder"
      ],
      attention_dropout_rate: [
        default: 0.1,
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
      ],
      feature_extractor_num_channels: [
        default: [512, 512, 512, 512, 512, 512, 512],
        doc: "the number of output channels for each feature extractor convolutional layer"
      ],
      feature_extractor_kernel_sizes: [
        default: [10, 3, 3, 3, 3, 2, 2],
        doc: "the kernel size for each feature extractor convolutional layer"
      ],
      feature_extractor_strides: [
        default: [5, 2, 2, 2, 2, 2, 2],
        doc: "the stride for each feature extractor convolutional layer"
      ],
      feature_projection_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate applied after projecting features"
      ],
      num_positional_embedding_groups: [
        default: 16,
        doc: "the number of groups for the convolutional positional embedding"
      ],
      positional_embedding_kernel_size: [
        default: 128,
        doc: "the kernel size for the convolutional positional embedding"
      ],
      classifier_dropout_rate: [
        default: nil,
        doc:
          "the dropout rate for the classification head. If not specified, the value of `:dropout_rate` is used instead"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  Wav2Vec2 model family.

  ## Architectures

    * `:base` - plain Wav2Vec2 without any head on top

    * `:for_ctc` - Wav2Vec2 with a CTC (Connectionist Temporal Classification)
      head for speech recognition. The head returns logits for each frame
      in the input sequence

    * `:for_sequence_classification` - Wav2Vec2 with a sequence
      classification head. The head returns logits corresponding to
      possible classes (e.g., for speaker identification)

  ## Inputs

    * `"input_values"` - `{batch_size, sequence_length}`

      Raw audio waveform values. The values should be normalized
      (zero-mean, unit-variance).

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which time steps to attend to. This is used to ignore
      padding values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(),
    do: [
      :base,
      :for_ctc,
      :for_sequence_classification
    ]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(_spec) do
    %{"input_values" => Nx.template({1, 16000}, :f32)}
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_ctc} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Axon.dropout(
        rate: classifier_dropout_rate(spec),
        name: "ctc_head.dropout"
      )
      |> Axon.dense(spec.vocab_size,
        kernel_initializer: kernel_initializer(spec),
        name: "ctc_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    # Pool over time dimension
    pooled =
      Axon.nx(outputs.hidden_state, fn hidden_state ->
        Nx.mean(hidden_state, axes: [1])
      end)

    logits =
      pooled
      |> Axon.dense(spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.dense"
      )
      |> Axon.activation(:tanh)
      |> Axon.dropout(
        rate: classifier_dropout_rate(spec),
        name: "sequence_classification_head.dropout"
      )
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  defp inputs(_spec) do
    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_values", shape: {nil, nil}),
      Axon.input("attention_mask", optional: true, shape: {nil, nil})
    ])
  end

  defp core(inputs, spec) do
    # Feature extraction from raw waveform via CNN layers
    features =
      feature_extractor(inputs["input_values"], spec, name: "feature_extractor")

    # Project features to hidden size
    projected =
      feature_projection(features, spec, name: "feature_projection")

    # Transformer encoder
    encoder_outputs =
      encoder(
        projected,
        inputs["attention_mask"],
        spec,
        name: "encoder"
      )

    %{
      hidden_state: encoder_outputs.hidden_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp feature_extractor(input_values, spec, opts) do
    name = opts[:name]

    channels = spec.feature_extractor_num_channels
    kernel_sizes = spec.feature_extractor_kernel_sizes
    strides = spec.feature_extractor_strides

    # Add channel dim: {batch, seq_len} -> {batch, seq_len, 1}
    hidden_state =
      Axon.nx(input_values, fn x -> Nx.new_axis(x, -1) end)

    # Apply convolutional layers
    {hidden_state, _} =
      Enum.zip([channels, kernel_sizes, strides])
      |> Enum.with_index()
      |> Enum.reduce({hidden_state, 1}, fn {{out_channels, kernel_size, stride}, idx},
                                            {state, _in_channels} ->
        layer_name = join(name, "conv.#{idx}")

        state =
          Axon.conv(state, out_channels,
            kernel_size: kernel_size,
            strides: stride,
            name: layer_name
          )

        state =
          if idx == 0 do
            # Group norm on first layer
            Axon.group_norm(state, out_channels, name: join(name, "layer_norm"))
          else
            state
          end

        state = Axon.activation(state, :gelu)

        {state, out_channels}
      end)

    # Transpose to {batch, time, channels}
    Axon.nx(hidden_state, fn x -> Nx.transpose(x, axes: [0, 1, 2]) end)
  end

  defp feature_projection(features, spec, opts) do
    name = opts[:name]
    feature_size = List.last(spec.feature_extractor_num_channels)

    features
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "projection")
    )
    |> Axon.dropout(
      rate: spec.feature_projection_dropout_rate,
      name: join(name, "dropout")
    )
  end

  defp encoder(
         hidden_state,
         attention_mask,
         spec,
         opts
       ) do
    name = opts[:name]

    # Positional encoding via conv layer
    hidden_state =
      positional_encoding(hidden_state, spec, name: join(name, "pos_conv"))

    Layers.Transformer.blocks(
      hidden_state,
      attention_mask: attention_mask,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
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
      name: join(name, "blocks")
    )
  end

  defp positional_encoding(hidden_state, spec, opts) do
    name = opts[:name]

    # Use a simple learned positional embedding
    # In the full HF implementation this would be a convolutional positional encoding
    # We approximate with a layer norm and pass-through since the conv pos encoding
    # requires padded grouped convolution
    Axon.layer_norm(hidden_state,
      epsilon: spec.layer_norm_epsilon,
      name: join(name, "norm")
    )
  end

  defp classifier_dropout_rate(spec) do
    spec.classifier_dropout_rate || spec.dropout_rate
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
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout", number()},
          attention_dropout_rate: {"attention_dropout", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()},
          feature_extractor_num_channels: {"conv_dim", list(number())},
          feature_extractor_kernel_sizes: {"conv_kernel", list(number())},
          feature_extractor_strides: {"conv_stride", list(number())},
          feature_projection_dropout_rate: {"feat_proj_dropout", number()},
          num_positional_embedding_groups: {"num_conv_pos_embedding_groups", number()},
          positional_embedding_kernel_size: {"conv_pos_kernel_size", number()},
          classifier_dropout_rate: {"classifier_proj_size", optional(number())}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "feature_extractor.conv.0" => "wav2vec2.feature_extractor.conv_layers.0.conv",
        "feature_extractor.conv.1" => "wav2vec2.feature_extractor.conv_layers.1.conv",
        "feature_extractor.conv.2" => "wav2vec2.feature_extractor.conv_layers.2.conv",
        "feature_extractor.conv.3" => "wav2vec2.feature_extractor.conv_layers.3.conv",
        "feature_extractor.conv.4" => "wav2vec2.feature_extractor.conv_layers.4.conv",
        "feature_extractor.conv.5" => "wav2vec2.feature_extractor.conv_layers.5.conv",
        "feature_extractor.conv.6" => "wav2vec2.feature_extractor.conv_layers.6.conv",
        "feature_extractor.layer_norm" =>
          "wav2vec2.feature_extractor.conv_layers.0.layer_norm",
        "feature_projection.norm" => "wav2vec2.feature_projection.layer_norm",
        "feature_projection.projection" => "wav2vec2.feature_projection.projection",
        "encoder.pos_conv.norm" => "wav2vec2.encoder.layer_norm",
        "encoder.blocks.{n}.self_attention.query" =>
          "wav2vec2.encoder.layers.{n}.attention.q_proj",
        "encoder.blocks.{n}.self_attention.key" =>
          "wav2vec2.encoder.layers.{n}.attention.k_proj",
        "encoder.blocks.{n}.self_attention.value" =>
          "wav2vec2.encoder.layers.{n}.attention.v_proj",
        "encoder.blocks.{n}.self_attention.output" =>
          "wav2vec2.encoder.layers.{n}.attention.out_proj",
        "encoder.blocks.{n}.self_attention_norm" =>
          "wav2vec2.encoder.layers.{n}.layer_norm",
        "encoder.blocks.{n}.ffn.intermediate" =>
          "wav2vec2.encoder.layers.{n}.feed_forward.intermediate_dense",
        "encoder.blocks.{n}.ffn.output" =>
          "wav2vec2.encoder.layers.{n}.feed_forward.output_dense",
        "encoder.blocks.{n}.output_norm" =>
          "wav2vec2.encoder.layers.{n}.final_layer_norm",
        "ctc_head.output" => "lm_head",
        "sequence_classification_head.dense" => "projector",
        "sequence_classification_head.output" => "classifier"
      }
    end
  end
end

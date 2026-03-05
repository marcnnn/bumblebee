defmodule Bumblebee.Vision.Davit do
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
      patch_size: [
        default: 4,
        doc: "the size of the patch spatial dimensions"
      ],
      hidden_sizes: [
        default: [96, 192, 384, 768],
        doc: "the dimensionality of hidden layers at each stage"
      ],
      depths: [
        default: [1, 1, 3, 1],
        doc: "the number of Transformer blocks at each stage"
      ],
      num_attention_heads: [
        default: [3, 6, 12, 24],
        doc: "the number of attention heads for each attention layer at each stage"
      ],
      intermediate_size_ratio: [
        default: 4,
        doc: """
        the dimensionality of the intermediate layer in the transformer feed-forward network (FFN),
        expressed as a multiplier of hidden size at the given stage
        """
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for encoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      layer_norm_epsilon: [
        default: 1.0e-5,
        doc: "the epsilon used by the layer normalization layers"
      ]
    ] ++ Shared.common_options([:num_labels, :id_to_label])

  @moduledoc """
  DaViT (Dual Attention Vision Transformer) model.

  ## Architectures

    * `:base` - plain DaViT without any head on top

    * `:for_image_classification` - DaViT model with a classification head

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [DaViT: Dual Attention Vision Transformers](https://arxiv.org/abs/2204.03645)

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
      Axon.dense(outputs.pooled_state, spec.num_labels,
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

    embeddings =
      embedder(inputs["pixel_values"], spec, name: join(name, "embedder"))

    encoder_outputs =
      encoder(embeddings, spec, name: join(name, "encoder"))

    hidden_state =
      Axon.layer_norm(encoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "norm")
      )

    pooled_state =
      hidden_state
      |> Axon.adaptive_avg_pool(output_size: {1}, name: join(name, "pooler"))
      |> Axon.flatten()

    %{
      hidden_state: hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp embedder(pixel_values, spec, opts) do
    name = opts[:name]
    first_hidden_size = hd(spec.hidden_sizes)

    pixel_values
    |> Axon.conv(first_hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "patch_embedding.projection")
    )
    |> Axon.reshape({:batch, :auto, first_hidden_size}, name: join(name, "patch_embedding.reshape"))
    |> Axon.layer_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]
    num_stages = length(spec.depths)

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({})
    }

    for stage_idx <- 0..(num_stages - 1), reduce: state do
      state ->
        stage_name = name |> join("stages") |> join(stage_idx)

        hidden_size = Enum.at(spec.hidden_sizes, stage_idx)
        num_blocks = Enum.at(spec.depths, stage_idx)
        num_heads = Enum.at(spec.num_attention_heads, stage_idx)

        grid_size = div(spec.image_size, spec.patch_size)
        input_resolution = div(grid_size, 2 ** stage_idx)

        encoder_outputs =
          Layers.Transformer.blocks(state.hidden_state,
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
              intermediate_size: floor(spec.intermediate_size_ratio * hidden_size),
              activation: spec.activation
            ],
            block_type: :norm_first,
            name: join(stage_name, "blocks")
          )

        hidden_state_before_downsample = encoder_outputs.hidden_state

        hidden_state =
          if stage_idx < num_stages - 1 do
            patch_merging(encoder_outputs.hidden_state,
              input_resolution: input_resolution,
              hidden_size: hidden_size,
              next_hidden_size: Enum.at(spec.hidden_sizes, stage_idx + 1),
              layer_norm_epsilon: spec.layer_norm_epsilon,
              kernel_initializer: kernel_initializer(spec),
              name: join(stage_name, "downsample")
            )
          else
            encoder_outputs.hidden_state
          end

        %{
          hidden_state: hidden_state,
          hidden_states: Layers.append(state.hidden_states, hidden_state_before_downsample),
          attentions: encoder_outputs.attentions
        }
    end
  end

  defp patch_merging(hidden_state, opts) do
    input_resolution = opts[:input_resolution]
    hidden_size = opts[:hidden_size]
    next_hidden_size = opts[:next_hidden_size]
    layer_norm_epsilon = opts[:layer_norm_epsilon]
    kernel_initializer = opts[:kernel_initializer]
    name = opts[:name]

    hidden_state
    |> Axon.nx(fn hidden_state ->
      {batch_size, _sequence_length, _hidden_size} = Nx.shape(hidden_state)

      hidden_state =
        Nx.reshape(hidden_state, {batch_size, input_resolution, input_resolution, :auto})

      input_feature_0 = hidden_state[[.., 0..-1//2, 0..-1//2, ..]]
      input_feature_1 = hidden_state[[.., 1..-1//2, 0..-1//2, ..]]
      input_feature_2 = hidden_state[[.., 0..-1//2, 1..-1//2, ..]]
      input_feature_3 = hidden_state[[.., 1..-1//2, 1..-1//2, ..]]

      Nx.concatenate([input_feature_0, input_feature_1, input_feature_2, input_feature_3],
        axis: -1
      )
      |> Nx.reshape({batch_size, :auto, 4 * hidden_size})
    end)
    |> Axon.layer_norm(epsilon: layer_norm_epsilon, name: join(name, "norm"))
    |> Axon.dense(next_hidden_size,
      kernel_initializer: kernel_initializer,
      name: join(name, "reduction"),
      use_bias: false
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
          patch_size: {"patch_size", number()},
          hidden_sizes: {"hidden_sizes", list(number())},
          depths: {"depths", list(number())},
          num_attention_heads: {"num_heads", list(number())},
          intermediate_size_ratio: {"mlp_ratio", number()},
          activation: {"hidden_act", activation()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.patch_embedding.projection" => "davit.embeddings.patch_embeddings.projection",
        "embedder.norm" => "davit.embeddings.norm",
        "encoder.stages.{n}.blocks.{m}.self_attention_norm" =>
          "davit.encoder.stages.{n}.blocks.{m}.layernorm_before",
        "encoder.stages.{n}.blocks.{m}.self_attention.query" =>
          "davit.encoder.stages.{n}.blocks.{m}.attention.self.query",
        "encoder.stages.{n}.blocks.{m}.self_attention.key" =>
          "davit.encoder.stages.{n}.blocks.{m}.attention.self.key",
        "encoder.stages.{n}.blocks.{m}.self_attention.value" =>
          "davit.encoder.stages.{n}.blocks.{m}.attention.self.value",
        "encoder.stages.{n}.blocks.{m}.self_attention.output" =>
          "davit.encoder.stages.{n}.blocks.{m}.attention.output.dense",
        "encoder.stages.{n}.blocks.{m}.ffn.intermediate" =>
          "davit.encoder.stages.{n}.blocks.{m}.intermediate.dense",
        "encoder.stages.{n}.blocks.{m}.ffn.output" =>
          "davit.encoder.stages.{n}.blocks.{m}.output.dense",
        "encoder.stages.{n}.blocks.{m}.output_norm" =>
          "davit.encoder.stages.{n}.blocks.{m}.layernorm_after",
        "encoder.stages.{n}.downsample.norm" =>
          "davit.encoder.stages.{n}.downsample.norm",
        "encoder.stages.{n}.downsample.reduction" =>
          "davit.encoder.stages.{n}.downsample.reduction",
        "norm" => "davit.layernorm",
        "image_classification_head.output" => "classifier"
      }
    end
  end
end

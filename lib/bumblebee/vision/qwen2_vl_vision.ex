defmodule Bumblebee.Vision.Qwen2VlVision do
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
        default: 14,
        doc: "the size of the patch spatial dimensions"
      ],
      hidden_size: [
        default: 1280,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 32,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      intermediate_size: [
        default: 5120,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      activation: [
        default: :quick_gelu,
        doc: "the activation function"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by the layer normalization layers"
      ]
    ]

  @moduledoc """
  The Qwen2-VL vision encoder model.

  ## Architectures

    * `:base` - the base vision model

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
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
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape)
    ])
  end

  defp core(inputs, spec) do
    embeddings = patch_embedding(inputs["pixel_values"], spec, name: "embedder.patch_embedding")

    encoder_outputs =
      encoder(embeddings, spec, name: "encoder")

    hidden_state =
      Axon.layer_norm(encoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: "norm"
      )

    pooled_state = Layers.take_token(hidden_state, index: 0, axis: 1)

    %{
      hidden_state: hidden_state,
      pooled_state: pooled_state,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]

    pixel_values
    |> Axon.conv(spec.hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      kernel_initializer: Axon.Initializers.normal(),
      name: name
    )
    |> Axon.reshape({:batch, :auto, spec.hidden_size}, name: join(name, "reshape"))
  end

  defp encoder(embeddings, spec, opts) do
    name = opts[:name]

    Layers.Transformer.blocks(embeddings,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: Axon.Initializers.normal(scale: 0.01),
      dropout_rate: 0.0,
      attention_dropout_rate: spec.attention_dropout_rate,
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

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, %{"model_type" => "qwen2_vl", "vision_config" => data}) do
      load(spec, data)
    end

    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          image_size: {"image_size", number()},
          patch_size: {"patch_size", number()},
          hidden_size: {"embed_dim", number()},
          num_blocks: {"depth", number()},
          num_attention_heads: {"num_heads", number()},
          intermediate_size: {"mlp_ratio", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      # Qwen2-VL vision config uses mlp_ratio as a multiplier
      opts =
        if mlp_ratio = opts[:intermediate_size] do
          hidden_size = opts[:hidden_size] || spec.hidden_size
          Keyword.put(opts, :intermediate_size, round(hidden_size * mlp_ratio))
        else
          opts
        end

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.patch_embedding" => "visual.patch_embed.proj",
        "encoder.blocks.{n}.self_attention_norm" => "visual.blocks.{n}.norm1",
        "encoder.blocks.{n}.self_attention.query" =>
          "visual.blocks.{n}.attn.qkv",
        "encoder.blocks.{n}.self_attention.key" =>
          "visual.blocks.{n}.attn.qkv",
        "encoder.blocks.{n}.self_attention.value" =>
          "visual.blocks.{n}.attn.qkv",
        "encoder.blocks.{n}.self_attention.output" =>
          "visual.blocks.{n}.attn.proj",
        "encoder.blocks.{n}.ffn.intermediate" => "visual.blocks.{n}.mlp.fc1",
        "encoder.blocks.{n}.ffn.output" => "visual.blocks.{n}.mlp.fc2",
        "encoder.blocks.{n}.output_norm" => "visual.blocks.{n}.norm2",
        "norm" => "visual.merger.ln_q"
      }
    end
  end
end

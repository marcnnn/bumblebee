defmodule Bumblebee.Vision.Sam do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: 1024,
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
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by the layer normalization layers"
      ],
      dropout_rate: [
        default: 0.0,
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
      ],
      neck_hidden_size: [
        default: 256,
        doc: "the hidden size for the neck projection layers"
      ]
    ]

  @moduledoc """
  SAM (Segment Anything Model) vision encoder.

  ## Architectures

    * `:base` - SAM vision encoder without any head on top

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Segment Anything](https://arxiv.org/abs/2304.02643)

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
    spec
    |> inputs()
    |> core(spec)
    |> Layers.output()
  end

  defp inputs(spec) do
    shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape)
    ])
  end

  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    # Patch embedding
    embeddings =
      patch_embedding(inputs["pixel_values"], spec, name: join(name, "embedder"))

    # Add position embeddings
    num_patches = div(spec.image_size, spec.patch_size)

    position_embeddings =
      Layers.learned_embeddings(num_patches * num_patches, spec.hidden_size,
        initializer: :zeros,
        name: join(name, "position_embedding")
      )

    hidden_state =
      Axon.add(embeddings, position_embeddings)

    # Transformer encoder
    encoder_outputs = encoder(hidden_state, spec, name: join(name, "encoder"))

    # Neck: project to neck_hidden_size
    hidden_state =
      encoder_outputs.hidden_state
      |> reshape_to_spatial(num_patches, spec.hidden_size, name: join(name, "reshape_to_spatial"))
      |> Axon.conv(spec.neck_hidden_size,
        kernel_size: 1,
        use_bias: false,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "neck.0")
      )
      |> Axon.layer_norm(
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "neck.1")
      )
      |> Axon.conv(spec.neck_hidden_size,
        kernel_size: 3,
        padding: [{1, 1}, {1, 1}],
        use_bias: false,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "neck.2")
      )
      |> Axon.layer_norm(
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "neck.3")
      )

    %{
      hidden_state: hidden_state,
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

  defp reshape_to_spatial(hidden_state, num_patches, hidden_size, opts) do
    name = opts[:name]

    Axon.layer(
      fn hidden_state, _opts ->
        Nx.reshape(hidden_state, {:auto, num_patches, num_patches, hidden_size})
      end,
      [hidden_state],
      name: name
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      # SAM stores vision config under "vision_config" key
      vision_data = data["vision_config"] || data

      opts =
        convert!(vision_data,
          image_size: {"image_size", number()},
          num_channels: {"num_channels", number()},
          patch_size: {"patch_size", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(vision_data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.projection" => "vision_encoder.patch_embed.projection",
        "position_embedding" => %{
          "embeddings" => {
            [{"vision_encoder", "pos_embed"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "encoder.blocks.{n}.self_attention_norm" =>
          "vision_encoder.layers.{n}.layer_norm1",
        "encoder.blocks.{n}.self_attention.query" =>
          "vision_encoder.layers.{n}.attn.q_proj",
        "encoder.blocks.{n}.self_attention.key" =>
          "vision_encoder.layers.{n}.attn.k_proj",
        "encoder.blocks.{n}.self_attention.value" =>
          "vision_encoder.layers.{n}.attn.v_proj",
        "encoder.blocks.{n}.self_attention.output" =>
          "vision_encoder.layers.{n}.attn.proj",
        "encoder.blocks.{n}.output_norm" =>
          "vision_encoder.layers.{n}.layer_norm2",
        "encoder.blocks.{n}.ffn.intermediate" =>
          "vision_encoder.layers.{n}.mlp.lin1",
        "encoder.blocks.{n}.ffn.output" =>
          "vision_encoder.layers.{n}.mlp.lin2",
        "neck.0" => "vision_encoder.neck.0",
        "neck.1" => "vision_encoder.neck.1",
        "neck.2" => "vision_encoder.neck.2",
        "neck.3" => "vision_encoder.neck.3"
      }
    end
  end
end

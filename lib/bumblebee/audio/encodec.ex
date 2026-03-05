defmodule Bumblebee.Audio.Encodec do
  alias Bumblebee.Shared

  options =
    [
      hidden_size: [
        default: 128,
        doc: "the dimensionality of the hidden representation"
      ],
      num_filters: [
        default: 32,
        doc: "the base number of filters in the convolutional layers"
      ],
      num_residual_layers: [
        default: 1,
        doc: "the number of residual layers in each residual block"
      ],
      upsampling_ratios: [
        default: [8, 5, 4, 2],
        doc: "the ratios for upsampling in the decoder (and downsampling in the encoder)"
      ],
      codebook_size: [
        default: 1024,
        doc: "the number of entries in each codebook for residual vector quantization"
      ],
      codebook_dim: [
        default: 128,
        doc: "the dimensionality of each codebook entry"
      ],
      activation: [
        default: :elu,
        doc: "the activation function"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ]
    ] ++ Shared.common_options([])

  @moduledoc """
  EnCodec model family.

  ## Architectures

    * `:base` - EnCodec encoder-decoder for neural audio compression.
      The encoder compresses audio into a latent representation, and
      the decoder reconstructs audio from the latent representation.

  ## Inputs

    * `"input_values"` - `{batch_size, channels, sequence_length}`

      Raw audio waveform values.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)

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
  def input_template(_spec) do
    %{"input_values" => Nx.template({1, 1, 16000}, :f32)}
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    Layers.output(%{
      audio_values: outputs.audio_values,
      code_ids: outputs.code_ids
    })
  end

  defp inputs(_spec) do
    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_values", shape: {nil, nil, nil})
    ])
  end

  defp core(inputs, spec) do
    # Encoder: compress audio to latent
    encoded = audio_encoder(inputs["input_values"], spec, name: "encoder")

    # Quantizer: project to codebook dimension
    quantized =
      Axon.dense(encoded, spec.codebook_dim,
        kernel_initializer: kernel_initializer(spec),
        name: "quantizer.projection"
      )

    # Generate code ids from quantized representation (argmax as placeholder)
    code_ids =
      Axon.nx(quantized, fn x ->
        # Simplified: return zeros as placeholder code indices
        batch_size = Nx.axis_size(x, 0)
        seq_len = Nx.axis_size(x, 1)
        Nx.broadcast(Nx.tensor(0, type: :s64), {batch_size, seq_len})
      end)

    # Decoder: reconstruct audio from latent
    decoded = audio_decoder(quantized, spec, name: "decoder")

    %{
      audio_values: decoded,
      code_ids: code_ids
    }
  end

  defp audio_encoder(input_values, spec, opts) do
    name = opts[:name]

    # Initial convolution
    hidden_state =
      Axon.conv(input_values, spec.num_filters,
        kernel_size: 7,
        padding: :same,
        name: join(name, "init_conv")
      )

    # Downsampling blocks
    {hidden_state, _channels} =
      spec.upsampling_ratios
      |> Enum.reverse()
      |> Enum.with_index()
      |> Enum.reduce({hidden_state, spec.num_filters}, fn {ratio, idx}, {state, in_channels} ->
        out_channels = in_channels * 2
        block_name = join(name, "down.#{idx}")

        # Residual layers
        state =
          Enum.reduce(0..(spec.num_residual_layers - 1), state, fn res_idx, s ->
            residual_block(s, in_channels, spec, name: join(block_name, "residual.#{res_idx}"))
          end)

        # Downsampling conv with stride
        state =
          Axon.conv(state, out_channels,
            kernel_size: ratio * 2,
            strides: ratio,
            padding: :same,
            name: join(block_name, "downsample")
          )

        state = Axon.activation(state, spec.activation)

        {state, out_channels}
      end)

    # Final convolution to hidden_size
    Axon.conv(hidden_state, spec.hidden_size,
      kernel_size: 3,
      padding: :same,
      name: join(name, "final_conv")
    )
  end

  defp audio_decoder(latent, spec, opts) do
    name = opts[:name]

    # Calculate the number of channels after all encoder downsampling
    num_ratios = length(spec.upsampling_ratios)
    initial_channels = spec.num_filters * trunc(:math.pow(2, num_ratios))

    # Initial convolution from codebook_dim to initial_channels
    hidden_state =
      Axon.conv(latent, initial_channels,
        kernel_size: 3,
        padding: :same,
        name: join(name, "init_conv")
      )

    # Upsampling blocks (reverse of encoder)
    {hidden_state, _channels} =
      spec.upsampling_ratios
      |> Enum.with_index()
      |> Enum.reduce({hidden_state, initial_channels}, fn {ratio, idx}, {state, in_channels} ->
        out_channels = div(in_channels, 2)
        block_name = join(name, "up.#{idx}")

        # Upsampling via transposed conv
        state =
          Axon.conv_transpose(state, out_channels,
            kernel_size: ratio * 2,
            strides: ratio,
            padding: :same,
            name: join(block_name, "upsample")
          )

        state = Axon.activation(state, spec.activation)

        # Residual layers
        state =
          Enum.reduce(0..(spec.num_residual_layers - 1), state, fn res_idx, s ->
            residual_block(s, out_channels, spec, name: join(block_name, "residual.#{res_idx}"))
          end)

        {state, out_channels}
      end)

    # Final convolution to 1 channel (mono audio)
    Axon.conv(hidden_state, 1,
      kernel_size: 7,
      padding: :same,
      name: join(name, "final_conv")
    )
  end

  defp residual_block(hidden_state, channels, spec, opts) do
    name = opts[:name]

    residual = hidden_state

    hidden_state =
      hidden_state
      |> Axon.activation(spec.activation)
      |> Axon.conv(channels,
        kernel_size: 3,
        padding: :same,
        name: join(name, "conv1")
      )
      |> Axon.activation(spec.activation)
      |> Axon.conv(channels,
        kernel_size: 1,
        name: join(name, "conv2")
      )

    Axon.add([hidden_state, residual])
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          hidden_size: {"hidden_size", number()},
          num_filters: {"num_filters", number()},
          num_residual_layers: {"num_residual_layers", number()},
          upsampling_ratios: {"upsampling_ratios", list(number())},
          codebook_size: {"codebook_size", number()},
          codebook_dim: {"codebook_dim", optional(number())}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "encoder.init_conv" => "encoder.layers.0",
        "encoder.down.{n}.residual.{m}.conv1" => "encoder.layers.{n}.block.{m}.block.1",
        "encoder.down.{n}.residual.{m}.conv2" => "encoder.layers.{n}.block.{m}.block.3",
        "encoder.down.{n}.downsample" => "encoder.layers.{n}.shortcut",
        "encoder.final_conv" => "encoder.layers.final",
        "quantizer.projection" => "quantizer.layers.0",
        "decoder.init_conv" => "decoder.layers.0",
        "decoder.up.{n}.upsample" => "decoder.layers.{n}.upsample",
        "decoder.up.{n}.residual.{m}.conv1" => "decoder.layers.{n}.block.{m}.block.1",
        "decoder.up.{n}.residual.{m}.conv2" => "decoder.layers.{n}.block.{m}.block.3",
        "decoder.final_conv" => "decoder.layers.final"
      }
    end
  end
end

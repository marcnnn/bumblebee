defmodule Bumblebee.Multimodal.Llava do
  alias Bumblebee.Shared

  options =
    [
      vision_spec: [
        default: nil,
        doc:
          "the specification of the vision model. See `Bumblebee.Vision.ClipVision` for details"
      ],
      text_spec: [
        default: nil,
        doc: "the specification of the text model. See `Bumblebee.Text.Llama` for details"
      ],
      projection_size: [
        default: nil,
        doc:
          "the dimensionality of the multi-modal projection. Defaults to the text model hidden size"
      ]
    ]

  @moduledoc """
  The LLaVA model for visual language understanding.

  LLaVA composes a CLIP vision encoder with a LLaMA text decoder
  via a multi-modal projection layer.

  ## Architectures

    * `:for_conditional_generation` - LLaVA model with a language
      modeling head

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"encoder_hidden_state"` - `{batch_size, sequence_length, hidden_size}`

      Last hidden state output from the vision encoder. This hidden state
      is used in cross-attention blocks in the decoder. If specified, the
      model will skip the image encoding process and use this value
      directly.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression). With cache, certain hidden states are
      taken from the cache, rather than recomputed on every decoding
      pass. The cache should be treated as opaque and initialized with
      `Bumblebee.Text.Generation.init_cache/4`.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

  """

  defstruct [architecture: :for_conditional_generation] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:for_conditional_generation]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(%{vision_spec: vision_spec}) do
    vision_shape = {1, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}

    %{
      "pixel_values" => Nx.template(vision_shape, :f32),
      "input_ids" => Nx.template({1, 1}, :u32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :for_conditional_generation} = spec) do
    %{vision_spec: vision_spec, text_spec: text_spec} = spec

    text_hidden_size = text_spec.hidden_size
    vision_hidden_size = vision_spec.hidden_size

    vision_shape = {nil, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}
    text_shape = {nil, nil}
    vision_hidden_shape = {nil, nil, vision_hidden_size}

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("pixel_values", shape: vision_shape),
        Axon.input("input_ids", optional: true, shape: text_shape),
        Axon.input("attention_mask", optional: true, shape: text_shape),
        Axon.input("position_ids", optional: true, shape: text_shape),
        Axon.input("encoder_hidden_state", optional: true, shape: vision_hidden_shape),
        Axon.input("cache", optional: true)
      ])

    vision_model =
      vision_spec
      |> Bumblebee.build_model()
      |> Bumblebee.Utils.Axon.prefix_names("vision_tower.")
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "pixel_values" => inputs["pixel_values"]
      })

    vision_model_outputs =
      Layers.if_present inputs["encoder_hidden_state"] do
        %{
          hidden_state: inputs["encoder_hidden_state"],
          hidden_states: Layers.none(),
          attentions: Layers.none()
        }
      else
        %{
          hidden_state: Axon.nx(vision_model, & &1.hidden_state),
          hidden_states: Axon.nx(vision_model, & &1.hidden_states),
          attentions: Axon.nx(vision_model, & &1.attentions)
        }
      end

    # Multi-modal projector: two dense layers with GELU activation
    projected_vision_state =
      vision_model_outputs.hidden_state
      |> Axon.dense(text_hidden_size,
        name: "multi_modal_projector.linear_1"
      )
      |> Axon.activation(:gelu)
      |> Axon.dense(text_hidden_size,
        name: "multi_modal_projector.linear_2"
      )

    text_model =
      text_spec
      |> Bumblebee.build_model()
      |> Bumblebee.Utils.Axon.prefix_names("language_model.")
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "input_ids" => inputs["input_ids"],
        "attention_mask" => inputs["attention_mask"],
        "position_ids" => inputs["position_ids"],
        "encoder_hidden_state" => projected_vision_state,
        "cache" => inputs["cache"]
      })

    Layers.output(%{
      logits: Axon.nx(text_model, & &1.logits),
      decoder_hidden_states: Axon.nx(text_model, & &1.hidden_states),
      decoder_attentions: Axon.nx(text_model, & &1.attentions),
      encoder_hidden_state: projected_vision_state,
      encoder_hidden_states: vision_model_outputs.hidden_states,
      encoder_attentions: vision_model_outputs.attentions,
      cache: Axon.nx(text_model, & &1.cache)
    })
  end

  @impl true
  def init_cache(
        %{vision_spec: _vision_spec, text_spec: text_spec},
        batch_size,
        max_length,
        inputs
      ) do
    inputs =
      %{
        "input_ids" => inputs["input_ids"],
        "attention_mask" => inputs["attention_mask"],
        "position_ids" => inputs["position_ids"]
      }
      |> Map.reject(&match?({_, nil}, &1))

    text_spec.__struct__.init_cache(text_spec, batch_size, max_length, inputs)
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      {vision_data, data} = Map.pop(data, "vision_config", %{})
      {text_data, data} = Map.pop(data, "text_config", %{})

      vision_spec =
        Bumblebee.Vision.ClipVision
        |> Bumblebee.configure()
        |> Bumblebee.HuggingFace.Transformers.Config.load(vision_data)

      text_spec =
        Bumblebee.Text.Llama
        |> Bumblebee.configure(architecture: :for_causal_language_modeling)
        |> Bumblebee.HuggingFace.Transformers.Config.load(text_data)

      opts =
        convert!(data,
          projection_size: {"projection_dim", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts ++ [text_spec: text_spec, vision_spec: vision_spec])
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.HuggingFace.Transformers

    def params_mapping(spec) do
      text_mapping =
        spec.text_spec
        |> Transformers.Model.params_mapping()
        |> Transformers.Utils.prefix_params_mapping("language_model", nil)

      vision_mapping =
        spec.vision_spec
        |> Transformers.Model.params_mapping()
        |> Transformers.Utils.prefix_params_mapping("vision_tower", nil)

      %{
        "multi_modal_projector.linear_1" => "multi_modal_projector.linear_1",
        "multi_modal_projector.linear_2" => "multi_modal_projector.linear_2"
      }
      |> Map.merge(text_mapping)
      |> Map.merge(vision_mapping)
    end
  end
end

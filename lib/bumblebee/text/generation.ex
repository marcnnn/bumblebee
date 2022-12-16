defmodule Bumblebee.Text.Generation do
  @moduledoc """
  An interface for language models supporting sequence generation.
  """

  @doc """
  Initializes an opaque cache input for iterative inference.
  """
  @callback init_cache(
              spec :: Bumblebee.ModelSpec.t(),
              batch_size :: pos_integer(),
              max_length :: pos_integer(),
              inputs :: map()
            ) :: Nx.Tensor.t() | Nx.Container.t()

  import Nx.Defn

  alias Bumblebee.Shared

  @doc """
  Initializes an opaque cache input for iterative inference.
  """
  @spec init_cache(Bumblebee.ModelSpec.t(), pos_integer(), pos_integer(), map()) :: Nx.t()
  def init_cache(%module{} = spec, batch_size, max_length, inputs) do
    module.init_cache(spec, batch_size, max_length, inputs)
  end

  @doc false
  def generation(model_info, tokenizer, opts \\ []) do
    {compile, opts} = Keyword.pop(opts, :compile)
    {defn_options, opts} = Keyword.pop(opts, :defn_options, [])

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    if compile != nil and (batch_size == nil or sequence_length == nil) do
      raise ArgumentError,
            "expected :compile to be a keyword list specifying :batch_size and :sequence_length, got: #{inspect(compile)}"
    end

    %{params: params} = model_info

    generate_fun = build_generate(model_info.model, model_info.spec, opts)

    Nx.Serving.new(
      fn ->
        generate_fun =
          Shared.compile_or_jit(generate_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :s64),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :s64)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          generate_fun.(params, inputs)
        end
      end,
      batch_size: batch_size
    )
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} = Shared.validate_serving_input!(input, &Shared.validate_string/1)

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, texts,
          length: sequence_length,
          pad_direction: :left,
          return_token_type_ids: false
        )

      {Nx.Batch.concatenate([inputs]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn token_ids, _metadata, multi? ->
      decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)

      decoded
      |> Enum.map(&%{results: [%{text: &1}]})
      |> Shared.normalize_output(multi?)
    end)
  end

  @doc """
  Builds a numerical definition that generates sequences of tokens using
  the given language model.

  The model should be either a decoder or an encoder-decoder. The tokens
  are generated by iterative inference using the decoder (autoregression),
  until the termination criteria are met.

  In case of encoder-decoder models, the corresponding encoder is run
  only once and the intermediate state is reused during all iterations.

  The length of the generated sequence is not fixed, however it can be
  controlled via several options.

  Note that either `:max_new_tokens` or `:max_length` must be specified.

  ## Options

    * `:max_new_tokens` - the maximum number of tokens to be generated,
      ignoring the number of tokens in the prompt

    * `:min_new_tokens` - the minimum number of tokens to be generated,
      ignoring the number of tokens in the prompt

    * `:max_length` - the maximum length of the sequence to be generated.
      Note that this length includes the length of the input prompt
      (including padding). In general, prefer `:max_new_tokens`, which
      ignores the number of tokens in the prompt

    * `:min_length` - the minimum length of the sequence to be generated.
      Note that this length includes the length of the input prompt
      (including padding). In general, prefer `:min_new_tokens`, which
      ignores the number of tokens in the prompt

    * `:decoder_start_token_id` - the id of the initial token when
      generating from scratch, in case of encoder-decoder models

    * `:bos_token_id` - the id of the beginning-of-sequence token

    * `:eos_token_id` - the id of the end-of-sequence token

    * `:pad_token_id` - the id of the padding token

    * `:forced_bos_token_id` - the id of the token to force as the first
      generated token

    * `:forced_eos_token_id` - the id of the token to force as the last
      generated token when `:max_length` is reached

  The default token option values are taken from the given model specification
  when available.
  """
  @spec build_generate(Axon.t(), Bumblebee.ModelSpec.t(), keyword()) ::
          (params :: map(), inputs :: map() -> Nx.t())
  def build_generate(model, spec, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        max_new_tokens: nil,
        min_new_tokens: nil,
        max_length: nil,
        min_length: nil,
        decoder_start_token_id: Map.get(spec, :decoder_start_token_id),
        bos_token_id: Map.get(spec, :bos_token_id),
        eos_token_id: Map.get(spec, :eos_token_id),
        pad_token_id: Map.get(spec, :pad_token_id),
        forced_bos_token_id: Map.get(spec, :forced_bos_token_id),
        forced_eos_token_id: Map.get(spec, :forced_eos_token_id)
      )

    decoder_start_token_id = opts[:decoder_start_token_id] || opts[:bos_token_id]
    eos_token_id = opts[:eos_token_id]
    pad_token_id = opts[:pad_token_id]
    forced_bos_token_id = opts[:forced_bos_token_id]
    forced_eos_token_id = opts[:forced_eos_token_id]

    {max_length_fun, min_length_fun} = lazy_lengths_from_opts(opts)

    {prepare_inputs_fun, update_inputs_fun} =
      input_callbacks(model, spec, max_length_fun, decoder_start_token_id)

    {_init_fun, predict_fun} = Axon.build(model)

    logits_processor_fun =
      get_logits_processor(min_length_fun, eos_token_id, forced_bos_token_id, forced_eos_token_id)

    &generate_impl(
      &2,
      predict_fun,
      &1,
      logits_processor_fun,
      prepare_inputs_fun,
      update_inputs_fun,
      pad_token_id: pad_token_id,
      eos_token_id: eos_token_id
    )
  end

  defp lazy_lengths_from_opts(opts) do
    max_length_fun =
      case {opts[:max_new_tokens], opts[:max_length]} do
        {nil, nil} ->
          raise ArgumentError,
                "expected either :max_new_tokens or :max_length option, but neither was given"

        {max_new_tokens, nil} ->
          fn input_length -> input_length + max_new_tokens end

        {nil, max_length} ->
          fn _ -> max_length end

        _ ->
          raise ArgumentError,
                "only one of :max_new_tokens or :max_length options must be given, but got both"
      end

    min_length_fun =
      case {opts[:min_new_tokens], opts[:min_length]} do
        {nil, nil} ->
          nil

        {min_new_tokens, nil} ->
          fn input_length -> input_length + min_new_tokens end

        {nil, min_length} ->
          fn _ -> min_length end

        _ ->
          raise ArgumentError,
                "only one of :min_new_tokens or :min_length options must be given, but got both"
      end

    {max_length_fun, min_length_fun}
  end

  defp encoder_from_encoder_decoder(model) do
    # We cherry-pick encoder outputs from the encoder-decoder outputs.
    # The expanded expression will have no decoder bits, so it will
    # effectively be the same as an encoder built from scratch

    Axon.nx(model, fn outputs ->
      case outputs do
        %{
          encoder_hidden_state: hidden_state,
          encoder_hidden_states: hidden_states,
          encoder_attentions: attentions
        } ->
          %{
            hidden_state: hidden_state,
            hidden_states: hidden_states,
            attentions: attentions
          }

        _ ->
          raise ArgumentError,
                "expected an encoder-decoder model, but it does not have the expected outputs"
      end
    end)
  end

  defp input_callbacks(model, spec, max_length_fun, decoder_start_token_id) do
    if encoder_decoder?(model) do
      encoder = encoder_from_encoder_decoder(model)
      {_encoder_init_fun, encoder_predict_fun} = Axon.build(encoder)

      prepare_inputs_fun = fn inputs, params ->
        encoder_outputs = encoder_predict_fun.(params, inputs)

        batch_size = Nx.axis_size(inputs["input_ids"], 0)
        decoder_input_ids = Nx.broadcast(decoder_start_token_id, {batch_size, 1})

        inputs =
          Map.merge(inputs, %{
            "encoder_hidden_state" => encoder_outputs.hidden_state,
            "decoder_input_ids" => decoder_input_ids
          })

        max_length = max_length_fun.(1)
        inputs = prepare_decoder_inputs(inputs, "decoder_", spec, max_length)
        {inputs, inputs["decoder_input_ids"], max_length}
      end

      update_inputs_fun = &update_decoder_inputs(&1, &2, &3, "decoder_")

      {prepare_inputs_fun, update_inputs_fun}
    else
      prepare_inputs_fun = fn inputs, _params ->
        sequence_length = Nx.axis_size(inputs["input_ids"], 1)
        max_length = max_length_fun.(sequence_length)
        inputs = prepare_decoder_inputs(inputs, "", spec, max_length)
        {inputs, inputs["input_ids"], max_length}
      end

      update_inputs_fun = &update_decoder_inputs(&1, &2, &3, "")

      {prepare_inputs_fun, update_inputs_fun}
    end
  end

  defp encoder_decoder?(model) do
    inputs = Axon.get_inputs(model)
    Map.has_key?(inputs, "input_ids") and Map.has_key?(inputs, "decoder_input_ids")
  end

  defp prepare_decoder_inputs(inputs, prefix, spec, max_length) do
    input_ids = inputs[prefix <> "input_ids"]
    attention_mask = inputs[prefix <> "attention_mask"] || Nx.broadcast(1.0, input_ids)

    position_ids =
      attention_mask
      |> Nx.cumulative_sum(axis: 1)
      |> Nx.subtract(1)

    inputs =
      inputs
      |> Map.put(prefix <> "attention_mask", attention_mask)
      |> Map.put(prefix <> "position_ids", position_ids)

    batch_size = Nx.axis_size(input_ids, 0)
    cache = init_cache(spec, batch_size, max_length, inputs)
    Map.put(inputs, "cache", cache)
  end

  defp update_decoder_inputs(inputs, outputs, token_ids, prefix) do
    inputs
    |> Map.replace!(prefix <> "input_ids", token_ids)
    |> Map.replace!(prefix <> "attention_mask", Nx.broadcast(1.0, token_ids))
    |> Map.update!(prefix <> "position_ids", fn position_ids ->
      position_ids
      |> Nx.slice_along_axis(Nx.axis_size(position_ids, -1) - 1, 1, axis: -1)
      |> Nx.add(1)
    end)
    |> Map.replace!("cache", outputs.cache)
  end

  defp get_logits_processor(
         min_length_fun,
         eos_token_id,
         forced_bos_token_id,
         forced_eos_token_id
       ) do
    processors = [
      if min_length_fun && eos_token_id do
        &min_length_logits_processor(&1, &2,
          min_length_fun: min_length_fun,
          eos_token_id: eos_token_id
        )
      end,
      if forced_bos_token_id do
        &bos_token_logits_processor(&1, &2, bos_token_id: forced_bos_token_id)
      end,
      if forced_eos_token_id do
        &eos_token_logits_processor(&1, &2, eos_token_id: forced_eos_token_id)
      end
    ]

    fn logits, context ->
      for processor <- processors, processor, reduce: logits do
        logits -> processor.(logits, context)
      end
    end
  end

  deftransformp generate_impl(
                  inputs,
                  predict_fun,
                  params,
                  logits_processor_fun,
                  prepare_inputs_fun,
                  update_inputs_fun,
                  opts \\ []
                ) do
    {decoder_inputs, decoder_input_ids, max_length} = prepare_inputs_fun.(inputs, params)

    greedy(
      decoder_inputs,
      decoder_input_ids,
      predict_fun,
      params,
      logits_processor_fun,
      update_inputs_fun,
      [max_length: max_length] ++ opts
    )
  end

  defnp greedy(
          inputs,
          decoder_input_ids,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          opts \\ []
        ) do
    max_length = opts[:max_length]
    pad_token_id = opts[:pad_token_id]
    eos_token_id = opts[:eos_token_id]

    {batch_size, length} = Nx.shape(decoder_input_ids)

    if length > max_length do
      raise ArgumentError, "expected the input to be at most #{max_length} tokens, got: #{length}"
    end

    sequences = Nx.broadcast(pad_token_id, {batch_size, max_length})
    sequences = Nx.put_slice(sequences, [0, 0], decoder_input_ids)

    finished? = Nx.broadcast(Nx.tensor(0, type: :u8), {batch_size})

    input_length = length

    # The loop works with inputs of length 1, so if the initial input
    # is longer, we make the initial pass outside
    {sequences, length, finished?, inputs} =
      if length > 1 do
        greedy_step(
          sequences,
          length,
          finished?,
          inputs,
          input_length,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          pad_token_id: pad_token_id,
          eos_token_id: eos_token_id
        )
      else
        {sequences, length, finished?, inputs}
      end

    {sequences, _length, _finished?, _inputs, _params} =
      while {sequences, length, finished?, inputs, params},
            greedy_condition(finished?, length, max_length) do
        {sequences, length, finished?, inputs} =
          greedy_step(
            sequences,
            length,
            finished?,
            inputs,
            input_length,
            predict_fun,
            params,
            logits_processor_fun,
            update_inputs_fun,
            pad_token_id: pad_token_id,
            eos_token_id: eos_token_id
          )

        {sequences, length, finished?, inputs, params}
      end

    sequences
  end

  defnp greedy_condition(finished?, length, max_length) do
    not Nx.all(finished?) and length < max_length
  end

  defnp greedy_step(
          sequences,
          length,
          finished?,
          inputs,
          input_length,
          predict_fun,
          params,
          logits_processor_fun,
          update_inputs_fun,
          opts
        ) do
    pad_token_id = opts[:pad_token_id]
    eos_token_id = opts[:eos_token_id]

    outputs = predict_fun.(params, inputs)

    logits = outputs.logits[[0..-1//1, -1]]

    logits =
      logits_processor_fun.(logits, %{
        sequences: sequences,
        length: length,
        input_length: input_length
      })

    token_id = Nx.argmax(logits, axis: -1)

    token_id = Nx.select(finished?, pad_token_id, token_id)

    finished? =
      case eos_token_id do
        nil -> finished?
        eos_token_id -> finished? or token_id == eos_token_id
      end

    token_id = Nx.new_axis(token_id, -1)

    sequences = Nx.put_slice(sequences, [0, length], token_id)

    inputs = update_inputs_fun.(inputs, outputs, token_id)

    {sequences, length + 1, finished?, inputs}
  end

  # Logit processors

  defnp bos_token_logits_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:bos_token_id])
    bos_token_id = opts[:bos_token_id]

    if context.length == 1 do
      force_token_id(logits, token_id: bos_token_id)
    else
      logits
    end
  end

  defnp eos_token_logits_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:eos_token_id])
    eos_token_id = opts[:eos_token_id]

    max_length = Nx.axis_size(context.sequences, 1)

    if context.length == max_length - 1 do
      force_token_id(logits, token_id: eos_token_id)
    else
      logits
    end
  end

  defnp min_length_logits_processor(logits, context, opts \\ []) do
    opts = keyword!(opts, [:eos_token_id, :min_length_fun])
    eos_token_id = opts[:eos_token_id]
    min_length_fun = opts[:min_length_fun]

    min_length = min_length_fun.(context.input_length)

    if context.length < min_length do
      ignore_token_id(logits, token_id: eos_token_id)
    else
      logits
    end
  end

  defnp force_token_id(logits, opts \\ []) do
    token_id = opts[:token_id]

    batch_size = Nx.axis_size(logits, 0)

    Nx.Constants.neg_infinity()
    |> Nx.broadcast(logits)
    |> Nx.put_slice([0, token_id], Nx.broadcast(0, {batch_size, 1}))
  end

  defnp ignore_token_id(logits, opts \\ []) do
    token_id = opts[:token_id]

    batch_size = Nx.axis_size(logits, 0)

    Nx.put_slice(
      logits,
      [0, token_id],
      Nx.broadcast(Nx.Constants.neg_infinity(), {batch_size, 1})
    )
  end
end

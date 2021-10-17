# from ...utils import logging
# from ..t5.modeling_t5 import T5EncoderModel, T5ForConditionalGeneration, T5Model
from ..t5.modeling import T5EncoderModel, T5ForConditionalGeneration, T5Model, T5PreTrainedModel, T5Stack, T5LayerNorm, T5DenseReluDense, T5DenseGatedGeluDense, T5Attention
# from .configuration_mt5 import MT5Config

import copy
import logging
import math

import numpy as np
import paddle

import paddle.nn as nn
import paddle.nn.functional as F

from ..model_utils import PretrainedModel, register_base_model
from ..nezha.modeling import ACT2FN

from .utils import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    ModelOutput,
    Config
)
from ..generation_utils import BeamSearchScorer


logger = logging.getLogger(__name__)


def finfo(dtype):
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)



class MT5PreTrainedModel(T5PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    base_model_prefix = "mt5"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "mt5-large": {
            "vocab_size": 250112,
            "d_model": 1024,
            "d_kv": 64,
            "d_ff": 2816,
            "num_layers": 24,
            "num_decoder_layers": 24,
            "num_heads": 16,
            "relative_attention_num_buckets": 32,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "initializer_factor": 1.0,
            "feed_forward_proj": "gated-gelu",
            "is_encoder_decoder": True,
            "use_cache": True,
            "tie_word_embeddings": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "decoder_start_token_id": 0,

            "return_dict": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "is_decoder": False,
            "use_return_dict": True
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "mt5-large": "E:/论文复现/model_state.pdparams"
        }
    }


    @property
    def dummy_inputs(self):
        DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
        DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]
        input_ids = paddle.to_tensor(DUMMY_INPUTS, dtype=paddle.int64)
        input_mask = paddle.to_tensor(DUMMY_MASK, dtype=paddle.int64)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def init_weights(self):
        """
        Initializes and tie weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)


    def _init_weights(self, layer):
        """Initialize the weights"""
        factor = (
            self.pd_config.initializer_factor
        )  # Used for testing weights initialization
        if isinstance(layer, T5LayerNorm):
            layer.weight.set_value(paddle.ones_like(layer.weight)*factor)
        elif isinstance(layer, T5Model):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            layer.shared.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * 1.0,
                    shape=layer.shared.weight.shape ))
        elif isinstance(layer, (T5ForConditionalGeneration, T5EncoderModel)):
            layer.t5.shared.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * 1.0,
                    shape=layer.t5.shared.weight.shape ))


        elif isinstance(layer, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            layer.wi.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((self.pd_config.d_model) ** -0.5),
                    shape=layer.wi.weight.shape ))

            if hasattr(layer.wi, "bias") and layer.wi.bias is not None:
                layer.wi.bias.set_value(paddle.zeros_like(layer.wi.bias))

            layer.wo.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((self.pd_config.d_ff) ** -0.5),
                    shape=layer.wo.weight.shape ))

            if hasattr(layer.wo, "bias") and layer.wo.bias is not None:
                layer.wo.bias.set_value(paddle.zeros_like(layer.wo.bias))

        elif isinstance(layer, T5DenseGatedGeluDense):
            layer.wi_0.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((self.pd_config.d_model) ** -0.5),
                    shape=layer.wi_0.weight.shape ))
            if hasattr(layer.wi_0, "bias") and layer.wi_0.bias is not None:
                layer.wi_0.bias.set_value(paddle.zeros_like(layer.wi_0.bias))

            layer.wi_1.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((self.pd_config.d_model) ** -0.5),
                    shape=layer.wi_1.weight.shape ))
            if hasattr(layer.wi_1, "bias") and layer.wi_1.bias is not None:
                layer.wi_1.bias.set_value(paddle.zeros_like(layer.wi_1.bias))

            layer.wo.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((self.pd_config.d_ff) ** -0.5),
                    shape=layer.wo.weight.shape ))

            if hasattr(layer.wo, "bias") and layer.wo.bias is not None:
                layer.wo.bias.set_value(paddle.zeros_like(layer.wo.bias))
        elif isinstance(layer, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.pd_config.d_model
            key_value_proj_dim = self.pd_config.d_kv
            n_heads = self.pd_config.num_heads

            layer.q.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((d_model * key_value_proj_dim) ** -0.5),
                    shape=layer.q.weight.shape ))

            layer.k.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * (d_model ** -0.5),
                    shape=layer.k.weight.shape ))

            layer.v.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * (d_model ** -0.5),
                    shape=layer.v.weight.shape ))

            layer.o.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((n_heads * key_value_proj_dim) ** -0.5),
                    shape=layer.o.weight.shape ))


            if layer.has_relative_attention_bias:
                layer.relative_attention_bias.weight.set_value(
                    paddle.normal(
                        mean=0.0,
                        std=factor * ((d_model) ** -0.5),
                        shape=layer.relative_attention_bias.weight.shape ))


    def _shift_right(self, input_ids):
        decoder_start_token_id = self.pd_config.decoder_start_token_id
        pad_token_id = self.pd_config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.pd_config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = paddle.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        assert (
            pad_token_id is not None
        ), "self.model.pd_config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = paddle.where(
            shifted_input_ids == -100, paddle.to_tensor(pad_token_id,dtype=shifted_input_ids.dtype), shifted_input_ids
        )

        assert paddle.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

@register_base_model
class MT5Model(MT5PreTrainedModel):
    def __init__(self, **kwargs):
        super().__init__()
        pd_config = Config(**kwargs)
        self.pd_config = pd_config

        self.shared = nn.Embedding(pd_config.vocab_size, pd_config.d_model)

        encoder_config = copy.deepcopy(pd_config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(pd_config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = pd_config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        use_cache = use_cache if use_cache is not None else self.pd_config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.pd_config.use_return_dict
        )

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class MT5ForConditionalGeneration(MT5PreTrainedModel):
    def __init__(self, mt5):
        super().__init__()
        self.mt5 = mt5
        self.pd_config = mt5.pd_config
        if not self.pd_config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                self.pd_config.d_model, self.pd_config.vocab_size, bias_attr=False
            )

        self.eos_token_id = self.pd_config.eos_token_id
        self.pad_token_id = self.pd_config.pad_token_id
        self.init_weights()

    def get_input_embeddings(self):
        return self.mt5.shared

    def set_input_embeddings(self, new_embeddings):
        self.mt5.shared = new_embeddings
        self.mt5.encoder.set_input_embeddings(new_embeddings)
        self.mt5.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        if not self.pd_config.tie_word_embeddings:
            return self.mt5.shared
        return self.lm_head

    def get_encoder(self):
        return self.mt5.encoder

    def get_decoder(self):
        return self.mt5.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        use_cache = use_cache if use_cache is not None else self.pd_config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.pd_config.use_return_dict
        )

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.mt5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

        # Decode
        decoder_outputs = self.mt5.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.pd_config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.pd_config.d_model ** -0.5)
            lm_logits = paddle.matmul(sequence_output,self.mt5.shared.weight, transpose_y=True)
        else:
            lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.reshape(shape=[-1, lm_logits.shape[-1]]), labels.flatten()
            )
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids, model_kwargs
    ):
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            model_kwargs["encoder_outputs"] = encoder(input_ids, return_dict=True, **encoder_kwargs)

        return model_kwargs

    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.pd_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.pd_config.get(bos_token_id,None)

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif (
            hasattr(self.pd_config, "decoder")
            and hasattr(self.pd_config.decoder, "decoder_start_token_id")
            and self.pd_config.decoder.decoder_start_token_id is not None
        ):
            return self.pd_config.decoder.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif (
            hasattr(self.pd_config, "decoder")
            and hasattr(self.pd_config.decoder, "bos_token_id")
            and self.pd_config.decoder.bos_token_id is not None
        ):
            return self.pd_config.decoder.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )


    def _prepare_decoder_input_ids_for_generation(
        self, input_ids, decoder_start_token_id: int = None, bos_token_id: int = None
    ):
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
            paddle.ones(shape=(input_ids.shape[0], 1),dtype=paddle.int64) * decoder_start_token_id
        )
        return decoder_input_ids


    @paddle.no_grad()
    def generate(self,
                 input_ids=None,
                 max_length=20,
                 min_length=0,
                 decode_strategy='greedy_search',
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
                 num_beams=1,
                 length_penalty=1.0,
                 early_stopping=False,
                 bos_token_id=None,
                 eos_token_id=None,
                 pad_token_id=None,
                 num_return_sequences=1,
                 use_cache=True,
                 **model_kwargs):

        # params check
        bos_token_id = bos_token_id if bos_token_id is not None else getattr(
            self, 'bos_token_id', None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(
            self, 'eos_token_id', None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(
            self, 'pad_token_id', None)

        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs[
                "attention_mask"] = self.prepare_attention_mask_for_generation(
                    input_ids, pad_token_id, eos_token_id)

        if pad_token_id is None and eos_token_id is not None:
            print("Setting `pad_token_id` to `eos_token_id`:{} for "
                  "open-end generation.".format(eos_token_id))
            pad_token_id = eos_token_id

        # TODO Add relevant processing for encoder_decoder model.
        if self.pd_config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=self.pd_config.decoder_start_token_id, bos_token_id=bos_token_id
                )
            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")



        model_kwargs["use_cache"] = use_cache

        max_length += input_ids.shape[-1]

        logits_processors = self.get_logits_processor(min_length, eos_token_id)

        if decode_strategy == 'greedy_search':
            if num_return_sequences > 1:
                raise ValueError(
                    "`num_return_sequences` has to be 1, but is {} "
                    "when doing greedy search.".format(num_return_sequences))

            return self.greedy_search(input_ids, logits_processors, max_length,
                                      pad_token_id, eos_token_id,
                                      **model_kwargs)

        elif decode_strategy == 'sampling':
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs)

            return self.sample(input_ids, logits_processors, max_length,
                               pad_token_id, eos_token_id, top_k, top_p,
                               temperature, **model_kwargs)

        elif decode_strategy == 'beam_search':
            batch_size = input_ids.shape[0]
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to "
                    "`num_beams`. But received `num_return_sequences` is {}, "
                    "`num_beams` is {}".format(num_return_sequences, num_beams))
            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` has to be bigger than 1. But received "
                    "`num_beams` is {}. If `num_beams` is 1, `decode_strategy` "
                    "should be 'greedy_search'".format(num_beams))

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences)

            input_ids, model_kwargs = self.expand_inputs_for_generation(
                input_ids, expand_size=num_beams, **model_kwargs)

            return self.beam_search(input_ids, beam_scorer, logits_processors,
                                    max_length, pad_token_id, eos_token_id,
                                    **model_kwargs)

        else:
            raise ValueError(
                '`decode_strategy` must be one of "greedy_search", "sampling" '
                'and "beam_search".')

    #### for mt5
    @staticmethod
    def expand_inputs_for_generation(input_ids,
                                     expand_size,
                                     attention_mask=None,
                                     encoder_outputs=None,
                                     **model_kwargs):
        index = paddle.tile(
            paddle.arange(input_ids.shape[0]).unsqueeze(-1),
            [1, expand_size]).reshape([-1])

        input_ids = paddle.index_select(input_ids, index)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = paddle.index_select(attention_mask,
                                                                 index)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.index_select(token_type_ids,
                                                                 index)

        assert encoder_outputs is not None
        encoder_outputs["last_hidden_state"] = paddle.index_select(encoder_outputs.last_hidden_state,index)
        model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs

    def beam_search(self, input_ids, beam_scorer, logits_processors, max_length,
                    pad_token_id, eos_token_id, **model_kwargs):
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size)

        beam_scores = paddle.zeros(
            (batch_size, num_beams), dtype=paddle.get_default_dtype())
        beam_scores[:, 1:] = -1e9
        beam_scores = paddle.reshape(beam_scores, [-1])

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs,return_dict=True)
            logits = outputs.logits
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # beam search
            # [batch_size * num_beams, vocab_size]
            next_scores = F.softmax(logits)
            next_scores = paddle.log(next_scores)

            next_scores = next_scores + beam_scores.unsqueeze(-1)
            # reshape for beam search
            vocab_size = next_scores.shape[-1]
            next_scores = next_scores.reshape(
                [batch_size, num_beams * vocab_size])

            next_scores, next_tokens = paddle.topk(
                next_scores, 2 * num_beams, axis=1)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                origin_len=origin_len,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id, )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            cur_len += 1
            input_ids = paddle.concat(
                [
                    paddle.index_select(input_ids, beam_idx),
                    beam_next_tokens.unsqueeze(-1)
                ],
                axis=-1)

            if beam_scorer.is_done:
                break
            model_kwargs = self.update_model_kwargs_for_generation(outputs,
                                                                   model_kwargs)

            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id)
        return pred_ids, scores



    def greedy_search(self, input_ids, logits_processors, max_length,
                      pad_token_id, eos_token_id, **model_kwargs):

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype='bool')
        scores = paddle.full(
            [batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs,return_dict=True)

            logits = outputs.logits
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # greedy
            probs = F.softmax(logits)
            probs = paddle.log(probs)
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens,
                                           paddle.full_like(next_tokens,
                                                            pad_token_id))

            scores = self.update_scores_for_generation(
                scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(
                    unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break

            model_kwargs = self.update_model_kwargs_for_generation(outputs,
                                                                   model_kwargs
                                                                   )
        return input_ids, scores

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id,
                                              eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id))
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids == pad_token_id
                              ).astype(paddle.get_default_dtype()) * -1e9
        else:
            attention_mask = paddle.zeros_like(
                input_ids, dtype=paddle.get_default_dtype())
        return attention_mask

    def update_model_kwargs_for_generation(
        self, outputs , model_kwargs
    ) :
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update attention mask
        if not self.pd_config.is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.cat(
                    [attention_mask, paddle.ones(shape=(attention_mask.shape[0], 1),dtype=attention_mask.dtype)], dim=-1
                )

        return model_kwargs


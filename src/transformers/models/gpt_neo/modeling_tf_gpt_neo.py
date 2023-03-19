# coding=utf-8
# Copyright 2023 VOID Operations, EleutherAI, and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TF 2.0 GPT-Neo model."""

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    DUMMY_INPUTS,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSharedEmbeddings,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_gpt_neo import GPTNeoConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neo-1.3B"
_CONFIG_FOR_DOC = "GPTNeoConfig"

GPTNeo_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-neo-1.3B",
    # See all GPT-Neo models at https://huggingface.co/models?filter=gpt_neo
]


def create_sinusoidal_positions(num_pos: int, dim: int) -> tf.Tensor:
    inv_freq = tf.cast(1.0 / (10000 ** (tf.range(0, dim, 2) / dim)), tf.float32)
    sinusoid_inp = tf.cast(tf.einsum("i , j -> i j", tf.range(num_pos, dtype=tf.float32), inv_freq), tf.float32)
    sin, cos = tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)
    out = tf.concat((sin, cos), axis=1)
    return out


def rotate_every_two(x: tf.Tensor) -> tf.Tensor:
    rotate_half_tensor = tf.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), axis=-1)
    new_shape = shape_list(rotate_half_tensor)[:-2] + [tf.math.reduce_prod(shape_list(rotate_half_tensor)[-2:])]
    rotate_half_tensor = tf.reshape(rotate_half_tensor, new_shape)
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor: tf.Tensor, sincos: tf.Tensor) -> tf.Tensor:
    sin_pos, cos_pos = sincos
    sin_pos = tf.repeat(sin_pos[:, :, None, :], 2, 3)
    cos_pos = tf.repeat(cos_pos[:, :, None, :], 2, 3)
    return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)

class TFGPTNeoAttention(tf.keras.layers.Layer):
    def __init__(self, config: GPTNeoConfig, layer_id: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]

        if self.attention_type in ["global", "local"]:
            self.attention = TFGPTNeoSelfAttention(config, self.attention_type)
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


class TFGPTNeoSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: GPTNeoConfig, attention_type: str, **kwargs):
        super().__init__(**kwargs)
        self.attention_type = attention_type

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.attn_dropout = tf.keras.layers.Dropout(float(config.attention_dropout))
        self.resid_dropout = tf.keras.layers.Dropout(float(config.resid_dropout))

        self.q_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="q_proj",
        )
        self.k_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="k_proj",
        )
        self.v_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="v_proj",
        )
        self.out_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="out_proj",
        )

        self.max_positions = config.max_position_embeddings
        self.lower_triangle_mask = tf.reshape(
            tf.cast(tf.experimental.numpy.tril(tf.ones((self.max_positions, self.max_positions))), tf.int8),
            (1, 1, self.max_positions, self.max_positions),
        )
        self.masked_bias = tf.constant(-1e9, dtype=tf.float32)

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        if attention_type == "local":
            self.lower_triangle_mask = tf.bitwise.bitwise_xor(
                self.lower_triangle_mask,
                tf.reshape(
                    tf.cast(tf.experimental.numpy.tril(tf.ones((self.max_positions, self.max_positions))), tf.int8),
                    (1, 1, self.max_positions, self.max_positions),
                ),
            )

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tf.reshape(tensor, new_shape)
        return tf.transpose(tensor, perm=[0, 2, 1, 3])  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
        new_shape = tensor.shape[:-2] + (num_heads * attn_head_size,)
        return tf.reshape(tensor, new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)

        attn_weights = tf.matmul(query, key, transpose_b=True)

        query_length, key_length = tf.shape(query)[-2], tf.shape(key)[-2]
        causal_mask = self.lower_triangle_mask[:, :, key_length - query_length : key_length, :key_length]
        mask_value = tf.cast(tf.keras.backend.epsilon(), attn_weights.dtype)
        attn_weights = tf.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = tf.matmul(attn_weights, value)

        return attn_output, attn_weights

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        layer_past: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        head_mask: Optional[tf.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = tf.concat((past_key, key), axis=-2)
            value = tf.concat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class TFGPTNeoAttention(tf.keras.layers.Layer):
    def __init__(self, config: GPTNeoConfig, layer_id: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]

        if self.attention_type in ["global", "local"]:
            self.attention = TFGPTNeoSelfAttention(config, self.attention_type)
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


class TFGPTNeoMLP(tf.keras.layers.Layer):
    def __init__(self, intermediate_size: int, config: GPTNeoConfig, **kwargs):
        super().__init__(**kwargs)
        embed_dim = config.n_embd

        self.fc_in = tf.keras.layers.Dense(
            intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="fc_in"
        )
        self.fc_out = tf.keras.layers.Dense(
            embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="fc_out"
        )

        self.act = get_tf_activation(config.activation_function)
        self.dropout = tf.keras.layers.Dropout(config.embd_pdrop)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TFGPTNeoBlock(tf.keras.layers.Layer):
    def __init__(self, config: GPTNeoConfig, layer_id: int, **kwargs):
        super().__init__(**kwargs)
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        self.attn = TFGPTNeoAttention(config, layer_id)
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")
        self.mlp = TFGPTNeoMLP(inner_dim, config)

    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


@keras_serializable
class TFGPTNeoMainLayer(tf.keras.layers.Layer):
    config_class = GPTNeoConfig

    def __init__(self, config: GPTNeoConfig, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict

        self.num_hidden_layers = config.num_layers
        self.n_embd = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.embed = TFSharedEmbeddings(config.vocab_size, config.hidden_size, initializer_range=config.initializer_range)
        self.drop = tf.keras.layers.Dropout(config.resid_pdrop)
        self.h = [TFGPTNeoBlock(config, i) for i in range(config.num_layers)]

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, new_embeddings):
        self.embed = new_embeddings

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def call(
        self,
        inputs,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            past = inputs[1] if len(inputs) > 1 else past
            attention_mask = inputs[2] if len(inputs) > 2 else attention_mask
            token_type_ids = inputs[3] if len(inputs) > 3 else token_type_ids
            position_ids = inputs[4] if len(inputs) > 4 else position_ids
            head_mask = inputs[5] if len(inputs) > 5 else head_mask
            inputs_embeds = inputs[6] if len(inputs) > 6 else inputs_embeds
            output_attentions = inputs[7] if len(inputs) > 7 else output_attentions
            output_hidden_states = inputs[8] if len(inputs) > 8 else output_hidden_states
            return_dict = inputs[9] if len(inputs) > 9 else return_dict
            assert len(inputs) <= 10, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            past = inputs.get("past", past)
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            assert len(inputs) <= 10, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: tf.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, tf.float32)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and self.config.cross_attention_type == "2d":
            extended_attention_mask = tf.expand_dims(extended_attention_mask, axis=1)
        if self.config.is_decoder and self.config.cross_attention_type == "3d":
            extended_attention_mask = tf.expand_dims(extended_attention_mask, axis=1)
            extended_attention_mask = tf.expand_dims(extended_attention_mask, axis=1)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
            # head_mask = tf.constant([0] * self.config.num_hidden_layers)

        embedding_output = self.embed(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        embedding_output = self.drop(embedding_output)

        hidden_states = embedding_output
        all_attentions = ()
        all_hidden_states = ()
        for i, layer_module in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                past=past,
                attention_mask=extended_attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.ln_f(hidden_states)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class TFGPTNeoPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoConfig
    base_model_prefix = "transformer"
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"h.\d+.attn.bias"]

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        dummy = {"input_ids": tf.constant(DUMMY_INPUTS, dtype=tf.int32)}
        return dummy

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)


GPT_NEO_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Parameters:
        config ([`GPTNeoConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT_NEO_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past` is `None` else `past[0].shape[-2]` (`sequence_length` of
            input past key value states). Indices of input sequence tokens in the vocabulary.

            If `past` is used, only input IDs that do not have their past calculated should be passed as `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past` output below). Can be used to speed up sequential decoding. The token ids which have their past
            given to this model should not be passed as input ids as they have already been computed.
        attention_mask (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple. This argument can be used
            in eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare GPT-Neo Model transformer outputting raw hidden-states without any specific head on top.",
    GPTNeo_START_DOCSTRING,
)
class TFGPTNeoModel(TFGPTNeoPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPTNeoMainLayer(config, name="transformer")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTNeo_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past`). Set to `False` during training, `True` during generation
        """

        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values) if self.config.use_cache else None
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutputWithPast(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            hidden_states=hs,
            attentions=attns,
        )


@add_start_docstrings(
    """
    The GPT-Neo Model transformer with a language modeling head on top.
    """,
    GPTNeo_START_DOCSTRING,
)
class TFGPTNeoForCausalLM(TFGPTNeoPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"lm_head.weight",
        r"h\.\d+\.attn\.attention\.bias",
    ]
    _keys_to_ignore_on_save = [r"lm_head.weight"]

    def __init__(self, config: GPTNeoConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPTNeoModel(config)
        self.lm_head = tf.keras.layers.Dense(
            config.vocab_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="lm_head",
        )

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = tf.cast(lm_logits, tf.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

            lm_logits = tf.cast(lm_logits, hidden_states.dtype)
            loss = tf.cast(loss, hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def serving_output(self, output):
        return output.logits

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[tf.Tensor]], beam_idx: tf.Tensor
    ) -> Tuple[Tuple[tf.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )


@add_start_docstrings(
    """
    The GPT-Neo Model transformer with a sequence classification head on top (linear layer).

    [`GPTNeoForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT, GPT-2, GPT-Neo) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPTNeo_START_DOCSTRING,
)
class TFGPTNeoForSequenceClassification(TFGPTNeoPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_missing = [r"h.\d+.attn.masked_bias", r"h.\d+.attn.bias", r"lm_head.weight"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFGPTNeoMainLayer(config, name="transformer")
        self.score = tf.keras.layers.Dense(
            self.num_labels,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="score",
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTNeo_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        labels (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        logits_shape = shape_list(logits)
        in_logits = None
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    tf.reduce_sum(
                        tf.cast(
                            tf.math.not_equal(input_ids, self.config.pad_token_id),
                            dtype=input_ids.dtype,
                        ),
                        -1,
                        keepdims=False,
                    )
                    - 1
                )
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        loss = None

        if labels is not None:
            if self.config.pad_token_id is None and logits_shape[0] != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0 : logits_shape[0], sequence_lengths]

            loss = self.hf_compute_loss(tf.reshape(labels, [-1]), tf.reshape(in_logits, [-1, self.num_labels]))
        pooled_logits = in_logits if in_logits is not None else logits

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values) if self.config.use_cache else None
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutputWithPast(
            logits=output.logits, past_key_values=pkv, hidden_states=hs, attentions=attns
        )


@add_start_docstrings(
    """
    The GPT-Neo Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPTNeo_START_DOCSTRING,
)
class TFGPTNeoForQuestionAnswering(TFGPTNeoPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_missing = [r"h.\d+.attn.masked_bias", r"h.\d+.attn.bias", r"lm_head.weight"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFGPTNeoMainLayer(config, name="transformer")
        self.qa_outputs = tf.keras.layers.Dense(
            self.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTNeo_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        start_positions: Optional[Union[np.ndarray, tf.Tensor]] = None,
        end_positions: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def serving_output(self, output: TFQuestionAnsweringModelOutput) -> TFQuestionAnsweringModelOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFQuestionAnsweringModelOutput(
            start_logits=output.start_logits, end_logits=output.end_logits, hidden_states=hs, attentions=attns
        )

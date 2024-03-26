import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio
from .untool import Tool, make_np2c, data_type, data_type_map

from tpu_perf.infer import SGInfer, nptype

if TYPE_CHECKING:
    from .model import Whisper

@torch.no_grad()
def detect_language(
    model: "Whisper", mel: Tensor, tokenizer: Tokenizer = None
) -> Tuple[Tensor, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : Tensor, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages
        )
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    if model.log:
        import pdb; pdb.set_trace()
        print(f"[Log] detect_language")
        print(f"[Log] call encoder: {model.encoder_infer}")
        print(f"[Log] input:")
        print(f"[Log] mel: {mel}")
    start_time = time.time()
    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        if model.inference:
            # import pdb; pdb.set_trace()
            # mel = mel.numpy()
            encoder_info = model.tool.model_info(model.encoder_handle)
            mel_input_dtype = data_type_map[encoder_info['input_dtypes'][0]]
            mel = mel.numpy().astype(mel_input_dtype)
            mel = mel if mel.flags.c_contiguous else np.ascontiguousarray(mel)

            model.tool.copy_data_from_numpy(model.tool.get_input_tensor(model.runtime1, 0), make_np2c(mel), data_type[mel_input_dtype])
            model.tool.force_host_to_device(model.tool.get_input_tensor(model.runtime1, 0), model.handle)

            mel_out = np.empty(encoder_info[0]['output_shapes'][0], dtype=data_type_map[encoder_info['output_dtypes'][0]])
            model.tool.copy_data_from_numpy(model.tool.get_output_tensor(model.runtime1, 0), make_np2c(mel_out), encoder_info['output_dtypes'][0])
            model.tool.inference(model.runtime1)
            model.tool.copy_output_data_to_host(model.runtime1)
            mel_out = torch.from_numpy(mel_out)
            # import pdb; pdb.set_trace()
            # model.tool.print_output_data(model.runtime1)

            # _ = model.encoder_infer.put(mel)
            # _, result, _ = model.encoder_infer.get()
            print(f"encoder infer time: {time.time() - start_time}")
            model.time += time.time() - start_time
            # mel_out = torch.from_numpy(result[0])

            # import pdb; pdb.set_trace()
            # if model.fp16:
            #     mel_out = torch.from_numpy(result[0].astype(np.float16))
            # else:
            #     mel_out = torch.from_numpy(result[0])
        else:
            # import pdb; pdb.set_trace()
            if model.export_mode:
                pass
                model_name= f"encoder_{model.model_name}_{model.beam_size}beam_{model.padding_size}pad"
                onnx_input_dict = {"mel":mel}
                import os
                encoder_folder = "./encoder/"
                if not os.path.exists(encoder_folder):
                    os.makedirs(encoder_folder)
                np.savez(encoder_folder + model_name + "_inputs.npz", **onnx_input_dict)
                if model.export_mode == "onnx":
                    onnx_input_names = ["mel"]
                    onnx_output_names = ["audio_features",]

                    torch.onnx.export(
                        model.encoder,
                        (mel,),  # Pass the actual input data
                        encoder_folder + model_name + ".onnx",
                        verbose=True,
                        input_names=onnx_input_names,  # Provide input names
                        output_names=onnx_output_names,  # Provide output names
                        opset_version=15,  # ONNX opset version to use
                    )
                elif model.export_mode == "pt":
                    torch.jit.trace(model.encoder, (mel,)).save(model_name + ".pt")
            mel_out = model.encoder(mel)
        if model.log:
            # import pdb; pdb.set_trace()
            print(f"[Log] output:")
            print(f"[Log] mel: {mel_out}")
        # mel = model.encoder(mel)
        model.call_encoder += 1
        print(f"detect_language encoder time: {time.time() - start_time}")

    # forward pass using a single token, startoftranscript
    n_audio = mel_out.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    # import pdb; pdb.set_trace()
    start_time = time.time()
    # logits = model.logits(x, mel)[:, 0] # TODO: export model
    logits = model.logits(x, mel_out)[:, 0].float()
    print(f"logits time: {time.time() - start_time}")
    # logits = model.logits_with_positional_embedding_firstly(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation
    padding_size: int = 448 # max pre-allocation of key-value cache

@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(
            self, 
            source_indices, 
            self_attention_kcache: Tensor, 
            self_attention_vcache: Tensor, 
            cross_attention_kcache: Tensor, 
            cross_attention_vcache: Tensor
        ) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass

class LogitsInferenceLoop(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model: "Whisper" = model
        self.decoder = model.decoder
        self.embedding = model.decoder.token_embedding
        self.blocks = model.decoder.blocks
        self.dims = model.dims
        self.n_state = self.dims.n_text_state
        self.n_head = self.dims.n_text_head
        self.scale = (self.n_state // self.n_head) ** -0.25
    
    def forward(
            self, 
            x, 
            xa, 
            positional_embedding, 
            mask, 
        ) -> Tensor:
        x_embedding = self.model.decoder.token_embedding(x)
        x = x_embedding + positional_embedding
        i = 0

        for block in self.blocks:
            attn_ln_x = block.attn_ln(x)
            q = block.attn.query(attn_ln_x)
            k = block.attn.key(attn_ln_x)
            v = block.attn.value(attn_ln_x)

            q = q * self.scale
            k = k * self.scale
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            qk = q @ k
            qk = qk + mask.permute(0, 2, 1, 3)
            w = F.softmax(qk, dim=-1)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            tmp_out = block.attn.out(wv)
            
            x = x + tmp_out
            cross_attn_ln_x = block.cross_attn_ln(x)
            q = block.cross_attn.query(cross_attn_ln_x)
            k = block.cross_attn.key(xa)
            v = block.cross_attn.value(xa)

            q = q * self.scale
            k = k * self.scale

            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

            qk = q @ k
            w = F.softmax(qk, dim=-1)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            tmp_out = block.cross_attn.out(wv)
            x = x + tmp_out
            x = x + block.mlp(block.mlp_ln(x))
            i += 1
        x = self.decoder.ln(x)[:, -1:]
        logits = (
            x @ torch.transpose(self.decoder.token_embedding.weight, 0, 1)
        ).float()
        return logits[:, -1]

class LogitsInferenceFirstlyEmbedding(nn.Module):
    def __init__(self, token_embedding):
        super().__init__()
        self.token_embedding = token_embedding

    def forward(self, x, positional_embedding):
        return self.token_embedding(x) + positional_embedding

class LogitsInferenceFirstlyAttentionBlock(nn.Module):
    def __init__(self, block, dims):
        super().__init__()
        self.block = block
        self.dims = dims
        self.n_state = self.dims.n_text_state
        self.n_head = self.dims.n_text_head
        self.scale = (self.n_state // self.n_head) ** -0.25

    def forward(self, x, xa, mask) -> Tensor:
        attn_ln_x = self.block.attn_ln(x)
        q = self.block.attn.query(attn_ln_x)
        sattn_k = self.block.attn.key(attn_ln_x)
        sattn_v = self.block.attn.value(attn_ln_x)

        q = q * self.scale
        k = sattn_k * self.scale
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
        v = sattn_v.view(*sattn_v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        qk = qk + mask.permute(0, 2, 1, 3)
        w = F.softmax(qk, dim=-1)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        x = x + self.block.attn.out(wv)
        cross_attn_ln_x = self.block.cross_attn_ln(x)
        q = self.block.cross_attn.query(cross_attn_ln_x)
        cattn_k = self.block.cross_attn.key(xa)
        cattn_v = self.block.cross_attn.value(xa)

        q = q * self.scale
        k = cattn_k * self.scale
        v = cattn_v

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        w = F.softmax(qk, dim=-1)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        x = x + self.block.cross_attn.out(wv)
        x = x + self.block.mlp(self.block.mlp_ln(x))
        return x, sattn_k, sattn_v, cattn_k, cattn_v

class LogitsInferenceFirstlyMainProcess(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model: "Whisper" = model
        self.decoder = model.decoder
        self.embedding = model.decoder.token_embedding
        self.blocks = model.decoder.blocks
        self.dims = model.dims
        self.n_state = self.dims.n_text_state
        self.n_head = self.dims.n_text_head
        self.scale = (self.n_state // self.n_head) ** -0.25
        self.use_kvcache = model.use_kvcache
    
    def forward(
            self, 
            x, 
            xa, 
            positional_embedding, 
            mask, 
        ) -> Tensor:
        x_embedding = self.model.decoder.token_embedding(x)
        x = x_embedding + positional_embedding
        i = 0
        if self.use_kvcache:
            self_attention_kcache = []
            self_attention_vcache = []
            cross_attention_kcache = []
            cross_attention_vcache = []

        for block in self.blocks:
            attn_ln_x = block.attn_ln(x)
            q = block.attn.query(attn_ln_x)
            k = block.attn.key(attn_ln_x)
            v = block.attn.value(attn_ln_x)
            if self.use_kvcache:
                self_attention_kcache.append(k)
                self_attention_vcache.append(v)

            q = q * self.scale
            k = k * self.scale
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            qk = q @ k
            qk = qk + mask.permute(0, 2, 1, 3)
            w = F.softmax(qk, dim=-1)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            tmp_out = block.attn.out(wv)
            
            x = x + tmp_out
            cross_attn_ln_x = block.cross_attn_ln(x)
            q = block.cross_attn.query(cross_attn_ln_x)
            k = block.cross_attn.key(xa)
            v = block.cross_attn.value(xa)
            if self.use_kvcache:
                cross_attention_kcache.append(k)
                cross_attention_vcache.append(v)

            q = q * self.scale
            k = k * self.scale

            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

            qk = q @ k
            w = F.softmax(qk, dim=-1)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            tmp_out = block.cross_attn.out(wv)
            x = x + tmp_out
            x = x + block.mlp(block.mlp_ln(x))
            i += 1
        if self.use_kvcache:
            return x, tuple(self_attention_kcache), tuple(self_attention_vcache), tuple(cross_attention_kcache), tuple(cross_attention_vcache)
        return x

class LogitsInferenceFirstlyPostProcess(nn.Module):
    def __init__(self, decoder, no_speech):
        super().__init__()
        self.ln = decoder.ln
        self.token_embedding_weight = decoder.token_embedding.weight
        self.no_speech = no_speech

    def forward(self, x_sot, x_last):
        x = torch.cat((x_sot, x_last), dim=1)
        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding_weight, 0, 1)
        ).float()
        probs_at_sot = logits[:, 0].float().softmax(dim=-1)
        no_speech_prob = probs_at_sot[:, self.no_speech]
        return logits[:, -1], no_speech_prob

class LogitsInferenceLoopAttentionBlockWithKVCache(nn.Module):
    def __init__(self, block, dims):
        super().__init__()
        self.block = block
        self.dims = dims
        self.n_state = self.dims.n_text_state
        self.n_head = self.dims.n_text_head
        self.scale = (self.n_state // self.n_head) ** -0.25

    def forward(self, x, mask, self_attention_kcache, self_attention_vcache, cross_attention_kcache, cross_attention_cache) -> Tensor:
        attn_ln_x = self.block.attn_ln(x)
        q = self.block.attn.query(attn_ln_x)
        sattn_k = torch.cat([self_attention_kcache, self.block.attn.key(attn_ln_x)], dim=1)
        sattn_v = torch.cat([self_attention_vcache, self.block.attn.value(attn_ln_x)], dim=1)
        # sattn_k = self.block.attn.key(attn_ln_x)
        # sattn_v = self.block.attn.value(attn_ln_x)

        q = q * self.scale
        k = sattn_k * self.scale
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
        v = sattn_v.view(*sattn_v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        qk = qk + mask.permute(0, 2, 1, 3)
        w = F.softmax(qk, dim=-1)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        x = x + self.block.attn.out(wv)
        cross_attn_ln_x = self.block.cross_attn_ln(x)
        q = self.block.cross_attn.query(cross_attn_ln_x)
        k = cross_attention_kcache
        v = cross_attention_cache

        q = q * self.scale
        k = k * self.scale
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        w = F.softmax(qk, dim=-1)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        x = x + self.block.cross_attn.out(wv)
        x = x + self.block.mlp(self.block.mlp_ln(x))
        return x, sattn_k, sattn_v

class LogitsInferenceLoopWithKVCache(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model: "Whisper" = model
        self.decoder = model.decoder
        self.embedding = model.decoder.token_embedding
        self.blocks = model.decoder.blocks
        self.dims = model.dims
        self.n_state = self.dims.n_text_state
        self.n_head = self.dims.n_text_head
        self.scale = (self.n_state // self.n_head) ** -0.25
    
    def forward(
            self, 
            x, 
            positional_embedding, 
            mask, 
            self_attention_kcache, 
            self_attention_vcache, 
            cross_attention_kcache, 
            cross_attention_vcache
        ) -> Tensor:
        x_embedding = self.model.decoder.token_embedding(x)
        x = x_embedding + positional_embedding
        i = 0
        sattn_kcache = []
        sattn_vcache = []

        for block in self.blocks:
            attn_ln_x = block.attn_ln(x)
            q = block.attn.query(attn_ln_x)
            k = torch.cat([self_attention_kcache[i][:, 1:, ...], block.attn.key(attn_ln_x)], dim=1)
            v = torch.cat([self_attention_vcache[i][:, 1:, ...], block.attn.value(attn_ln_x)], dim=1)

            # renew kv cache
            sattn_kcache.append(k)
            sattn_vcache.append(v)

            q = q * self.scale
            k = k * self.scale
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

            qk = q @ k
            qk = qk + mask.permute(0, 2, 1, 3)
            w = F.softmax(qk, dim=-1)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            tmp_out = block.attn.out(wv)
            
            x = x + tmp_out
            cross_attn_ln_x = block.cross_attn_ln(x)
            q = block.cross_attn.query(cross_attn_ln_x)
            k = cross_attention_kcache[i]
            v = cross_attention_vcache[i]

            q = q * self.scale
            k = k * self.scale
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

            qk = q @ k
            w = F.softmax(qk, dim=-1)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            tmp_out = block.cross_attn.out(wv)
            x = x + tmp_out
            x = x + block.mlp(block.mlp_ln(x))

            # onnx_input = (x, mask, self_attention_kcache[i], self_attention_vcache[i], cross_attention_kcache[i], cross_attention_vcache[i])
            # onnx_input_names = ["x", "positional_embedding", "mask", "self_attention_kcache", "self_attention_vcache", "cross_attention_kcache", "cross_attention_vcache"]
            # onnx_output_names = ["x", "sattn_kcache", "sattn_vcache"]
            # onnx_input_dict = {}
            # model_name = f"decoder_block{i}_with_kvcache_{self.model.model_name}_{self.model.beam_size}beam_{self.model.padding_size}pad"
            # for name, value in zip(onnx_input_names, onnx_input):
            #     onnx_input_dict[name] = value
            # np.savez(model_name + "_inputs.npz", **onnx_input_dict)
            # attn_block = LogitsInferenceLoopAttentionBlockWithKVCache(block, self.dims)
            # torch.onnx.export(
            #     attn_block,
            #     onnx_input,  # Pass the actual input data
            #     model_name + ".onnx",
            #     verbose=True,
            #     input_names=onnx_input_names,  # Provide input names
            #     output_names=onnx_output_names,  # Provide output names
            #     opset_version=15,  # ONNX opset version to use
            # )
            # exit()
            # x, sattn_k, sattn_v = attn_block(*onnx_input)
            # sattn_kcache.append(sattn_k)
            # sattn_vcache.append(sattn_v)
            i += 1
        x = self.decoder.ln(x)[:, -1:]
        logits = (
            x @ torch.transpose(self.decoder.token_embedding.weight, 0, 1)
        ).float()
        
        return logits[:, -1], tuple(sattn_kcache), tuple(sattn_vcache)

class PyTorchInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model

    def rearrange_kv_cache(
            self, 
            source_indices, 
            self_attention_kcache: Tuple[Tensor] = None, 
            self_attention_vcache: Tuple[Tensor] = None, 
        ):
        if source_indices != list(range(len(source_indices))):
            # import pdb; pdb.set_trace()
            if self_attention_kcache:
                for i in range(len(self_attention_kcache)):
                    self_attention_kcache[i] = self_attention_kcache[i][source_indices]
                    self_attention_vcache[i] = self_attention_vcache[i][source_indices]
                return
            indices = np.array(source_indices, dtype=np.int32)
            indices = indices if indices.flags.contiguous else indices.copy()
            tool = self.model.tool
            tool.copy_data_from_numpy(tool.get_input_tensor(self.model.kvcache_rearrange_runtime[0], 1), make_np2c(indices), 6)
            tool.force_host_to_device(tool.get_input_tensor(self.model.kvcache_rearrange_runtime[0], 1), self.model.handle)
            for i in range(2 * self.model.dims.n_text_layer):
                # print(f"rearange {i} layer")
                # import pdb; pdb.set_trace()
                runtime = self.model.kvcache_rearrange_runtime[i]
                tool.malloc_device_address(runtime)
                tool.inference(runtime)
            return

class SequenceRanker:
    def rank(
        self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, 
        tokens: Tensor, 
        logits: Tensor, 
        sum_logprobs: Tensor, 
    ) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, 
        tokens: Tensor, 
        logits: Tensor, 
        sum_logprobs: Tensor, 
        self_attention_kcache: Tensor = None, 
        self_attention_vcache: Tensor = None, 
    ) -> Tuple[Tensor, bool]:
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()


class BeamSearchDecoder(TokenDecoder):
    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference: Inference,
        patience: Optional[float] = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(
        self, 
        tokens: Tensor, 
        logits: Tensor, 
        sum_logprobs: Tensor, 
        self_attention_kcache: Tensor = None, 
        self_attention_vcache: Tensor = None, 
    ) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        # import pdb; pdb.set_trace()

        if self_attention_kcache:
            self.inference.rearrange_kv_cache(
                source_indices, 
                self_attention_kcache, 
                self_attention_vcache, 
            )
        else:
            self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if (
                len(sequences) < self.beam_size
            ):  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: Tensor, tokens: Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: Tensor, tokens: Tensor):
        logits[:, self.suppress_tokens] = -np.inf


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: Tensor, tokens: Tensor):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = [t for t in sampled_tokens.tolist()]
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

            timestamps = sampled_tokens[
                sampled_tokens.ge(self.tokenizer.timestamp_begin)
            ]
            if timestamps.numel() > 0:
                # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                # also force each segment to have a nonzero length, to prevent infinite looping
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                logits[k, self.tokenizer.timestamp_begin : timestamp_last] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(
                dim=-1
            )
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model
        
        language = options.language or "en"
        # import pdb;pdb.set_trace()
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2
        self.n_text_head = self.model.dims.n_text_head
        self.n_text_layer = self.model.dims.n_text_layer
        self.padding_size = options.padding_size

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = PyTorchInference(model, len(self.initial_tokens))

        # Inference module with torch for tracing model
        if not self.model.inference:
            model.eval()
            # no kv cache inference
            self.inference_main_process = LogitsInferenceFirstlyMainProcess(model)
            self.inference_post_process = LogitsInferenceFirstlyPostProcess(model.decoder, self.tokenizer.no_speech)
            if self.model.use_kvcache:
                self.inference_loop = LogitsInferenceLoopWithKVCache(model)
            else:
                self.inference_loop = LogitsInferenceLoop(model)

            # infernece with kv cache
            # self.inference_firstly = LogitsInferenceFirstly(model)
            # self.inference_firstly_seperate = LogitsInferenceFirstlySeperate(model)
            # self.inference_firstly_main_process = LogitsInferenceFirstlyMainProcess(model)
            # self.inference_firstly_after_process = LogitsInferenceFirstlyAfterProcess(model.decoder, self.tokenizer.no_speech)

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: Tensor):
        if self.options.fp16:
            mel = mel.half()

        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            if self.model.log:
                # import pdb; pdb.set_trace()
                print(f"[Log] _get_audio_features")
                print(f"[Log] call encoder: {self.model.encoder_infer}")
                print(f"[Log] input:")
                print(f"[Log] mel: {mel}")
            start_time = time.time()
            if self.model.inference:
                # mel = mel.numpy()
                # mel = mel.numpy().astype(nptype(self.model.encoder_infer.get_input_info()["mel"]["dtype"]))



                encoder_info = self.model.tool.model_info(self.model.encoder_handle)
                mel_input_dtype = data_type_map[encoder_info['input_dtypes'][0]]
                mel = mel.numpy().astype(mel_input_dtype)
                mel = mel if mel.flags.c_contiguous else np.ascontiguousarray(mel)

                self.model.tool.copy_data_from_numpy(self.model.tool.get_input_tensor(self.model.runtime1, 0), make_np2c(mel), data_type[mel_input_dtype])
                self.model.tool.force_host_to_device(self.model.tool.get_input_tensor(self.model.runtime1, 0), self.model.handle)

                mel_out = np.empty(encoder_info[0]['output_shapes'][0], dtype=data_type_map[encoder_info['output_dtypes'][0]])
                self.model.tool.copy_data_from_numpy(self.model.tool.get_output_tensor(self.model.runtime1, 0), make_np2c(mel_out), encoder_info['output_dtypes'][0])
                self.model.tool.inference(self.model.runtime1)
                self.model.tool.copy_output_data_to_host(self.model.runtime1)
                audio_features = torch.from_numpy(mel_out)



                # _ = self.model.encoder_infer.put(mel)
                # _, result, _ = self.model.encoder_infer.get()
                # audio_features = torch.from_numpy(result[0])
                print(f"encoder time: {time.time() - start_time}")
                self.model.time += time.time() - start_time

                # if self.options.fp16:
                #     audio_features = torch.from_numpy(result[0].astype(np.float16))
                # else:
                #     audio_features = torch.from_numpy(result[0])
            else:
                audio_features = self.model.encoder(mel)
            # audio_features = self.model.encoder(mel)
            if self.model.log:
                # import pdb; pdb.set_trace()
                print(f"[Log] output:")
                print(f"[Log] mel: {mel}")
            self.model.call_encoder +=1
            print(f"_get_audio_features encoder time: {time.time() - start_time}")

        # TODO check dtype
        # if audio_features.dtype != (
        #     torch.float16 if self.options.fp16 else torch.float32
        # ):
        #     return TypeError(
        #         f"audio_features has an incorrect dtype: {audio_features.dtype}"
        #     )

        return audio_features

    def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs
    
    def _main_loop_cpu(self, audio_features: Tensor, tokens: Tensor):
        print("{:=^80}".format(f" start main_loop {self.model.main_loop_cnt} "))
        # import pdb; pdb.set_trace()
        self.model.main_loop_cnt += 1
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        initial_tokens_length = len(self.initial_tokens)
        padding_num = self.padding_size
        print("{:%^60}".format(f" initial padding size: {padding_num} "))

        # padding_num = self.padding_size
        # padding_num_with_kvcache = self.padding_size
    
        attention_mask_firstly = torch.empty(padding_num, padding_num).fill_(-10000).triu_(1)
        attention_mask_with_kvcache_max = torch.empty(448, 448).fill_(-10000).triu_(1)
        attention_mask_with_kvcache = attention_mask_with_kvcache_max[-padding_num:, -padding_num:]
        # attention_mask_with_kvcache = torch.empty(padding_num_with_kvcache, padding_num_with_kvcache).fill_(-10000).triu_(1)
        loop_start_time = time.time()
        if self.model.inference:
            tool = self.model.tool

        try:
            for i in range(self.sample_len):
                # print("{:=^80}".format(f" start {i} "))
                # print(f"tokens: {tokens}")
                # print(f"audio_features: {audio_features}")
                # import pdb; pdb.set_trace()
                if i == 0 or not self.model.use_kvcache:
                    tokens_input = F.pad(tokens, (padding_num - tokens.shape[-1], 0, 0, 0), value=0)
                    positional_embedding_input = F.pad(self.model.positional_embedding[:i+initial_tokens_length], (0, 0, padding_num - initial_tokens_length - i, 0), value=0)
                    mask = F.pad(attention_mask_firstly[:tokens.shape[-1], :tokens.shape[-1]], (padding_num - tokens.shape[-1], 0, 0, 0), value=-10000)
                    mask = F.pad(mask, (0, 0, padding_num - tokens.shape[-1], 0), value=0)
                    mask = mask.reshape(1, 1, *mask.shape).repeat(n_batch, self.n_text_head, 1, 1).permute(0, 2, 1, 3).contiguous()
                else:
                    tokens_input = tokens[:, -1:]
                    offset = i + initial_tokens_length - 1
                    positional_embedding_input = self.model.positional_embedding[offset:offset+1]
                    mask = attention_mask_with_kvcache[offset:offset+1].flip(1)
                    mask = mask.reshape(1, 1, *mask.shape).repeat(n_batch, self.n_text_head, 1, 1).permute(0, 2, 1, 3).contiguous()
                # import pdb; pdb.set_trace()

                if i == 0:
                    # if self.model.log:
                    #     print(f"[Log] decoder_main_infer")
                    #     print(f"[Log] input:")
                    #     print(f"[Log] bmodel_input: {(tokens_input, audio_features, positional_embedding_input, mask,)}")
                    #     import pdb; pdb.set_trace()
                    if self.model.export_mode:
                        pass
                        input = (tokens_input, audio_features, positional_embedding_input, mask,)
                        input_dict = {}
                        input_names = ["tokens_input", "audio_features", "positional_embedding_input", "mask"]
                        output_names = ["x",]
                        for name, value in zip(input_names, input):
                            input_dict[name] = value
                        if self.model.use_kvcache:
                            model_name = f"decoder_main_with_kvcache_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad"
                            for idx in range(self.model.dims.n_text_layer):
                                output_names.append(f"self_attn_kcache_in.{idx}")
                            for idx in range(self.model.dims.n_text_layer):
                                output_names.append(f"self_attn_vcache_in.{idx}")
                            for idx in range(self.model.dims.n_text_layer):
                                output_names.append(f"cross_attn_kcache_in.{idx}")
                            for idx in range(self.model.dims.n_text_layer):
                                output_names.append(f"cross_attn_vcache_in.{idx}")
                        else:
                            model_name = f"decoder_main_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad"
                        import os
                        decoder_main_folder = "./decoder_main_with_kvcache/"
                        if not os.path.exists(decoder_main_folder):
                            os.makedirs(decoder_main_folder)
                        np.savez(decoder_main_folder + model_name + "_inputs.npz", **input_dict)
                        if self.model.export_mode == "onnx":
                            torch.onnx.export(
                                self.inference_main_process,
                                input,  # Pass the actual input data
                                decoder_main_folder + model_name + ".onnx",
                                verbose=True,
                                input_names=input_names,  # Provide input names
                                output_names=output_names,  # Provide output names
                                opset_version=15,  # ONNX opset version to use
                            )
                        elif self.model.export_mode == "pt":
                            torch.jit.trace(self.inference_main_process, input).save(model_name + ".pt")
                    if self.model.use_kvcache:
                        x, self_attention_kcache, self_attention_vcache, cross_attention_kcache, cross_attention_vcache = self.inference_main_process(
                            tokens_input, 
                            audio_features, 
                            positional_embedding_input, 
                            mask, 
                            )
                        self_attention_kcache = list(self_attention_kcache)
                        self_attention_vcache = list(self_attention_vcache)
                        cross_attention_kcache = list(cross_attention_kcache)
                        cross_attention_vcache = list(cross_attention_vcache)
                    else:
                        x = self.inference_main_process(
                            tokens_input, 
                            audio_features, 
                            positional_embedding_input, 
                            mask, 
                            )
                    # import pdb; pdb.set_trace()
                    x_sot = x[:, padding_num - initial_tokens_length + self.sot_index:padding_num - initial_tokens_length + self.sot_index + 1]
                    x_last = x[:, -1:]
                    if self.model.export_mode:
                        pass
                        model_name = f"decoder_post_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad"
                        input = (x_sot, x_last)
                        input_dict = {}
                        input_names = ["x_sot", "x_last"]
                        output_names = ["logits", "no_speech_probs"]
                        for name, value in zip(input_names, input):
                            input_dict[name] = value
                        import os
                        decoder_post_folder = "./decoder_post/"
                        if not os.path.exists(decoder_post_folder):
                            os.makedirs(decoder_post_folder)
                        np.savez(decoder_post_folder + model_name + "_inputs.npz", **input_dict)
                        if self.model.export_mode == "onnx":
                            torch.onnx.export(
                                    self.inference_post_process,
                                    input,  # Pass the actual input data
                                    decoder_post_folder + model_name + ".onnx",
                                    verbose=True,
                                    input_names=input_names,  # Provide input names
                                    output_names=output_names,  # Provide output names
                                    opset_version=15,  # ONNX opset version to use
                                )
                        elif self.model.export_mode == "pt":
                            torch.jit.trace(self.inference_post_process, input).save(model_name + ".pt")
                    logits, no_speech_probs = self.inference_post_process(x_sot, x_last)
                    no_speech_probs = no_speech_probs.tolist()
                    # if self.model.use_kvcache:
                    #     for idx in range(len(self_attention_kcache)):
                    #         self_attention_kcache[idx] = self_attention_kcache[idx][:, 1:]
                    #         self_attention_vcache[idx] = self_attention_vcache[idx][:, 1:]
                    
                    # import pdb; pdb.set_trace()
                    self.model.call_decoder_firstly += 1

                else:
                    # if self.model.log:
                    #     print(f"[Log] decoder_loop_infer")
                    #     print(f"[Log] input:")
                    #     import pdb; pdb.set_trace()
                    if self.model.export_mode:
                        if self.model.use_kvcache:
                            input = (tokens_input, positional_embedding_input, mask, self_attention_kcache, self_attention_vcache, cross_attention_kcache, cross_attention_vcache,)
                            npz_input = (tokens_input, positional_embedding_input, mask, *self_attention_kcache, *self_attention_vcache, *cross_attention_kcache, *cross_attention_vcache,)
                            input_names = ["tokens_input", "positional_embedding_input", "mask",]
                            output_names = ["logits",]
                            model_name = f"decoder_loop_with_kvcache_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad"
                            for idx in range(self.model.dims.n_text_layer):
                                input_names.append(f"self_attn_kcache_in.{idx}")
                            for idx in range(self.model.dims.n_text_layer):
                                input_names.append(f"self_attn_vcache_in.{idx}")
                            for idx in range(self.model.dims.n_text_layer):
                                input_names.append(f"cross_attn_kcache_in.{idx}")
                            for idx in range(self.model.dims.n_text_layer):
                                input_names.append(f"cross_attn_vcache_in.{idx}")

                            for idx in range(self.model.dims.n_text_layer):
                                output_names.append(f"self_attn_kcache_out.{idx}")
                            for idx in range(self.model.dims.n_text_layer):
                                output_names.append(f"self_attn_vcache_out.{idx}")
                        else:
                            input = (tokens_input, audio_features, positional_embedding_input, mask,)
                            npz_input = (tokens_input, audio_features, positional_embedding_input, mask,)
                            input_names = ["tokens_input", "audio_features", "positional_embedding_input", "mask",]
                            output_names = ["logits",]
                            model_name = f"decoder_loop_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad"
                        input_dict = {}
                        for name, value in zip(input_names, npz_input):
                            input_dict[name] = value
                        import os
                        decoder_loop_folder = "./decoder_loop_with_kvcache/"
                        if not os.path.exists(decoder_loop_folder):
                            os.makedirs(decoder_loop_folder)
                        np.savez(decoder_loop_folder + model_name + "_inputs.npz", **input_dict)
                        if self.model.export_mode == "onnx":
                            torch.onnx.export(
                                self.inference_loop,
                                input,  # Pass the actual input data
                                decoder_loop_folder + model_name + ".onnx",
                                verbose=True,
                                input_names=input_names,  # Provide input names
                                output_names=output_names,  # Provide output names
                                opset_version=15,  # ONNX opset version to use
                            )
                        elif self.model.export_mode == "pt":
                            torch.jit.trace(self.inference_loop, input).save(model_name + ".pt")
                        exit()
                    else:
                        if self.model.use_kvcache:
                            logits, self_attention_kcache, self_attention_vcache = self.inference_loop(
                                tokens_input, 
                                positional_embedding_input, 
                                mask, 
                                self_attention_kcache, 
                                self_attention_vcache, 
                                cross_attention_kcache, 
                                cross_attention_vcache
                            )
                            self_attention_kcache = list(self_attention_kcache)
                            self_attention_vcache = list(self_attention_vcache)
                            cross_attention_kcache = list(cross_attention_kcache)
                            cross_attention_vcache = list(cross_attention_vcache)
                        else:
                            logits = self.inference_loop(tokens_input, audio_features, positional_embedding_input, mask)
                    # if self.model.use_kvcache:
                    #     for idx in range(len(self_attention_kcache)):
                    #         self_attention_kcache[idx] = self_attention_kcache[idx][:, 1:]
                    #         self_attention_vcache[idx] = self_attention_vcache[idx][:, 1:]

                    # import pdb; pdb.set_trace()
                    self.model.call_decoder_loop += 1

                # print(f"logits: {logits}")
                # import pdb; pdb.set_trace()
                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                if self.model.use_kvcache:
                    # print(f"logits dtype: {logits.dtype}")
                    # print(i)
                    # import pdb; pdb.set_trace()
                    self_attention_kcache = list(self_attention_kcache)
                    self_attention_vcache = list(self_attention_vcache)
                    tokens, completed = self.decoder.update(tokens, 
                                                            logits.float(), 
                                                            sum_logprobs, 
                                                            self_attention_kcache, 
                                                            self_attention_vcache, 
                                                        )
                else:
                    tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break

        finally:
            pass
        print(f'loop cost time: {time.time() - loop_start_time}')
        return tokens, sum_logprobs, no_speech_probs

    def _main_loop_untool(self, audio_features: Tensor, tokens: Tensor):
        print("{:=^80}".format(f" start main_loop {self.model.main_loop_cnt} "))
        # import pdb; pdb.set_trace()
        self.model.main_loop_cnt += 1
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        initial_tokens_length = len(self.initial_tokens)
        padding_num = self.padding_size
        print("{:%^60}".format(f" initial padding size: {padding_num} "))

        # padding_num = self.padding_size
        # padding_num_with_kvcache = self.padding_size

        # if self.options.fp16:
        #     attention_mask_firstly = torch.empty(padding_num, padding_num, dtype=torch.float16).fill_(-10000).triu_(1)
        #     attention_mask_with_kvcache = torch.empty(padding_num, padding_num, dtype=torch.float16).fill_(-10000).triu_(1)
        # else:
        #     attention_mask_firstly = torch.empty(padding_num, padding_num).fill_(-10000).triu_(1)
        #     attention_mask_with_kvcache = torch.empty(padding_num, padding_num).fill_(-10000).triu_(1)
    
        attention_mask_firstly = torch.empty(padding_num, padding_num).fill_(-10000).triu_(1)
        attention_mask_with_kvcache_max = torch.empty(448, 448).fill_(-10000).triu_(1)
        attention_mask_with_kvcache = attention_mask_with_kvcache_max[-padding_num:, -padding_num:]
        # attention_mask_with_kvcache = torch.empty(padding_num_with_kvcache, padding_num_with_kvcache).fill_(-10000).triu_(1)
        loop_start_time = time.time()
        if self.model.inference:
            tool = self.model.tool
        # import pdb; pdb.set_trace()

        try:
            for i in range(self.sample_len):
                # import pdb; pdb.set_trace()
                # print("{:=^80}".format(f" start {i} "))
                # print(f"tokens: {tokens}")
                # print(f"audio_features: {audio_features}")
                if i == 0 or not self.model.use_kvcache:
                    tokens_input = F.pad(tokens, (padding_num - tokens.shape[-1], 0, 0, 0), value=0)
                    positional_embedding_input = F.pad(self.model.positional_embedding[:i+initial_tokens_length], (0, 0, padding_num - initial_tokens_length - i, 0), value=0)
                    mask = F.pad(attention_mask_firstly[:tokens.shape[-1], :tokens.shape[-1]], (padding_num - tokens.shape[-1], 0, 0, 0), value=-10000)
                    mask = F.pad(mask, (0, 0, padding_num - tokens.shape[-1], 0), value=0)
                    mask = mask.reshape(1, 1, *mask.shape).repeat(n_batch, self.n_text_head, 1, 1).permute(0, 2, 1, 3).contiguous()
                else:
                    tokens_input = tokens[:, -1:]
                    offset = i + initial_tokens_length - 1
                    positional_embedding_input = self.model.positional_embedding[offset:offset+1]
                    mask = attention_mask_with_kvcache[offset:offset+1].flip(1)
                    mask = mask.reshape(1, 1, *mask.shape).repeat(n_batch, self.n_text_head, 1, 1).permute(0, 2, 1, 3).contiguous()
                # import pdb; pdb.set_trace()

                if i == 0:
                    start_time = time.time()

                    decoder_main_info = tool.model_info(self.model.decoder_main_handle)

                    tokens_input = tokens_input.numpy().astype(np.int32)
                    audio_features_dtype = data_type_map[decoder_main_info['input_dtypes'][1]]
                    audio_features = audio_features.numpy().astype(audio_features_dtype)
                    positional_embedding_input_dtype = data_type_map[decoder_main_info['input_dtypes'][2]]
                    positional_embedding_input = positional_embedding_input.numpy().astype(positional_embedding_input_dtype)
                    mask_dtype = data_type_map[decoder_main_info['input_dtypes'][3]]
                    mask = mask.numpy().astype(mask_dtype)

                    tokens_input = tokens_input if tokens_input.flags.c_contiguous else np.ascontiguousarray(tokens_input)
                    audio_features = audio_features if audio_features.flags.c_contiguous else np.ascontiguousarray(audio_features)
                    positional_embedding_input = positional_embedding_input if positional_embedding_input.flags.c_contiguous else np.ascontiguousarray(positional_embedding_input)
                    mask = mask if mask.flags.c_contiguous else np.ascontiguousarray(mask)

                    ########################################################################################################################
                    # Debug
                    ########################################################################################################################
                    # mask = mask.numpy().astype(mask_dtype).copy()
                    # tool1 = Tool()
                    # tool1.print_data_fp32(make_np2c(mask), mask.size, 445*12*448, 100, 1)
                    # onnx_input = [tokens_input, audio_features, positional_embedding_input, mask]
                    # onnx_input_names = ["tokens_input", "audio_features", "positional_embedding_input", "mask"]
                    # onnx_input_dict = {}
                    # for name, value in zip(onnx_input_names, onnx_input):
                    #     onnx_input_dict[name] = value
                    # np.savez("test_input.npz", **onnx_input_dict)
                    # import pdb; pdb.set_trace()

                    # decoder_bmodel_path = f"all_quant_decoder_main_with_kvcache_small_5beam_448pad_1684x_f16.bmodel"
                    # bmodel_dir = "./bmodel"
                    # decoder_bmodel_path = os.path.join(bmodel_dir, decoder_bmodel_path)
                    # inputs = np.load("./test_input.npz")
                    # tokens_input = inputs["tokens_input"]
                    # audio_features = inputs["audio_features"]
                    # positional_embedding_input = inputs["positional_embedding_input"]
                    # mask = inputs["mask"]
                    
                    # handle = tool1.bmhandle(0)
                    # bmrt1 = tool1.bmrt(handle)
                    # decoder_handle = tool1.create_model(decoder_bmodel_path.encode("utf-8"), bmrt1)
                    # runtime3       = tool1.create_un_runtime(handle)
                    # # print(runtime3)
                    # tool1.set_bmodel_info(runtime3, decoder_handle)
                    # # print("!!!!")
                    # tool1.set_stage(runtime3, 0)
                    # tool1.init_all_tensors(runtime3)
                    # tool1.malloc_device_address(runtime3)
                    # decoder_info = tool1.model_info(decoder_handle)
                    # tool1.copy_data_from_numpy(tool1.get_input_tensor(runtime3, 0), make_np2c(tokens_input), data_type[np.int32])
                    # tool1.copy_data_from_numpy(tool1.get_input_tensor(runtime3, 1), make_np2c(audio_features), data_type[np.float16])
                    # tool1.copy_data_from_numpy(tool1.get_input_tensor(runtime3, 2), make_np2c(positional_embedding_input), data_type[np.float16])
                    # tool1.copy_data_from_numpy(tool1.get_input_tensor(runtime3, 3), make_np2c(mask), data_type[np.float16])
                    # import pdb;pdb.set_trace()
                    # output_untool = np.empty(decoder_info[0]['output_shapes'][0], dtype=data_type_map[decoder_info['output_dtypes'][0]])
                    # print(output_untool)
                    # tool1.copy_data_from_numpy(tool1.get_output_tensor(runtime3, 0), make_np2c(output_untool), decoder_info['output_dtypes'][0])
                    # tool1.copy_input_data_to_device(runtime3)

                    # tool1.print_device_data(tool1.get_input_tensor(runtime3, 0), handle, 0, 10, True, make_np2c(tokens_input))
                    # tool1.print_device_data(tool1.get_input_tensor(runtime3, 1), handle, 0, 10, True, make_np2c(audio_features))
                    # tool1.print_device_data(tool1.get_input_tensor(runtime3, 2), handle, 0, 10, True, make_np2c(positional_embedding_input))
                    # tool1.print_device_data(tool1.get_input_tensor(runtime3, 3), handle, 0, 10, True, make_np2c(mask))
                    # # tool1.print_device_dat
                    # import pdb; pdb.set_trace()
                    # tool1.inference(runtime3)
                    # tool1.copy_output_data_to_host(runtime3)
                    # tool1.print_output_data(runtime3)
                    # # print(output_untool)
                    # # tool1.device_to_host(tool1.get_output_tensor(runtime3, 0))
                    # import pdb; pdb.set_trace()
                    # output_untool = torch.from_numpy(output_untool)
                    # print("{:=^80}".format(f" untool_infer "))
                    # print(output_untool)
                    # print("untool time: ", time.time() - start_time)

                    # tool1.destroy_un_runtime(runtime3)
                    # tool1.destroy_model(decoder_handle)
                    # tool1.free_bmrt(bmrt1)

                    ########################################################################################################################

                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime3, 0), make_np2c(tokens_input), data_type[np.int32])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime3, 1), make_np2c(audio_features), data_type[audio_features_dtype])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime3, 2), make_np2c(positional_embedding_input), data_type[positional_embedding_input_dtype])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime3, 3), make_np2c(mask), data_type[mask_dtype])
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime3, 0), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime3, 1), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime3, 2), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime3, 3), self.model.handle)

                    x = np.empty(decoder_main_info[0]['output_shapes'][0], dtype=data_type_map[decoder_main_info['output_dtypes'][0]])
                    tool.copy_data_from_numpy(tool.get_output_tensor(self.model.runtime3, 0), make_np2c(x), decoder_main_info['output_dtypes'][0])

                    tool.inference(self.model.runtime3)
                    tool.device_to_host(tool.get_output_tensor(self.model.runtime3, 0), self.model.handle)
                    self.model.time += time.time() - start_time

                    x_sot = x[:, padding_num - initial_tokens_length + self.sot_index:padding_num - initial_tokens_length + self.sot_index + 1].copy()
                    x_last = x[:, -1:].copy()

                    decoder_post_info = tool.model_info(self.model.decoder_post_handle)
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime4, 0), make_np2c(x_sot), data_type[x_sot.dtype])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime4, 1), make_np2c(x_last), data_type[x_last.dtype])
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime4, 0), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime4, 1), self.model.handle)
                    
                    logits = np.empty(decoder_post_info[0]['output_shapes'][0], dtype=data_type_map[decoder_post_info['output_dtypes'][0]])
                    no_speech_probs = np.empty(decoder_post_info[0]['output_shapes'][1], dtype=data_type_map[decoder_post_info['output_dtypes'][1]])
                    tool.copy_data_from_numpy(tool.get_output_tensor(self.model.runtime4, 0), make_np2c(logits), decoder_post_info['output_dtypes'][0])
                    tool.copy_data_from_numpy(tool.get_output_tensor(self.model.runtime4, 1), make_np2c(no_speech_probs), decoder_post_info['output_dtypes'][1])
                    # pdb.set_trace()
                    # tool.malloc_device_address(self.model.runtime4)
                    tool.inference(self.model.runtime4)
                    tool.copy_output_data_to_host(self.model.runtime4)
                    # tool.print_output_data(self.model.runtime4)
                    # import pdb; pdb.set_trace()

                    logits = torch.from_numpy(logits)
                    no_speech_probs = no_speech_probs.tolist()
                    
                    # import pdb; pdb.set_trace()
                    self.model.call_decoder_firstly += 1

                else:
                    start_time = time.time()

                    decoder_loop_info = tool.model_info(self.model.decoder_loop_handle)

                    tokens_input = tokens_input.numpy().astype(np.int32)
                    positional_embedding_input_dtype = data_type_map[decoder_loop_info['input_dtypes'][1]]
                    positional_embedding_input = positional_embedding_input.numpy().astype(positional_embedding_input_dtype)
                    mask_dtype = data_type_map[decoder_loop_info['input_dtypes'][2]]
                    mask = mask.numpy().astype(mask_dtype)

                    tokens_input = tokens_input if tokens_input.flags.contiguous else np.ascontiguousarray(tokens_input)
                    positional_embedding_input = positional_embedding_input if positional_embedding_input.flags.contiguous else np.ascontiguousarray(positional_embedding_input)
                    mask = mask if mask.flags.contiguous else np.ascontiguousarray(mask)
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime5, 0), make_np2c(tokens_input), data_type[np.int32])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime5, 1), make_np2c(positional_embedding_input), data_type[positional_embedding_input_dtype])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime5, 2), make_np2c(mask), data_type[mask_dtype])
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime5, 0), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime5, 1), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime5, 2), self.model.handle)
                    if i == 1:
                        tool.copy_data_from_numpy(tool.get_output_tensor(self.model.runtime5, 0), make_np2c(logits.numpy()), decoder_loop_info['output_dtypes'][0])
                    # logits = np.empty(decoder_loop_info[0]['output_shapes'][0], dtype=data_type_map[decoder_loop_info['output_dtypes'][0]])
                    # tool.copy_data_from_numpy(tool.get_output_tensor(self.model.runtime5, 0), make_np2c(logits), decoder_loop_info['output_dtypes'][0])

                    # import pdb; pdb.set_trace()
                    tool.malloc_device_address(self.model.runtime5)
                    tool.inference(self.model.runtime5)

                    tool.device_to_host(tool.get_output_tensor(self.model.runtime5, 0), self.model.handle)

                    # logits = torch.from_numpy(logits)
                    self.model.time += time.time() - start_time

                    # import pdb; pdb.set_trace()
                    self.model.call_decoder_loop += 1

                # print(f"logits: {logits}")
                # if (i == 16):
                #     import pdb; pdb.set_trace()
                # if (i == 5):
                #     exit()
                # import pdb; pdb.set_trace()
                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                if self.model.use_kvcache:
                    # print(f"logits dtype: {logits.dtype}")
                    # print(i)
                    # import pdb; pdb.set_trace()
                    tokens, completed = self.decoder.update(tokens, 
                                                            logits.float(), 
                                                            sum_logprobs, 
                                                        )
                else:
                    tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
                # if self.model.inference:
                #     cur_token_length += 1
        finally:
            pass
        print(f'loop cost time: {time.time() - loop_start_time}')
        return tokens, sum_logprobs, no_speech_probs
    
    def _main_loop_SGInfer(self, audio_features: Tensor, tokens: Tensor):
        print("{:=^80}".format(f" start main_loop {self.model.main_loop_cnt} "))
        # import pdb; pdb.set_trace()
        self.model.main_loop_cnt += 1
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        initial_tokens_length = len(self.initial_tokens)
        padding_num = self.padding_size
        print("{:%^60}".format(f" initial padding size: {padding_num} "))

        if self.model.inference:
            padding_id = 0
            for pad in self.model.paddings:
                if initial_tokens_length <= pad:
                    padding_num = pad
                    self.model.decoder_main_infer = self.model.decoder_main_infer_zoo[padding_num]
                    self.model.decoder_post_infer = self.model.decoder_post_infer_zoo[padding_num]
                    self.model.decoder_loop_infer = self.model.decoder_loop_infer_zoo[padding_num]
                    print("{:%^60}".format(f" switch padding size: {padding_num} "))
                    break
                padding_id += 1
            cur_token_length = initial_tokens_length
            cur_kvcache_length = padding_num

        # padding_num = self.padding_size
        # padding_num_with_kvcache = self.padding_size

        # if self.options.fp16:
        #     attention_mask_firstly = torch.empty(padding_num, padding_num, dtype=torch.float16).fill_(-10000).triu_(1)
        #     attention_mask_with_kvcache = torch.empty(padding_num, padding_num, dtype=torch.float16).fill_(-10000).triu_(1)
        # else:
        #     attention_mask_firstly = torch.empty(padding_num, padding_num).fill_(-10000).triu_(1)
        #     attention_mask_with_kvcache = torch.empty(padding_num, padding_num).fill_(-10000).triu_(1)
    
        attention_mask_firstly = torch.empty(padding_num, padding_num).fill_(-10000).triu_(1)
        attention_mask_with_kvcache_max = torch.empty(448, 448).fill_(-10000).triu_(1)
        attention_mask_with_kvcache = attention_mask_with_kvcache_max[-padding_num:, -padding_num:]
        # attention_mask_with_kvcache = torch.empty(padding_num_with_kvcache, padding_num_with_kvcache).fill_(-10000).triu_(1)
        loop_start_time = time.time()
        # import pdb; pdb.set_trace()

        try:
            for i in range(self.sample_len):
                # import pdb; pdb.set_trace()
                # print("{:=^80}".format(f" start {i} "))
                # print(f"tokens: {tokens}")
                # if self.model.log:
                #     if i == 3:
                #         exit()
                    # print("{:=^80}".format(f" start {i} "))
                if self.model.inference and cur_token_length > padding_num:
                    padding_id += 1
                    padding_num = self.model.paddings[padding_id]
                    attention_mask_with_kvcache = attention_mask_with_kvcache_max[-padding_num:, -padding_num:]
                    self.model.decoder_main_infer = self.model.decoder_main_infer_zoo[padding_num]
                    self.model.decoder_post_infer = self.model.decoder_post_infer_zoo[padding_num]
                    self.model.decoder_loop_infer = self.model.decoder_loop_infer_zoo[padding_num]
                # print(f"tokens: {tokens}")
                # print(f"audio_features: {audio_features}")
                if i == 0 or not self.model.use_kvcache:
                    tokens_input = F.pad(tokens, (padding_num - tokens.shape[-1], 0, 0, 0), value=0)
                    positional_embedding_input = F.pad(self.model.positional_embedding[:i+initial_tokens_length], (0, 0, padding_num - initial_tokens_length - i, 0), value=0)
                    mask = F.pad(attention_mask_firstly[:tokens.shape[-1], :tokens.shape[-1]], (padding_num - tokens.shape[-1], 0, 0, 0), value=-10000)
                    mask = F.pad(mask, (0, 0, padding_num - tokens.shape[-1], 0), value=0)
                    mask = mask.reshape(1, 1, *mask.shape).repeat(n_batch, self.n_text_head, 1, 1).permute(0, 2, 1, 3)
                else:
                    tokens_input = tokens[:, -1:]
                    offset = i + initial_tokens_length - 1
                    positional_embedding_input = self.model.positional_embedding[offset:offset+1]
                    mask = attention_mask_with_kvcache[offset:offset+1].flip(1)
                    mask = mask.reshape(1, 1, *mask.shape).repeat(n_batch, self.n_text_head, 1, 1).permute(0, 2, 1, 3)
                # import pdb; pdb.set_trace()

                if i == 0:
                    if self.model.inference:
                        # if self.model.quant:
                        #     bmodel_input = (tokens_input.numpy().astype(np.int32), audio_features.numpy().astype(np.float16), positional_embedding_input.numpy().astype(np.float16), mask.numpy(),)
                        # else:
                        #     bmodel_input = (tokens_input.numpy().astype(np.int32), audio_features.numpy(), positional_embedding_input.numpy(), mask.numpy(),)
                        # bmodel_input = (tokens_input.numpy().astype(np.int32), audio_features.numpy(), positional_embedding_input.numpy(), mask.numpy(),)
                        bmodel_input = (tokens_input.numpy().astype(np.int32), 
                                        audio_features.numpy().astype(nptype(self.model.decoder_main_infer.get_input_info()["audio_features"]["dtype"])), 
                                        positional_embedding_input.numpy().astype(nptype(self.model.decoder_main_infer.get_input_info()["positional_embedding_input"]["dtype"])), 
                                        mask.numpy().astype(nptype(self.model.decoder_main_infer.get_input_info()["mask"]["dtype"])),)
                        # import pdb; pdb.set_trace()
                        # if self.model.log:
                        #     print(f"[Log] decoder_main_infer")
                        #     print(f"[Log] input:")
                        #     print(f"[Log] bmodel_input: {bmodel_input}")
                        #     import pdb; pdb.set_trace()

                        start_time = time.time()
                        _ = self.model.decoder_main_infer.put(*bmodel_input)
                        _, results, _ = self.model.decoder_main_infer.get()
                        self.model.time += time.time() - start_time
                        x = results[0]

                        # if self.model.log:
                        #     print(f"[Log] output:")
                        #     for i in range(len(results)):
                        #         print(f"[Log] results[{i}]: {results[i]}")
                        #     import pdb; pdb.set_trace()
                        if self.model.use_kvcache:
                            self_attention_kcache = results[1:1+self.n_text_layer]
                            self_attention_vcache = results[1+self.n_text_layer:1+self.n_text_layer*2]
                            cross_attention_kcache = results[1+self.n_text_layer*2:1+self.n_text_layer*3]
                            cross_attention_vcache = results[1+self.n_text_layer*3:]
                        # import pdb; pdb.set_trace()
                        x_sot = x[:, padding_num - initial_tokens_length + self.sot_index:padding_num - initial_tokens_length + self.sot_index + 1]
                        x_last = x[:, -1:]
                        bmodel_input = (x_sot, x_last)
                        if self.model.log:
                            print(f"[Log] decoder_post_infer")
                            print(f"[Log] input:")
                            print(f"[Log] bmodel_input: {bmodel_input}")
                            # import pdb; pdb.set_trace()
                        _ = self.model.decoder_post_infer.put(*bmodel_input)
                        _, results, _ = self.model.decoder_post_infer.get()
                        self.model.time += time.time() - start_time
                        logits = torch.from_numpy(results[0])
                        no_speech_probs = results[1].tolist()
                        # if self.model.log:
                        #     print(f"[Log] output:")
                        #     for i in range(len(results)):
                        #         print(f"[Log] results[{i}]: {results[i]}")
                        #     import pdb; pdb.set_trace()
                        # import pdb; pdb.set_trace()
                    else:
                        # if self.model.log:
                        #     print(f"[Log] decoder_main_infer")
                        #     print(f"[Log] input:")
                        #     print(f"[Log] bmodel_input: {(tokens_input, audio_features, positional_embedding_input, mask,)}")
                        #     import pdb; pdb.set_trace()
                        if self.model.export_mode:
                            pass
                            input = (tokens_input, audio_features, positional_embedding_input, mask,)
                            input_names = ["tokens_input", "audio_features", "positional_embedding_input", "mask"]
                            output_names = ["x",]
                            input_dict = {}
                            for name, value in zip(input_names, input):
                                input_dict[name] = value
                            if self.model.use_kvcache:
                                model_name = f"decoder_main_with_kvcache_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad"
                                for idx in range(self.model.dims.n_text_layer):
                                    output_names.append(f"self_attn_kcache_in.{idx}")
                                for idx in range(self.model.dims.n_text_layer):
                                    output_names.append(f"self_attn_vcache_in.{idx}")
                                for idx in range(self.model.dims.n_text_layer):
                                    output_names.append(f"cross_attn_kcache_in.{idx}")
                                for idx in range(self.model.dims.n_text_layer):
                                    output_names.append(f"cross_attn_vcache_in.{idx}")
                            else:
                                model_name = f"decoder_main_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad"
                            np.savez(model_name + "_inputs.npz", **input_dict)
                            if self.model.export_mode == "onnx":
                                torch.onnx.export(
                                    self.inference_main_process,
                                    input,  # Pass the actual input data
                                    model_name + ".onnx",
                                    verbose=True,
                                    input_names=input_names,  # Provide input names
                                    output_names=output_names,  # Provide output names
                                    opset_version=15,  # ONNX opset version to use
                                )
                            elif self.model.export_mode == "pt":
                                torch.jit.trace(self.inference_main_process, input).save(model_name + ".pt")
                        if self.model.use_kvcache:
                            x, self_attention_kcache, self_attention_vcache, cross_attention_kcache, cross_attention_vcache = self.inference_main_process(
                                tokens_input, 
                                audio_features, 
                                positional_embedding_input, 
                                mask, 
                                )
                        else:
                            x = self.inference_main_process(
                                tokens_input, 
                                audio_features, 
                                positional_embedding_input, 
                                mask, 
                                )
                        # import pdb; pdb.set_trace()
                        x_sot = x[:, padding_num - initial_tokens_length + self.sot_index:padding_num - initial_tokens_length + self.sot_index + 1]
                        x_last = x[:, -1:]
                        if self.model.export_mode:
                            input = (x_sot, x_last)
                            input_names = ["x_sot", "x_last"]
                            output_names = ["logits", "no_speech_probs"]
                            input_dict = {}
                            for name, value in zip(input_names, input):
                                input_dict[name] = value
                            np.savez(f"decoder_post_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad_inputs.npz", **input_dict)
                            if self.model.export_mode == "onnx":
                                torch.onnx.export(
                                        self.inference_post_process,
                                        input,  # Pass the actual input data
                                        f"decoder_post_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad.onnx",
                                        verbose=True,
                                        input_names=input_names,  # Provide input names
                                        output_names=output_names,  # Provide output names
                                        opset_version=15,  # ONNX opset version to use
                                    )
                            elif self.model.export_mode == "pt":
                                torch.jit.trace(self.inference_post_process, input).save(f"decoder_post_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad.pt")
                        logits, no_speech_probs = self.inference_post_process(x_sot, x_last)
                        no_speech_probs = no_speech_probs.tolist()
                    # if self.model.use_kvcache:
                    #     for idx in range(len(self_attention_kcache)):
                    #         self_attention_kcache[idx] = self_attention_kcache[idx][:, 1:]
                    #         self_attention_vcache[idx] = self_attention_vcache[idx][:, 1:]

                    # import pdb; pdb.set_trace()
                    self.model.call_decoder_firstly += 1

                else:
                    if self.model.inference:
                        # print(self.model.decoder_loop_infer.get_input_info())
                        # import pdb; pdb.set_trace()
                        if self.model.use_kvcache:
                            if padding_num != cur_kvcache_length:
                                # import pdb; pdb.set_trace()
                                cache_padding = ((0, 0), (padding_num - cur_kvcache_length, 0), (0, 0))
                                for i in range(len(self_attention_kcache)):
                                    self_attention_kcache[i] = np.pad(self_attention_kcache[i], cache_padding, mode='constant', constant_values=0)
                                    self_attention_vcache[i] = np.pad(self_attention_vcache[i], cache_padding, mode='constant', constant_values=0)
                                cur_kvcache_length = padding_num

                            bmodel_input = (tokens_input.numpy().astype(np.int32), 
                                            positional_embedding_input.numpy().astype(nptype(self.model.decoder_loop_infer.get_input_info()["positional_embedding_input"]["dtype"])), 
                                            mask.numpy().astype(nptype(self.model.decoder_loop_infer.get_input_info()["mask"]["dtype"])), 
                                            *self_attention_kcache, *self_attention_vcache, *cross_attention_kcache, *cross_attention_vcache,)
                            # bmodel_input = (tokens_input.numpy().astype(np.int32), 
                            #                 positional_embedding_input.numpy(), 
                            #                 mask.numpy(), 
                            #                 *self_attention_kcache, *self_attention_vcache, *cross_attention_kcache, *cross_attention_vcache,)
                        else:
                            bmodel_input = (tokens_input.numpy().astype(np.int32), audio_features.numpy(), 
                                            positional_embedding_input.numpy().astype(nptype(self.model.decoder_loop_infer.get_input_info()["positional_embedding_input"]["dtype"])), 
                                            mask.numpy().astype(nptype(self.model.decoder_loop_infer.get_input_info()["mask"]["dtype"])),)
                        # if self.model.log:
                        #     print(f"[Log] decoder_loop_infer")
                        #     print(f"[Log] input:")
                        #     print(f"[Log] bmodel_input: {bmodel_input}")
                        #     import pdb; pdb.set_trace()
                        start_time = time.time()
                        _ = self.model.decoder_loop_infer.put(*bmodel_input)
                        _, results, _ = self.model.decoder_loop_infer.get()
                        self.model.time += time.time() - start_time
                        # if self.model.log:
                        #     import pdb; pdb.set_trace()
                        #     print(f"[Log] output:")
                        #     for i in range(len(results)):
                        #         print(f"[Log] results[{i}]: {results[i]}")
                        logits = torch.from_numpy(results[0])
                        if self.model.use_kvcache:
                            self_attention_kcache = results[1:1+self.n_text_layer]
                            self_attention_vcache = results[1+self.n_text_layer:1+self.n_text_layer*2]
                    else:
                        # if self.model.log:
                        #     print(f"[Log] decoder_loop_infer")
                        #     print(f"[Log] input:")
                        #     if self.model.use_kvcache:
                        #         print(f"[Log] bmodel_input: {(tokens_input, positional_embedding_input, mask, self_attention_kcache, self_attention_vcache, cross_attention_kcache, cross_attention_vcache,)}")
                        #     else:
                        #         print(f"[Log] bmodel_input: {(tokens_input, audio_features, positional_embedding_input, mask,)}")
                        #     import pdb; pdb.set_trace()
                        if self.model.export_mode:
                            if self.model.use_kvcache:
                                input = (tokens_input, positional_embedding_input, mask, self_attention_kcache, self_attention_vcache, cross_attention_kcache, cross_attention_vcache,)
                                npz_input = (tokens_input, positional_embedding_input, mask, *self_attention_kcache, *self_attention_vcache, *cross_attention_kcache, *cross_attention_vcache,)
                                input_names = ["tokens_input", "positional_embedding_input", "mask",]
                                output_names = ["logits",]
                                model_name = f"decoder_loop_with_kvcache_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad"
                                for idx in range(self.model.dims.n_text_layer):
                                    input_names.append(f"self_attn_kcache_in.{idx}")
                                for idx in range(self.model.dims.n_text_layer):
                                    input_names.append(f"self_attn_vcache_in.{idx}")
                                for idx in range(self.model.dims.n_text_layer):
                                    input_names.append(f"cross_attn_kcache_in.{idx}")
                                for idx in range(self.model.dims.n_text_layer):
                                    input_names.append(f"cross_attn_vcache_in.{idx}")

                                for idx in range(self.model.dims.n_text_layer):
                                    output_names.append(f"self_attn_kcache_out.{idx}")
                                for idx in range(self.model.dims.n_text_layer):
                                    output_names.append(f"self_attn_vcache_out.{idx}")
                            else:
                                input = (tokens_input, audio_features, positional_embedding_input, mask,)
                                npz_input = (tokens_input, audio_features, positional_embedding_input, mask,)
                                input_names = ["tokens_input", "audio_features", "positional_embedding_input", "mask",]
                                output_names = ["logits",]
                                model_name = f"decoder_loop_{self.model.model_name}_{self.model.beam_size}beam_{padding_num}pad"
                            input_dict = {}
                            for name, value in zip(input_names, npz_input):
                                input_dict[name] = value
                            np.savez(model_name + "_inputs.npz", **input_dict)
                            if self.model.export_mode == "onnx":
                                torch.onnx.export(
                                    self.inference_loop,
                                    input,  # Pass the actual input data
                                    model_name + ".onnx",
                                    verbose=True,
                                    input_names=input_names,  # Provide input names
                                    output_names=output_names,  # Provide output names
                                    opset_version=15,  # ONNX opset version to use
                                )
                            elif self.model.export_mode == "pt":
                                torch.jit.trace(self.inference_loop, input).save(model_name + ".pt")
                            exit()
                        else:
                            if self.model.use_kvcache:
                                logits, self_attention_kcache, self_attention_vcache = self.inference_loop(
                                    tokens_input, 
                                    positional_embedding_input, 
                                    mask, 
                                    self_attention_kcache, 
                                    self_attention_vcache, 
                                    cross_attention_kcache, 
                                    cross_attention_vcache
                                )
                            else:
                                logits = self.inference_loop(tokens_input, audio_features, positional_embedding_input, mask)
                    # if self.model.use_kvcache:
                    #     for idx in range(len(self_attention_kcache)):
                    #         self_attention_kcache[idx] = self_attention_kcache[idx][:, 1:]
                    #         self_attention_vcache[idx] = self_attention_vcache[idx][:, 1:]
                    
                    # import pdb; pdb.set_trace()
                    self.model.call_decoder_loop += 1

                # print(f"logits: {logits}")
                # if (i == 16):
                #     import pdb; pdb.set_trace()
                # if (i == 5):
                #     exit()
                # import pdb; pdb.set_trace()
                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                if self.model.use_kvcache:
                    # print(f"logits dtype: {logits.dtype}")
                    tokens, completed = self.decoder.update(tokens, 
                                                            logits.float(), 
                                                            sum_logprobs, 
                                                            self_attention_kcache, 
                                                            self_attention_vcache, 
                                                        )
                else:
                    tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
                if self.model.inference:
                    cur_token_length += 1
        finally:
            pass
        print(f'loop cost time: {time.time() - loop_start_time}')
        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, mel: Tensor) -> List[DecodingResult]:
        # import pdb; pdb.set_trace()
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        # print(f"mel: {mel}")
        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        # print(f"audio_features: {audio_features}")
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens) # encoder forward pass

        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(torch.int32)

        # call the main sampling loop
        if self.model.inference:
            if self.model.runtime == "untool":
                tokens, sum_logprobs, no_speech_probs = self._main_loop_untool(audio_features, tokens) # decoder forward pass
            elif self.model.runtime == "SGInfer":
                tokens, sum_logprobs, no_speech_probs = self._main_loop_SGInfer(audio_features, tokens)
            else:
                raise NotImplementedError(f"runtime {self.model.runtime} not implemented")
        else:
            tokens, sum_logprobs, no_speech_probs = self._main_loop_cpu(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        # import pdb;pdb.set_trace()
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


@torch.no_grad()
def decode(
    model: "Whisper",
    mel: Tensor,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    # import pdb; pdb.set_trace()
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)

    return result[0] if single else result

import os
import time
import base64
import gzip
from dataclasses import dataclass
from typing import Dict
import sophon.sail as sail
import numpy as np
import torch

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function
from .utils import fp16_cast, uint16_to_fp16


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class Whisper():
    def __init__(self, dims: ModelDimensions, args):
        super().__init__()
        self.dims = dims
        self.model_name = args["model_name"]
        self.encoder = None
        self.decoder = None
        self.encoder_infer = None
        self.logits_decoder_infer = None
        self.decoder_main_infer = None
        self.decoder_post_infer = None
        self.decoder_loop_infer = None
        self.bmodel_dir = args["bmodel_dir"]
        self.beam_size = args["beam_size"]
        self.padding_size = args["padding_size"]
        self.dev_id = args["dev_id"]

        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = all_heads.to_sparse()

        # get positional embedding from npz file
        positional_embedding_path = os.path.join(os.path.dirname(__file__), "assets", f"positional_embedding_{self.model_name}.npz")
        assert os.path.exists(positional_embedding_path), f"{positional_embedding_path} not found"
        self.positional_embedding = torch.tensor(np.load(positional_embedding_path)["positional_embedding"])

        ########################################
        ## Using sail to load BModel
        ########################################
        start_time = time.time()
        quant_str = "all_quant"
        encoder_bmodel_path           = f"{quant_str}_encoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
        logits_decoder_bmodel_path    = f"{quant_str}_logits_decoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
        decoder_main_bmodel_path      = f"{quant_str}_decoder_main_with_kvcache_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
        decoder_post_bmodel_path      = f"{quant_str}_decoder_post_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
        decoder_loop_bmodel_path      = f"{quant_str}_decoder_loop_with_kvcache_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
        kvcache_rearrange_bmodel_path = f"{quant_str}_kvcache_rearrange_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
        encoder_bmodel_path           = os.path.join(self.bmodel_dir, encoder_bmodel_path)
        logits_decoder_bmodel_path    = os.path.join(self.bmodel_dir, logits_decoder_bmodel_path)
        decoder_main_bmodel_path      = os.path.join(self.bmodel_dir, decoder_main_bmodel_path)
        decoder_post_bmodel_path      = os.path.join(self.bmodel_dir, decoder_post_bmodel_path)
        decoder_loop_bmodel_path      = os.path.join(self.bmodel_dir, decoder_loop_bmodel_path)
        kvcache_rearrange_bmodel_path = os.path.join(self.bmodel_dir, kvcache_rearrange_bmodel_path)
        assert os.path.exists(encoder_bmodel_path), f"{encoder_bmodel_path} not found"
        assert os.path.exists(logits_decoder_bmodel_path), f"{logits_decoder_bmodel_path} not found"
        assert os.path.exists(decoder_main_bmodel_path), f"{decoder_main_bmodel_path} not found"
        assert os.path.exists(decoder_post_bmodel_path), f"{decoder_post_bmodel_path} not found"
        assert os.path.exists(decoder_loop_bmodel_path), f"{decoder_loop_bmodel_path} not found"
        assert os.path.exists(kvcache_rearrange_bmodel_path), f"{kvcache_rearrange_bmodel_path} not found"

        # initial encoder engine
        self.encoder_engine = sail.Engine(encoder_bmodel_path, self.dev_id, sail.IOMode.SYSIO)
        self.encoder_engine_graph_name = self.encoder_engine.get_graph_names()[0]
        self.encoder_input_names = self.encoder_engine.get_input_names(self.encoder_engine_graph_name)
        self.encoder_input_tensors_map = self.encoder_engine.create_input_tensors_map(self.encoder_engine_graph_name)
        self.encoder_output_tensors_map = self.encoder_engine.create_output_tensors_map(self.encoder_engine_graph_name)

        # initial logits_decoder engine
        self.logits_decoder_engine = sail.Engine(logits_decoder_bmodel_path, self.dev_id, sail.IOMode.SYSIO)
        self.logits_decoder_graph_name = self.logits_decoder_engine.get_graph_names()[0]
        self.logits_decoder_input_names = self.logits_decoder_engine.get_input_names(self.logits_decoder_graph_name)
        self.logits_decoder_input_tensors_map = self.logits_decoder_engine.create_input_tensors_map(self.logits_decoder_graph_name)
        self.logits_decoder_output_tensors_map = self.logits_decoder_engine.create_output_tensors_map(self.logits_decoder_graph_name)

        # initial decoder_main engine
        self.decoder_main_engine = sail.Engine(decoder_main_bmodel_path, self.dev_id, sail.IOMode.DEVIO)
        self.decoder_main_graph_name = self.decoder_main_engine.get_graph_names()[0]
        self.decoder_main_input_names = self.decoder_main_engine.get_input_names(self.decoder_main_graph_name)
        self.decoder_main_output_names = self.decoder_main_engine.get_output_names(self.decoder_main_graph_name)
        self.decoder_main_input_tensors_map = self.decoder_main_engine.create_input_tensors_map(self.decoder_main_graph_name)
        self.decoder_main_output_tensors_map = self.decoder_main_engine.create_output_tensors_map(self.decoder_main_graph_name)

        # initial decoder_post engine
        self.decoder_post_engine = sail.Engine(decoder_post_bmodel_path, self.dev_id, sail.IOMode.SYSIO)
        self.decoder_post_graph_name = self.decoder_post_engine.get_graph_names()[0]
        self.decoder_post_input_names = self.decoder_post_engine.get_input_names(self.decoder_post_graph_name)
        self.decoder_post_output_names = self.decoder_post_engine.get_output_names(self.decoder_post_graph_name)
        self.decoder_post_input_tensors_map = self.decoder_post_engine.create_input_tensors_map(self.decoder_post_graph_name)
        self.decoder_post_output_tensors_map = self.decoder_post_engine.create_output_tensors_map(self.decoder_post_graph_name)

        # initial decoder_loop engine
        self.decoder_loop_engine = sail.Engine(decoder_loop_bmodel_path, self.dev_id, sail.IOMode.DEVIO)
        self.decoder_loop_graph_name = self.decoder_loop_engine.get_graph_names()[0]
        self.decoder_loop_input_names = self.decoder_loop_engine.get_input_names(self.decoder_loop_graph_name)
        self.decoder_loop_output_names = self.decoder_loop_engine.get_output_names(self.decoder_loop_graph_name)
        self.decoder_loop_input_tensors_map = self.decoder_loop_engine.create_input_tensors_map(self.decoder_loop_graph_name)
        self.decoder_loop_output_tensors_map = self.decoder_loop_engine.create_output_tensors_map(self.decoder_loop_graph_name)

        self.kvcache_rearrange_engine_list = []
        self.kvcache_rearrange_input_dict = {}
        self.kvcache_rearrange_output_dict = {}
        for i in range(self.dims.n_text_layer * 2):
            # initial kvcache_rearrange engine
            kvcache_rearrange_engine = sail.Engine(kvcache_rearrange_bmodel_path, self.dev_id, sail.IOMode.DEVIO)
            kvcache_rearrange_graph_name = kvcache_rearrange_engine.get_graph_names()[0]
            kvcache_rearrange_input_names = kvcache_rearrange_engine.get_input_names(kvcache_rearrange_graph_name)
            kvcache_rearrange_output_names = kvcache_rearrange_engine.get_output_names(kvcache_rearrange_graph_name)
            kvcache_rearrange_input_tensors_map = kvcache_rearrange_engine.create_input_tensors_map(kvcache_rearrange_graph_name)
            kvcache_rearrange_output_tensors_map = kvcache_rearrange_engine.create_output_tensors_map(kvcache_rearrange_graph_name)

            kvcache_rearrange_input_tensors_map[kvcache_rearrange_input_names[0]] = self.decoder_main_output_tensors_map[self.decoder_main_output_names[i + 1]]

            kvcache_rearrange_output_tensors_map[kvcache_rearrange_output_names[0]] = kvcache_rearrange_input_tensors_map[kvcache_rearrange_input_names[0]]

            self.kvcache_rearrange_engine_list.append(kvcache_rearrange_engine)
            self.kvcache_rearrange_input_dict[kvcache_rearrange_engine] = kvcache_rearrange_input_tensors_map
            self.kvcache_rearrange_output_dict[kvcache_rearrange_engine] = kvcache_rearrange_output_tensors_map

        for i in range(self.dims.n_text_layer * 4):
            self.decoder_loop_input_tensors_map[self.decoder_loop_input_names[i + 3]] = self.decoder_main_output_tensors_map[self.decoder_main_output_names[i + 1]]

        for i in range(self.dims.n_text_layer * 2):
            self.decoder_loop_output_tensors_map[self.decoder_loop_output_names [i + 1]] = self.decoder_loop_input_tensors_map[self.decoder_loop_input_names[i + 3]]

        kvcache_rearrange_engine_base = self.kvcache_rearrange_engine_list[0]
        kvcache_rearrange_engine_base_input_names = kvcache_rearrange_engine_base.get_input_names(kvcache_rearrange_engine_base.get_graph_names()[0])

        for i in range(self.dims.n_text_layer * 2 - 1):
            kvcache_rearrange_engine_next = self.kvcache_rearrange_engine_list[i + 1]
            kvcache_rearrange_engine_next_input_names = kvcache_rearrange_engine_next.get_input_names(kvcache_rearrange_engine_next.get_graph_names()[0])
            self.kvcache_rearrange_input_dict[kvcache_rearrange_engine_next][kvcache_rearrange_engine_next_input_names[1]] = self.kvcache_rearrange_input_dict[kvcache_rearrange_engine_base][kvcache_rearrange_engine_base_input_names[1]]

        self.init_time = time.time() - start_time
        print(f"\nTPU bmodel init time: {self.init_time}s")

        self.time = 0
        self.main_loop_cnt = 0
        self.call_encoder = 0
        self.call_logits_decoder= 0
        self.call_decoder_loop= 0
        self.call_decoder_firstly= 0
        self.call_decoder_with_kvcache = 0
        self.call_kvcache_rearrange = 0
        self.max_ctx = 0

    def init_cnt(self):
        self.main_loop_cnt = 0
        self.call_encoder = 0
        self.call_logits_decoder= 0
        self.call_decoder_loop= 0
        self.call_decoder_firstly= 0
        self.call_kvcache_rearrange = 0

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.alignment_heads = mask.to_sparse()

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        tokens = tokens.numpy().astype(np.int32)
        audio_features = audio_features.numpy().astype(np.float16)
        tokens = tokens if tokens.flags.c_contiguous else np.ascontiguousarray(tokens)
        audio_features = audio_features if audio_features.flags.c_contiguous else np.ascontiguousarray(audio_features)

        self.logits_decoder_input_tensors_map[self.logits_decoder_input_names[0]].update_data(tokens)
        unint16_audio_features = fp16_cast(audio_features)
        self.logits_decoder_input_tensors_map[self.logits_decoder_input_names[1]].update_data(unint16_audio_features);

        self.logits_decoder_engine.process(self.logits_decoder_graph_name, self.logits_decoder_input_tensors_map, self.logits_decoder_output_tensors_map)
        logits_tensor = list(self.logits_decoder_output_tensors_map.values())[0]
        logits = torch.from_numpy(uint16_to_fp16(logits_tensor.asnumpy()))

        self.call_logits_decoder += 1
        return logits

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        print("{:=^100}".format(" model.forward "))
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

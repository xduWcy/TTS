# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import torch
from tqdm import tqdm

from AR.modules.embedding_onnx import SinePositionalEmbedding
from AR.modules.embedding_onnx import TokenEmbedding
from AR.modules.transformer_onnx import LayerNorm
from AR.modules.transformer_onnx import TransformerEncoder
from AR.modules.transformer_onnx import TransformerEncoderLayer
from AR.models.utils import sample
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy



inf_tensor_value = torch.FloatTensor([-float("Inf")]).float()

def logits_to_probs(
    logits,
    previous_tokens = None,
    temperature: float = 1.0,
    top_k = None,
    top_p = None,
    repetition_penalty: float = 1.35,
):
    previous_tokens = previous_tokens.squeeze()
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        print(logits)
        print(top_k)
        v, _ = torch.topk(logits, top_k)
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, inf_tensor_value, logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync(
    probs_sort
):  # Does multinomial sampling without a cuda synchronization
    lambda_ = 1.0
    q = -torch.log(torch.rand_like(probs_sort)) / lambda_
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_p(
    logits,
    previous_tokens,
    **sampling_kwargs,
):
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


class OnnxEncoder(nn.Module):
    def __init__(self, ar_text_embedding, bert_proj, ar_text_position):
        super().__init__()
        self.ar_text_embedding = ar_text_embedding
        self.bert_proj = bert_proj
        self.ar_text_position = ar_text_position
    
    def forward(self, x, bert_feature):
        self._inputs = (x, bert_feature)
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        return self.ar_text_position(x)


class OnnxDecoder(nn.Module):
    def __init__(self, ar_audio_embedding, ar_audio_position, h, ar_predict_layer, num_layers):
        super().__init__()
        self.ar_audio_embedding = ar_audio_embedding
        self.ar_audio_position = ar_audio_position
        self.h = h
        self.ar_predict_layer = ar_predict_layer
        self.num_layers = num_layers

    def forward(self, x, y, k, v, prev_y_length, attn_mask):
        if not hasattr(self, "_inputs"):
            self._inputs = (x, y, k, v, prev_y_length, attn_mask)
        y_length = y.shape[1]
        y = self.ar_audio_embedding(y)
        y = y + self.ar_audio_position.pe[:, prev_y_length:prev_y_length+y_length] * self.ar_audio_position.alpha
        cache = {
            "all_stage": self.num_layers,
            "k": [k_chunk[0] for k_chunk in k.chunk(24)], #torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 1)),
            "v": [v_chunk[0] for v_chunk in v.chunk(24)], #torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 1)),
            "y_emb": None,
            "first_infer": 0,
            "stage": 0,
        }
        xy_pos = torch.cat([x,y], dim=1)
        xy_dec = self.h(xy_pos, mask=attn_mask, cache=cache)
        logits = self.ar_predict_layer(xy_dec)
        k = torch.stack(cache["k"])
        v = torch.stack(cache["v"])
        return logits, k, v


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=1):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = float(config["model"]["dropout"])
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )
        self.top_k = top_k
        self.early_stop_num = torch.LongTensor([-1])
    
    @classmethod
    def from_pretrained(cls, path):
        dict_s1 = torch.load(path, map_location="cpu")
        config = dict_s1["config"]
        model = cls(config)
        model.load_state_dict({k.lstrip("models."):v for k,v in dict_s1["weight"].items()})
        model.eval()
        model.init_onnx()

        return model

    def init_onnx(self):
        self.onnx_encoder = OnnxEncoder(self.ar_text_embedding, self.bert_proj, self.ar_text_position)
        self.onnx_decoder = OnnxDecoder(self.ar_audio_embedding, self.ar_audio_position, self.h, 
            self.ar_predict_layer, self.num_layers)

    def forward(self, prefix, prompt):
        prefix_len = prefix.shape[1]
        prompt_len = prompt.shape[1]
        pred_tokens = torch.empty((1, 0,), dtype=torch.int64)

        x_attn_mask = F.pad(
            torch.zeros((prefix_len, prefix_len), dtype=torch.bool, ),
            (0, prompt_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(prompt_len, prompt_len, dtype=torch.bool, ),
                diagonal=1,
            ),
            (prefix_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        cache_k = torch.empty((24, 0, 1, 512))
        cache_v = torch.empty((24, 0, 1, 512))
        prev_y_length = 0

        stop = False
        for idx in tqdm(range(1500)):
            logits, cache_k, cache_v = self.onnx_decoder(prefix, prompt, cache_k, cache_v, prev_y_length, xy_attn_mask)
            logits = logits[:, -1]
            if idx == 0:
                logits = logits[:, :-1]  ###刨除1024终止符号的概率
            samples = sample(logits[0], pred_tokens[0], top_k=self.top_k, top_p=1.0, repetition_penalty=1.35)[0]
            samples = samples.unsqueeze(0)
            # update stage
            prev_y_length += prompt.shape[1]
            pred_tokens = torch.cat([pred_tokens, samples], dim=1)
            xy_attn_mask = torch.zeros((1, prefix_len + prompt_len + idx + 1), dtype=torch.bool)
            prefix = torch.zeros((1, 0, 512))
            prompt = samples
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0,0] == self.EOS:
                stop = True
            if stop:
                break
        pred_tokens[0, -1] = 0
        return pred_tokens.unsqueeze(0)
    

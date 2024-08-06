# %%
import os
import LangSegment
import re
import soundfile
import shutil
import torch
from torch import nn
import librosa
import sys
now_dir = os.path.dirname(__file__)
sys.path.append(now_dir)
sys.path.append("D:/workspace/TTS")
sys.path.append("%s/GPT_SoVITS" % (now_dir))

from feature_extractor import cnhubert
from module.models_onnx import SynthesizerTrn, symbols
#from module.models import SynthesizerTrn
from text import chinese
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from transformers import AutoModelForMaskedLM, AutoTokenizer
from AR.models.t2s_model_onnx2 import Text2SemanticDecoder


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def get_spepc(hps, filename):
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class T2SEncoder(nn.Module):
    def __init__(self, t2s, vits):
        super().__init__()
        self.encoder = t2s.onnx_encoder
        self.vits = vits.vq_model
    
    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        self._inputs = (ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        codes = self.vits.extract_latent(ssl_content)
        prompt = codes[0]
        bert = torch.cat([ref_bert, text_bert], 1).unsqueeze(0)
        all_phoneme_ids = torch.cat([ref_seq, text_seq],dim=1)

        return self.encoder(all_phoneme_ids, bert), prompt


class VitsModel(nn.Module):
    def __init__(self, vits_path):
        super().__init__()
        dict_s2 = torch.load(vits_path,map_location="cpu")
        self.hps = dict_s2["config"]
        print(self.hps["model"])
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.hps.model.version = "v1"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        
    def forward(self, text_seq, pred_semantic, ref_audio):
        self._inputs = (text_seq, pred_semantic, ref_audio)
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False
        )
        return self.vq_model(pred_semantic, text_seq, refer)[0, 0]


class SSLModel(nn.Module):
    def __init__(self, ssl_model_path):
        super().__init__()
        cnhubert.cnhubert_base_path=ssl_model_path
        self.ssl = cnhubert.get_model()

    def forward(self, ref_audio_16k):
        self._inputs = (ref_audio_16k, )
        return self.ssl.model(ref_audio_16k)["last_hidden_state"].transpose(1, 2)
    

class BertModel(nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.bert_model.eval()
    
    def forward(self, inputs):
        self._inputs = inputs
        res = self.bert_model(*inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        return res

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i]
            res = self.forward(list(inputs.values()))
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        # if(is_half==True):phone_level_feature=phone_level_feature.half()
        return phone_level_feature.T


    def clean_text_inf(self, text, language, version):
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text


    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language=language.replace("all_","")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph)#.to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            )

        return bert

    def get_phones_and_bert(self, text,language,version):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_","")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.text_normalize(formattext)
                    return self.get_phones_and_bert(formattext,"zh",version)
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                    bert = self.get_bert_feature(norm_text, word2ph)
            elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.text_normalize(formattext)
                    return self.get_phones_and_bert(formattext,"yue",version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float32,
                )
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist=[]
            langlist=[]
            LangSegment.setfilters(["zh","ja","en","ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)
        phones = torch.LongTensor(phones).unsqueeze(0)
        return phones,bert,norm_text


class GptSoVits(nn.Module):
    def __init__(self, model_paths):
        super().__init__()
        self.model_paths = model_paths
        self.bert = BertModel(model_paths["bert_path"])
        self.ssl = SSLModel(model_paths["cnhubert_path"])
        self.vits = VitsModel(model_paths["vits_path"])
        self.gpt = Text2SemanticDecoder.from_pretrained(model_paths["gpt_path"], )
        self.encoder = T2SEncoder(self.gpt, self.vits)
        self.bert.eval()
        self.ssl.eval()
        self.vits.eval()
        self.gpt.eval()
        self.encoder.eval()

    @torch.no_grad
    def forward(self, text, ref_wav_path, prompt_text):
        self.eval()
        audio, sr = librosa.load(ref_wav_path)
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio_32k = librosa.resample(audio, orig_sr=sr, target_sr=32000)
        audio_16k = torch.from_numpy(audio_16k).unsqueeze(0)
        audio_32k = torch.from_numpy(audio_32k).unsqueeze(0)
        
        ssl_content = self.ssl.forward(audio_16k)
        ref_seq, ref_bert, _ = self.bert.get_phones_and_bert(prompt_text, "zh", "v1")
        text_seq, text_bert, _ = self.bert.get_phones_and_bert(text, "zh", "v1")
        prefix, prompt = self.encoder.forward(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        pred_tokens = self.gpt.forward(prefix, prompt)
        audio = self.vits.forward(text_seq, pred_tokens, audio_32k)

        return audio

    def export(self, base_path):
        torch.onnx.export(
            self.ssl,
            self.ssl._inputs,
            f"{base_path}/cnhubert.onnx",
            input_names=["ref_audio_16k"],
            output_names=["hubert_feature"],
            dynamic_axes={
                "ref_audio_16k":{1:"audio_length"}
            },
            opset_version=16
        )
        shutil.copyfile(f"{self.model_paths['bert_path']}/config.json", f"{base_path}/config.json")
        shutil.copyfile(f"{self.model_paths['bert_path']}/tokenizer.json", f"{base_path}/tokenizer.json")
        torch.onnx.export(
            self.bert,
            self.bert._inputs,
            f"{base_path}/cnroberta.onnx",
            input_names=["input_ids", "token_type_ids", "attention_mask"],
            output_names=["bert_feature"],
            dynamic_axes={
                "input_ids":{1:"seq_length"},
                "token_type_ids":{1:"seq_length"},
                "attention_mask":{1:"seq_length"},
            },
            opset_version=16
        )
        torch.onnx.export(
            self.encoder,
            self.encoder._inputs,
            f"{base_path}/t2s_encoder.onnx",
            input_names=["ref_seq", "text_seq", "ref_bert", "text_bert", "ssl_content"],
            output_names=["prefix", "prompt"],
            dynamic_axes={
                "ref_seq":{1:"ref_length"},
                "text_seq":{1:"text_length"},
                "ref_bert":{1:"ref_length"},
                "text_bert":{1:"text_length"},
                "ssl_content":{2:"ssl_length"}
            },
            opset_version=16
        )

        torch.onnx.export(
            self.gpt.onnx_decoder,
            self.gpt.onnx_decoder._inputs,
            f"{base_path}/t2s_decoder.onnx",
            input_names=["prefix", "tokens", "cache_k", "cache_v", "prev_seq_length", "attn_mask"],
            output_names=["logits", "out_k", "out_v"],
            dynamic_axes={
                "prefix":{1:"prefix_length"},
                "tokens":{1:"prompt_length"},
                "cache_k":{1:"cache_length"},
                "cache_v":{1:"cache_length"},
                "prev_seq_length":{},
                "attn_mask":{0:"attn_mask_h", 1:"attn_mask_w"}
            },
            opset_version=16
        )

        torch.onnx.export(
            self.vits,
            self.vits._inputs,
            f"{base_path}/vits.onnx",
            input_names=["text_seq", "pred_semantic", "ref_audio"],
            output_names=["audio"],
            dynamic_axes={
                "text_seq":{1:"text_length"},
                "pred_semantic":{2:"pred_token_length"},
                "ref_audio":{1:"audio_length"},
            },
            opset_version=17
        )


# %%
if __name__ == "__main__":
    OUTPUT_PATH = "./cache/onnx/gsv3"

    text = "获取当前文件所在的位置。"
    lang = "zh"
    ref_wav_path = "./Service150\\APY210531004\\data\\G00058\\Wave\\058369.wav"
    prompt_text = "那您现在需要办理吗，想办吗？"
    ref_lang = "zh"

    PATH = {
        "cnhubert_path":"./cache/GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_path":"./cache/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "vits_path":"./cache/GPT_SoVITS/pretrained_models/s2G488k.pth",
        "gpt_path":"./cache/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    }

    model = GptSoVits(PATH)
    audio1 = model.forward(
        text,
        ref_wav_path,
        prompt_text
    )
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    model.export(OUTPUT_PATH)
    soundfile.write("./debug_wav.wav", audio1, 32000)

   
# %%

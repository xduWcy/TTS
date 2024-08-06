# %%
import os
import LangSegment
import re
import soundfile
import builtins
import shutil
import torch
from torch import nn
import torch.nn.functional as F
import librosa
import sys
import onnxruntime
import numpy as np
from tqdm import tqdm
now_dir = os.path.dirname(__file__)
sys.path.append(now_dir)
sys.path.append("D:/workspace/TTS")
sys.path.append("%s/GPT_SoVITS" % (now_dir))
from text import chinese
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from transformers import  AutoTokenizer
from AR.models.utils import sample

# environment variables required by dynamic quant on NPU
PWD=".\\ryzen-ai\\transformers"
os.environ["THIRD_PARTY"] = f"{PWD}\\third_party"
os.environ["TVM_LIBRARY_PATH"] = f"os.environ['TVM_LIBRARY_PATH']\\lib;{os.environ['THIRD_PARTY']}\\bin"

os.environ["PATH"]=f"{os.environ['PATH']};{os.environ['TVM_LIBRARY_PATH']};{PWD}\\ops\\cpp\\;{os.environ['THIRD_PARTY']}" 
os.environ["PYTORCH_AIE_PATH"] = f"{PWD}"

os.environ["PYTORCH_AIE_PATH"] = f"{PWD}"
os.environ["XRT_PATH"] = f"os.environ['THIRD_PARTY']\\xrt"
os.environ["DEVICE"] = f"phx"
os.environ["XLNX_VART_FIRMWARE"] = f"{PWD}\\xclbin\\phx"

os.environ["TVM_MODULE_PATH"] = f"{PWD}\\dll\\phx\\qlinear\\libGemmQnnAie_1x2048_2048x2048.dll,{PWD}\\dll\\phx\\qlinear\\libGemmQnnAie_8x2048_2048x2048.dll,"
os.environ["TVM_GEMM_M"] = f"1,8,"
os.environ["TVM_DLL_NUM"] = f"2"
class ONNXModel():
    def __init__(self, path, device="cpu", **npu_args):
        if device == "npu":
            if npu_args["clean_cache"]:

                print("=============   clean cache  =============\n")
                if os.path.exists(f"{npu_args['cache_path']}/{npu_args['model_name']}"):
                    shutil.rmtree(f"{npu_args['cache_path']}/{npu_args['model_name']}")
            session_options = onnxruntime.SessionOptions()
            session_options.enable_profiling = False
            builtins.impl = "v0"
            builtins.quant_mode = "w8a8"
            self.sess = onnxruntime.InferenceSession(
                path,
                providers=['VitisAIExecutionProvider'] ,
                provider_options=[{"config_file":npu_args["config_path"],
                                "cacheDir": npu_args["cache_path"],
                                'cacheKey': npu_args["model_name"]}]
            )
        elif device == "cpu":
            self.sess = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])

    def __call__(self, inputs):
        for keys in inputs:
            if isinstance(inputs[keys], torch.Tensor):
                inputs[keys] = inputs[keys].numpy()
        outputs = self.sess.run(None, inputs)
        return [torch.from_numpy(x) for x in outputs]

class T2SEncoder():
    def __init__(self, base_path):
        self.model = ONNXModel(f"{base_path}/t2s_encoder.onnx")
    
    def __call__(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        inputs = {
            "ref_seq":ref_seq.to(torch.int64),
            "text_seq": text_seq.to(torch.int64),
            "ref_bert":ref_bert,
            "text_bert":text_bert,
            "ssl_content":ssl_content
        }
        return self.model(inputs)

class VitsModel():
    def __init__(self, base_path):
        self.model = ONNXModel(f"{base_path}/vits.onnx")

    def __call__(self, text_seq, pred_semantic, ref_audio):
        inputs = {
            "text_seq":text_seq.to(torch.int64),
            "pred_semantic": pred_semantic,
            "ref_audio":ref_audio,
        }
        return self.model(inputs)[0]


class SSLModel():
    def __init__(self, base_path):
        self.model = ONNXModel(f"{base_path}/cnhubert.onnx")

    def __call__(self, ref_audio_16k):
        inputs = {
            "ref_audio_16k":ref_audio_16k,
        }
        return self.model(inputs)[0]

class BertModel():
    def __init__(self, base_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.model = ONNXModel(f"{base_path}/cnroberta.onnx")
    
    def forward(self, inputs):
        inputs = {
            "input_ids":inputs[0].astype(np.int64),
            "token_type_ids":inputs[1].astype(np.int64),
            "attention_mask":inputs[2].astype(np.int64),
        }
        return self.model(inputs)[0]

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="np")
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

class T2sDecoder():
    def __init__(self, base_path, npu_args):
        self.top_k=1
        self.EOS=1024
        self.model = ONNXModel(f"{base_path}/t2s_decoder_quantized.onnx", **npu_args)

    def forward(self, prefix, prompt, cache_k, cache_v, prev_y_length, xy_attn_mask):
        inputs = {
            "prefix":prefix,
            "tokens": prompt.to(torch.int64),
            "cache_k":cache_k,
            "cache_v":cache_v,
            "prev_seq_length":np.array(prev_y_length).astype(np.int64),
            "attn_mask":xy_attn_mask
        }
        return self.model(inputs)
    
    def generate(self, prefix, prompt):
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
            logits, cache_k, cache_v = self.forward(prefix, prompt, cache_k, cache_v, prev_y_length, xy_attn_mask)
            logits = logits[:, -1]
            if idx == 0:
                logits = logits[:, :-1]  ###刨除1024终止符号的概率
            samples = sample(logits[0], pred_tokens[0], top_k=self.top_k, top_p=0.8, repetition_penalty=1.35)[0]
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
        return pred_tokens.unsqueeze(1)




class GPTSoVITSONNX():
    def __init__(self, base_path, **npu_args):
        super().__init__()
        self.bert = BertModel(base_path)
        self.ssl = SSLModel(base_path)
        self.vits = VitsModel(base_path)
        self.gpt = T2sDecoder(base_path, npu_args)
        self.encoder = T2SEncoder(base_path)
        print("done loading onnx model")

    def tts(self, ref_wav_path, prompt_text, text, speed=1.0, text_language="zh"):
        audio, sr = librosa.load(ref_wav_path)
        if speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed)
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio_32k = librosa.resample(audio, orig_sr=sr, target_sr=32000)
        audio_16k = torch.from_numpy(audio_16k).unsqueeze(0)
        audio_32k = torch.from_numpy(audio_32k).unsqueeze(0)
        
        ssl_content = self.ssl(audio_16k)
        ref_seq, ref_bert, _ = self.bert.get_phones_and_bert(prompt_text, "zh", "v1")
        text_seq, text_bert, _ = self.bert.get_phones_and_bert(text, "zh", "v1")
        prefix, prompt = self.encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        pred_tokens = self.gpt.generate(prefix, prompt)
        print(pred_tokens.shape)
        audio = self.vits(text_seq, pred_tokens, audio_32k)

        return audio

if __name__ == "__main__":
    OUTPUT_PATH = "./cache/onnx/gsv_2"

    text = "Most of the Olympic events are being held in the city of Paris and its metropolitan region, "
    lang = "zh"
    ref_wav_path = "./Service150\\APY210531004\\data\\G00058\\Wave\\058369.wav"
    prompt_text = "那您现在需要办理吗，想办吗？"

    model = GPTSoVITSONNX(OUTPUT_PATH)
    audio = model.tts(ref_wav_path, prompt_text, text)

    from IPython.display import Audio
    Audio(audio, rate=32000)

# %%
    Audio(audio, rate=32000)

# %%

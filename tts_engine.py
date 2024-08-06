from gpt_sovits.onnx_infer import GPTSoVITSONNX
from Service150 import UtterDs
from time import time,sleep
from io import BytesIO
from queue import Queue
from threading import Thread
import pygame
import os
import numpy as np

DIR_NAME = os.path.dirname(__file__)


PWD="D:\\workspace\\RyzenAI-SW\\example\\transformers"
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





class TTSEngine():

    def __init__(self, ref_id=10, device="cpu"):
        self.ref_ds = UtterDs(DIR_NAME + "/Service150")
        self.ref_wav_path, self.prompt_text = self.ref_ds[ref_id]
        if device == "cpu":
            npu_kwargs = {
                "max_sec":20,
                "device":"cpu",
            }
        elif device == "npu":
            npu_kwargs = {
                "max_sec":20,
                "device":"npu",
                "cache_path":f"{DIR_NAME}/tmp",
                "config_path":f"{DIR_NAME}/ryzen-ai/vaip_config.json",
                "clean_cache":True,
                "model_name":"decoder"
            }
        else:
            raise NotImplementedError("Not supported device")
        self.gpt_sovits = GPTSoVITSONNX(DIR_NAME + "/cache/onnx/gsv3", **npu_kwargs)

    def tts(self, text, speed=1.0, language="zh"):
        #t=time()
        all_audio = []
        for text_piece in text.split("\n"):
            if text_piece == "":
                continue
            audio = self.gpt_sovits.tts(self.ref_wav_path, self.prompt_text, text_piece, speed=speed, text_language=language)
            all_audio.append(audio)
            all_audio.append(np.zeros((2000,)))
        #print(f"tts cost P{time()-t:.3f}s")
        return np.concatenate(all_audio)


class Player():
    def __init__(self,):
        pygame.mixer.init(frequency=16000, size=32)

    def play(self, audio):
        print("开始播放")
        self.sound = pygame.mixer.Sound(audio.tobytes())#audio_bytes.getbuffer())
        self.sound.play()
        #print(audio.dtype)
        #soundfile.write("out.wav", audio, samplerate=32000)
    
    def is_complete(self):
        a = hasattr(self, "sound")
        b = pygame.mixer.get_busy()
        print(f"a {a}  b {b}")
        if hasattr(self, "sound") and pygame.mixer.get_busy():
            return False
        else:
            return True
    
    def pause(self):
        if hasattr(self, "sound"):
            pygame.mixer.pause()
    
    def resume(self):
        if hasattr(self, "sound"):
            pygame.mixer.unpause()
    
    def set_volumn(self, volumn):
        if hasattr(self, "sound"):
            self.sound.set_volume(volumn)
    
    def stop(self):
        if hasattr(self, "sound"):
            pygame.mixer.stop()

    def fadeout(self):
        if hasattr(self, "sound"):
            pygame.mixer.fadeout(1)
    

# class PlayerQueue():
#     def __init__(self, tasks):
#         self.tasks  = tasks
#         self.audio_queue = Queue()
#         self.engine = TTSEngine()
#         self.player = Player()
#         self.thread = Thread(target=self.compute_voice)
#         self.thread.setDaemon(True)

#     def main_loop(self):
#         self.thread.start()
#         sleep(10)
#         while True:
#             if not self.audio_queue.empty() and self.player.is_complete():
#                 entry = self.audio_queue.get()
#                 print(f"[Playing] : {entry['text']}")
#                 self.player.play(entry["audio"])
#             sleep(0.05)

#     def compute_voice(self):
#         for text in self.tasks:
#             audio = self.engine.tts(text)
#             self.audio_queue.put({"audio":audio, "text":text})


# %%

# if __name__ == "__main__":
#     texts = ["阿勒泰市在古代是中国少数民族的牧居地。",
#          "据史书记载，秦代牧居在此的部落是由今甘肃省河西走廊一带迁来的塞种人。",
#          "从西汉开始，历代中央政府均在此设行政管理机构。",
#          "1953年11月20日，改承化县为阿泰县，1984年11月17日，撤销阿勒泰县，改置阿勒泰市",
#          "“阿勒泰”是突厥语，意为“金山”，因山中蕴藏黄金而得名，有“阿尔泰山七十二条沟，沟沟有黄金”之说。"]
#     main = PlayerQueue(texts)
#     main.main_loop()
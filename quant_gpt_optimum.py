import os
import builtins
import numpy as np
import h5py
import shutil
import onnx
import torch

import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, shape_inference
from optimum.onnxruntime import ORTQuantizer, AutoQuantizationConfig
from optimum.onnxruntime.configuration import AutoCalibrationConfig, CalibrationConfig, CalibrationMethod

from pathlib import Path

base_path = "./cache/onnx/gsv_2"
full_prec_model = "t2s_decoder.onnx"
inter_model = "t2s_decoder.onnx"
model_target = "t2s_decoder_quantized.onnx"

REQUANT = True
CLEAN_CACHE =True


# Define the environment variable for NPU


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



# Load a calibration dataset
class SubsetModelInputs(torch.utils.data.Dataset):
    def __init__(self, num=450):
        super().__init__()
        self.num = min(num, 450)
        self.data = h5py.File("./cache/quant_ds.hdf5")
    def __len__(self):
        return self.num
    def __getitem__(self, idx):
            if idx >= self.num:
                raise StopIteration
            prefix = self.data[str(idx)]["prefix"][:]
            tokens = self.data[str(idx)]["tokens"][:]
            start_length = self.data[str(idx)].attrs["start_length"]
            k = np.empty((24, 0, 1, 512), dtype=np.float32)
            v = np.empty((24, 0, 1, 512), dtype=np.float32)
            prefix_len = prefix.shape[1]
            tokens_len = tokens.shape[1]
            x_attn_mask = np.zeros((prefix_len, prefix_len), dtype=np.bool_)
            x_attn_mask_pad = np.concatenate((x_attn_mask, np.ones((prefix_len, tokens_len), dtype=np.bool_)), axis=1)
            y_attn_mask = np.concatenate((  ###yy的右上1扩展到左边xy的0,(y,x+y)
                np.zeros((tokens_len, prefix_len), dtype=np.bool_),
                np.triu(np.ones((tokens_len, tokens_len), dtype=np.bool_), k=1),
                ), axis=1
            )
            xy_attn_mask = np.concatenate([x_attn_mask_pad, y_attn_mask], axis=0)
            output =  {"prefix":prefix,
                "tokens":tokens,
                "cache_k":k,
                "cache_v":v,
                "attn_mask":xy_attn_mask,
                "prev_seq_length":np.array(0, dtype=np.int64)
            }
            return output

    def __del__(self):
        self.data.close()


if REQUANT or not os.path.exists(f"{base_path}/{model_target}"):
    print("quant_prepare")
    shape_inference.quant_pre_process(
            input_model_path=f"{base_path}/{full_prec_model}",
            output_model_path=f"{base_path}/{inter_model}",
            skip_optimization = False,
            skip_onnx_shape = False,
            skip_symbolic_shape = True,
            auto_merge = False,
            int_max = 2**31 - 1,
            guess_output_rank = False,
            verbose = 3,
            save_as_external_data = False,
            all_tensors_to_one_file = False,)
    full_prec_model = inter_model

    print("quant with optimum")
    onnx_model = onnx.load(f"{base_path}/{full_prec_model}")
    nodes = onnx_model.graph.node
    quant_exclude_nodes = []
    for n in nodes:
        # Unfortunately, quantitizing the QKV project layer raise unexplanable error.
        if "self_attn/MatMul" in n.name:
            quant_exclude_nodes.append(n.name)
    print(f"{quant_exclude_nodes=}")
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False, use_symmetric_activations=True, operators_to_quantize=["MatMul", "GEMM"],nodes_to_exclude=quant_exclude_nodes)
    quantizer = ORTQuantizer.from_pretrained(base_path, full_prec_model)
    
    # Using static quant reduce accuracy, so static quant is not adopted.

    #calibration_dataset = SubsetModelInputs()
    #calibration_config = CalibrationConfig(
    #    dataset_name="decoder_inputs",
    #    dataset_config_name="test1",
    #    dataset_split="example",
    #    dataset_num_samples=450,
    #    method=CalibrationMethod.MinMax,
    #    moving_average=False,
    #    averaging_constant=0.01,
    #)
    #ranges = quantizer.fit(dataset=calibration_dataset, calibration_config=calibration_config, operators_to_quantize=["MatMul"])
    quantizer.quantize(save_dir=base_path, quantization_config=dqconfig,)# calibration_tensors_range=ranges)

    #quit()


if __name__ == "__main__":

    sample_dataset = SubsetModelInputs()
    OUTPUT_PATH = "./cache/tmp"

    if CLEAN_CACHE and os.path.exists(f"{OUTPUT_PATH}/gpt"):
        print("=== Clean cache ===")
        shutil.rmtree(f"{OUTPUT_PATH}/gpt")


    provider_options = [{
                    'config_file': "./ryzen-ai/vaip_config.json",#"D:/workspace/TTS/ryzen-ai/vaip_config.json",
                    "cacheDir": OUTPUT_PATH,
                    'cacheKey': "gpt"}]

    session_options = ort.SessionOptions()
    session_options.enable_profiling = False
    builtins.impl = "v0"
    builtins.quant_mode = "w8a8"
    session = ort.InferenceSession(f"{base_path}/{model_target}", 
                                providers=['VitisAIExecutionProvider'],
                                sess_options=session_options,                               
                                provider_options=provider_options)

    cpu_session = ort.InferenceSession(f"{base_path}/{model_target}", providers=["CPUExecutionProvider"])
    cpu_full_session = ort.InferenceSession(f"{base_path}/{full_prec_model}", providers=["CPUExecutionProvider"])
    inputs = sample_dataset[0]

    outputs = session.run(None, inputs)
    outputs_cpu = cpu_session.run(None, inputs)
    outputs_full = cpu_full_session.run(None, inputs)

    tokens1 = outputs[0][0].argmax(axis=1)
    tokens2 = outputs_cpu[0][0].argmax(axis=1)
    tokens3 = outputs_full[0][0].argmax(axis=1)
    
    print(outputs[0].shape)
    print(f"NPU tokens: {tokens1[:20]}")
    print(f"CPU tokens: {tokens2[:20]}")
    print(f"CPU Full tokens: {tokens3[:20]}")

    print(f"NPU logits: {outputs[0][0][-1,:20]}")
    print(f"CPU logits: {outputs_cpu[0][0][-1,:20]}")
    print(f"CPU Full logits: {outputs_full[0][0][-1,:20]}")

    print(f"quant on npu, acc = {(tokens3 == tokens1).mean()}")
    print(f"quant on cpu, acc = {(tokens3 == tokens2).mean()}")


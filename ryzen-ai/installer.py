import os
import shutil
import logging
import platform
import site
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def copy(source_file,dest_path):
    if os.path.exists(source_file):
      logging.info(f"copying {source_file} to {dest_path}")
      shutil.copy(source_file, dest_path)
    else:
      raise FileNotFoundError(f"{source_file} does not exist.")

if platform.system() == "Windows":
    xrt_source_files = [
        "C:\\Windows\\System32\\AMD\\xrt_core.dll",
        "C:\\Windows\\System32\\AMD\\xrt_coreutil.dll",
        "C:\\Windows\\System32\\AMD\\amd_xrt_core.dll",
        "C:\\Windows\\System32\\AMD\\xdp_ml_timeline_plugin.dll",
        "C:\\Windows\\System32\\AMD\\xdp_core.dll"
    ]
    ort_source_files = {
        "onnxruntime.dll" : [os.path.join(os.getcwd(),"voe-0.1.0-cp39-cp39-win_amd64"), os.path.join(os.getcwd(),"..","onnxruntime", "bin")]
    }
    
    dest_found = False
    site_package_paths = site.getsitepackages()
    for site_package_path in site_package_paths:
        dest_path = os.path.join(site_package_path + "\\onnxruntime\\capi")
        if os.path.exists(dest_path):
            dest_found = True
            for source_file in xrt_source_files:
                try:
                    copy(source_file, dest_path)
                except Exception as e:
                    logging.fatal(str(e))
            for source_file in ort_source_files:
                ort_found = False
                for path in ort_source_files[source_file]:
                    if os.path.exists(os.path.join(path, source_file)):
                        ort_found = True
                        copy(os.path.join(path, source_file), dest_path)
                if not ort_found:
                    logging.fatal(source_file + " was not found")
                

    if not dest_found:
        raise FileNotFoundError("Installer failed! Please install onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl, \
voe-0.1.0-cp39-cp39-win_amd64.whl and try again.")

else:
    logging.info("This script is intended to run on Windows only.")

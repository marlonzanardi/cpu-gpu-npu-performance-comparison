import os
import subprocess


def c_locale_env():
    env = os.environ.copy()
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    return env


def hidden_process_flags():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def load_onnxruntime():
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    return ort

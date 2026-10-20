# Kaggle notebook: run the LLM RPS experiments on free GPUs.
# Setup: New Notebook -> Settings -> Accelerator "GPU T4 x2", Internet "On".
# Paste each "# %%" block into its own cell, or upload this file and run it with %run.

# %% 1. Install and start an OpenAI-compatible vLLM server (T4 has no bf16, so fp16; 2 GPUs -> tensor parallel)
import os
import subprocess
import time
import urllib.request

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REPO = "https://github.com/<your-username>/<repo>.git"  # TODO: your repo URL

subprocess.run("pip install -q vllm", shell=True, check=True)
server = subprocess.Popen(
    f"python -m vllm.entrypoints.openai.api_server --model {MODEL} --dtype half "
    "--tensor-parallel-size 2 --max-model-len 4096 --port 8000 > /kaggle/working/vllm.log 2>&1",
    shell=True,
)
for _ in range(120):  # model download + load takes a few minutes
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
        print("server ready")
        break
    except Exception:
        time.sleep(10)
else:
    raise RuntimeError("vLLM did not start; see /kaggle/working/vllm.log")

# %% 2. Get the code
subprocess.run(f"git clone {REPO} /kaggle/working/netinf && pip install -q -e /kaggle/working/netinf",
               shell=True, check=True)
os.chdir("/kaggle/working/netinf")
os.environ.update(NETINF_BASE_URL="http://localhost:8000/v1", NETINF_MODEL=MODEL)

# %% 3. Pilot: 2 short runs per condition. Check parse_fail is ~0 and chat looks sane before the full run.
for cond in ["independent", "channel", "channel_shared", "instructed"]:
    subprocess.run(f"python experiments/llm_rps.py --condition {cond} --runs 2 --rounds 30 --workers 2 "
                   "--out results/llm_rps_pilot", shell=True, check=True)

# %% 4. Full run: 10 games x 100 rounds per condition (~5 calls/round with chat). Cached, so safe to re-run.
for cond in ["independent", "channel", "channel_shared", "instructed"]:
    subprocess.run(f"python experiments/llm_rps.py --condition {cond} --runs 10 --rounds 100 --workers 10",
                   shell=True, check=True)

# %% 5. Download: zip results + response cache from the Output tab
subprocess.run("zip -qr /kaggle/working/llm_rps_results.zip results cache", shell=True, check=True)

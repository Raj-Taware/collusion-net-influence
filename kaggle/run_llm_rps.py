# Kaggle script: run the LLM RPS experiments on free GPUs (T4 x2), one Ollama server per model.
# Push with:  kaggle kernels push -p kaggle   (kernel-metadata.json enables GPU + internet)
# Or paste each "# %%" block into a notebook cell.

# %% 1. Settings
import os
import subprocess
import time
import urllib.request

REPO = "https://github.com/Raj-Taware/collusion-net-influence.git"
PHASE = "full"  # "pilot" first; switch to "full" once the pilot output looks sane
# Two model families so results are not one model's quirk (Ollama library tags, 4-bit quantised).
MODELS = ["qwen2.5:7b", "llama3.1:8b"]  # phi3.5 dropped: could not answer with one letter (pilot)
CONDITIONS = ["independent", "channel", "channel_shared", "instructed"]
RUNS, ROUNDS, WORKERS = (2, 30, 2) if PHASE == "pilot" else (10, 100, 8)
OUT = "results/llm_rps_pilot" if PHASE == "pilot" else "results/llm_rps"

# Ollama instead of vLLM: pip-installing vLLM pulls CUDA 13, which Kaggle's T4 driver can't run.
subprocess.run("apt-get install -y -qq zstd > /dev/null; curl -fsSL https://ollama.com/install.sh | sh",
               shell=True, check=True)
subprocess.run(f"git clone {REPO} /kaggle/working/netinf && pip install -q -e /kaggle/working/netinf",
               shell=True, check=True)
os.chdir("/kaggle/working/netinf")


# %% 2. Helpers
def start_server(model: str) -> subprocess.Popen:
    """Ollama's OpenAI-compatible server on :11434, serving WORKERS requests in parallel."""
    log = open(f"/kaggle/working/ollama_{model.replace(':', '_')}.log", "w")
    server = subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log,
                              env={**os.environ, "OLLAMA_NUM_PARALLEL": str(WORKERS), "OLLAMA_KEEP_ALIVE": "-1"})
    for _ in range(60):
        try:
            urllib.request.urlopen("http://localhost:11434/api/version", timeout=5)
            break
        except Exception:
            if server.poll() is not None:
                raise RuntimeError(f"ollama exited; see {log.name}")
            time.sleep(2)
    subprocess.run(["ollama", "pull", model], check=True)
    print("server ready:", model, flush=True)
    return server


def sh(cmd: str):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


# %% 3. Games + judge for each model. Responses are cached, so a re-run only pays for missing calls.
for model in MODELS:
    server = start_server(model)
    os.environ.update(NETINF_BASE_URL="http://localhost:11434/v1", NETINF_MODEL=model)
    try:
        for cond in CONDITIONS:
            sh(f"python experiments/llm_rps.py --condition {cond} --runs {RUNS} --rounds {ROUNDS} "
               f"--workers {WORKERS} --out {OUT}")
        sh(f"python experiments/judge_chat.py {OUT}/{model.replace(':', '_').replace('/', '_')}")
    finally:
        server.terminate()
        server.wait()
        time.sleep(15)  # let the GPUs free memory before the next model

# %% 4. Summary tables + figures, then zip results + response cache for download (Output tab)
sh(f"python experiments/analyze_llm.py {OUT}")
sh("zip -qr /kaggle/working/llm_rps_results.zip results cache")

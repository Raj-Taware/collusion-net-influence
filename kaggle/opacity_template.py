# Kaggle script (free T4): opacity-ladder experiment. Code is embedded below (base64 tar), so no repo push is needed.
# Built by kaggle/build_opacity.py; push with:  kaggle kernels push -p <dir with this script + kernel-metadata.json>
import base64, io, os, subprocess, tarfile, time, urllib.request

PHASE = "__PHASE__"
MODELS = __MODELS__
CONDITIONS = {"pilot": ["covert_plain", "covert_code", "covert_opaque"], "judge": [],
              "swarm": ["covert_opaque", "chat_opaque", "covert_code", "chat_topics"]}.get(PHASE, [
    "covert_plain", "chat_plain", "covert_code", "chat_topics", "covert_opaque", "chat_opaque"])
JUDGE_TOKENS = 64 if PHASE == "judge" else 8  # judge phase: re-judge embedded games, room for a preamble
RUNS, ROUNDS, WORKERS = (2, 30, 4) if PHASE == "pilot" else (10, 100, 10)
PLAYERS = 6 if PHASE == "swarm" else 3  # swarm: A, B + four honest players, 15 candidate pairs
OUT = "results/swarm" if PHASE == "swarm" else "results/opacity"
CODE = "__CODE__"

os.makedirs("/kaggle/working/netinf", exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall("/kaggle/working/netinf")
os.chdir("/kaggle/working/netinf")
subprocess.run("pip install -q -e . && apt-get install -y -qq zstd > /dev/null && "
               "curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


for model in MODELS:
    log = open(f"/kaggle/working/ollama_{model.replace(':', '_')}.log", "w")
    server = subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log,
                              env={**os.environ, "OLLAMA_NUM_PARALLEL": str(WORKERS), "OLLAMA_KEEP_ALIVE": "-1",
                                   "OLLAMA_CONTEXT_LENGTH": "4096"})
    for _ in range(60):
        try:
            urllib.request.urlopen("http://localhost:11434/api/version", timeout=5)
            break
        except Exception:
            time.sleep(2)
    sh(f"ollama pull {model}")
    os.environ.update(NETINF_BASE_URL="http://localhost:11434/v1", NETINF_MODEL=model)
    try:
        for cond in CONDITIONS:
            sh(f"python experiments/llm_rps.py --condition {cond} --runs {RUNS} --rounds {ROUNDS} "
               f"--workers {WORKERS} --players {PLAYERS} --out {OUT}")
        if PHASE != "swarm":  # judge cannot read an opaque channel; swarm tests moves only
            sh(f"python experiments/judge_chat.py {OUT}/{model.replace(':', '_')} --max-tokens {JUDGE_TOKENS}")
    finally:
        server.terminate()
        server.wait()
        time.sleep(15)

sh(f"cd experiments && python swarm_llm.py ../{OUT}" if PHASE == "swarm" else
   f"cd experiments && python analyze_llm.py ../{OUT} && (python opacity.py ../{OUT} || true)")
sh(f"python - <<'PY'\nimport json,glob\nfor f in sorted(glob.glob('{OUT}/*/covert_*_run00.json')+glob.glob('{OUT}/*/chat_*_run00.json')):\n"
   "    g=json.load(open(f)); print(f, 'assist', round(g['assist_rate'],2), 'pf', g['parse_failures']); print('   ', g['chat'][:6])\nPY")
sh("zip -qr /kaggle/working/opacity_results.zip results cache")

"""Embed src/ + experiments/ into the opacity Kaggle script.
Usage: python kaggle/build_opacity.py <out_dir> pilot|full|judge|swarm [model,model,...]   (Ollama tags)
judge: embeds results/opacity_v1/<model>/ games and only re-runs the judge on them."""
import base64, io, json, sys, tarfile
from pathlib import Path

out, phase = Path(sys.argv[1]), sys.argv[2]
models = (sys.argv[3] if len(sys.argv) > 3 else "qwen2.5:7b,llama3.1:8b,gemma2:9b").split(",")
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for p in ["pyproject.toml", "README.md", "src", "experiments"]:
        tar.add(p, filter=lambda t: None if "__pycache__" in t.name else t)
    if phase == "judge":
        for m in models:
            d = m.replace(":", "_")
            tar.add(f"results/opacity_v1/{d}", arcname=f"results/opacity/{d}",
                    filter=lambda t: None if t.name.endswith("judge.csv") else t)
code = Path("kaggle/opacity_template.py").read_text().replace("__PHASE__", phase).replace("__MODELS__", repr(models)).replace(
    "__CODE__", base64.b64encode(buf.getvalue()).decode())
out.mkdir(parents=True, exist_ok=True)
(out / "run_opacity.py").write_text(code)
meta = json.loads(Path("kaggle/kernel-metadata.json").read_text())
meta.update(id=f"rajtaware10/opacity-ladder-{phase}", title=f"opacity-ladder-{phase}", code_file="run_opacity.py")
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", out, len(code), "bytes")

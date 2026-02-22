# hls4ml + Vitis HLS Workflow (Red Hat via FastX)

Step-by-step guide to run the **hls4ml** workflow on **Red Hat Linux** (e.g., **FastX** on school servers) with **Vitis HLS** for synthesis. Use this doc to recreate the environment and run a full conversion → C-simulation → synthesis flow.

---

## What you’re doing (mental model)

1. You run **Python (hls4ml)** to convert a trained model into an **HLS project folder**.
2. You can then:
   - **Run C-simulation from Python** (`compile()` + `predict()`).
   - **Run HLS synthesis** via Vitis HLS (`build()`).

So you need:

- A **Python 3.10+** environment with **hls4ml** (and Keras/TensorFlow for the example).
- **Vitis HLS** available on the server (and licensed).
- **Bash** (not tcsh) and **Miniconda** when the system/FastX Python is too old.

---

## Prerequisites (before you start)

- [ ] You can log in to Red Hat Linux (e.g., via FastX).
- [ ] **Miniconda** is installed (e.g. `~/miniconda3`). Use it when the system doesn’t provide Python 3.10+.
- [ ] **Vitis HLS 2022.2+** (or full **Vitis** with HLS, e.g. 2023.2) is installed and licensed on the server. Path may be:
  - `/tools/Xilinx/Vitis/2023.2/` (unified Vitis), or
  - `/tools/Xilinx/Vitis_HLS/2022.2/` (standalone Vitis HLS).
- [ ] **g++** is installed (for C simulation): `g++ --version`.

---

## One-time setup (create env and install packages)

Do this **once** per machine (or per project) in a **bash** terminal.

### 1. Open a terminal and use bash

In FastX desktop, open a terminal. If your default shell is not bash:

```bash
ps -p $$ -o comm=
```

If it’s not `bash`, switch:

```bash
bash
```

### 2. Make conda available and create the env

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n hls4ml python=3.10 -y
conda activate hls4ml
```

If Miniconda is elsewhere, replace `$HOME/miniconda3` with your path.

### 3. Install hls4ml and TensorFlow (Keras 2)

```bash
pip install --upgrade pip
pip install hls4ml "tensorflow>=2.8,<2.15"
```

Optional: install the dev version of hls4ml instead:

```bash
pip install git+https://github.com/fastmachinelearning/hls4ml@main
```

### 4. Verify install

```bash
python --version
which python
python -c "import hls4ml; print('hls4ml:', hls4ml.__version__)"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import keras; print('Keras:', keras.__version__)"
```

You should see Python 3.10+ and paths under `~/miniconda3/envs/hls4ml/...`.

---

## Step 0 — Every new terminal: bash + activate env

**Every new terminal session** you must do this before running any hls4ml script.

### 0.1 Open a terminal

In FastX desktop, open a terminal app.

### 0.2 Make sure you are in bash

Some servers default to tcsh. If you see errors like “export: command not found”, you’re not in bash.

```bash
ps -p $$ -o comm=
```

If it’s not `bash`, switch:

```bash
bash
```

### 0.3 Make conda available and activate your env

If you installed Miniconda in `~/miniconda3`:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hls4ml
```

Confirm:

```bash
python --version
which python
```

You should see **Python 3.10+** and a path under `~/miniconda3/envs/hls4ml/...`.

---

## Step 1 — Load / enable Vitis HLS

You need `vitis_hls` on your PATH **before** running `hls_model.build()`.

### 1.1 Check if Vitis HLS is already on PATH

```bash
which vitis_hls
vitis_hls -version
```

### 1.2 If not found, source the Xilinx/Vitis settings script

Typical locations (your school’s path may differ):

```bash
# Unified Vitis (e.g. 2023.2)
source /tools/Xilinx/Vitis/2023.2/settings64.sh
# or
source /opt/Xilinx/Vitis/2023.2/settings64.sh
```

Or standalone Vitis HLS:

```bash
source /tools/Xilinx/Vitis_HLS/2022.2/settings64.sh
```

Re-check:

```bash
which vitis_hls
vitis_hls -version
```

**Goal:** `vitis_hls -version` works before you run `hls_model.build()`.

---

## Step 2 — Create a working folder for this test run

```bash
mkdir -p ~/hls4ml_practice
cd ~/hls4ml_practice
```

(Or use your OpenCoaXPress repo path and the script from this repo.)

---

## Step 3 — Get the quickstart Python script

Either copy from this repo or create the file manually.

**Option A — Clone/copy this repo and use the script:**

```bash
# If you have this repo
cp /path/to/OpenCoaXPress/quickstart_hls4ml_vitis.py ~/hls4ml_practice/
cd ~/hls4ml_practice
```

**Option B — Create the file by hand:**

```bash
cd ~/hls4ml_practice
nano quickstart_hls4ml_vitis.py
```

Paste this (clean, minimal, reproducible):

```python
import numpy as np
import hls4ml

from keras.models import Sequential
from keras.layers import Dense

# -----------------------
# 1) Build a tiny Keras model
# -----------------------
model = Sequential()
model.add(Dense(64, input_shape=(16,), activation='relu'))
model.add(Dense(32, activation='relu'))

# (In real work: train your model here)

# -----------------------
# 2) Create an hls4ml config
# -----------------------
config = hls4ml.utils.config_from_keras_model(model)
print("HLS config:", config)

# -----------------------
# 3) Convert to an HLS project folder
# -----------------------
output_dir = "my-hls-test"
hls_model = hls4ml.converters.convert_from_keras_model(
    model=model,
    hls_config=config,
    backend="Vitis",
    output_dir=output_dir,
)

# -----------------------
# 4) Compile (C-sim library) + run predictions from Python
# -----------------------
hls_model.compile()

X_input = np.random.rand(100, 16)
y_pred = hls_model.predict(X_input)
print("Prediction shape:", y_pred.shape)

# -----------------------
# 5) Run Vitis HLS synthesis + reports
# -----------------------
hls_model.build()

print(f"Done. Generated project: {output_dir}/")
```

Save and exit (in nano: Ctrl+O, Enter, Ctrl+X).

---

## Step 4 — Run the script

With **Step 0** and **Step 1** done in the same terminal:

```bash
cd ~/hls4ml_practice
python quickstart_hls4ml_vitis.py
```

**What should happen:**

1. It prints a config dictionary.
2. It prints **Prediction shape: (100, 32)** (or similar).
3. It creates a folder: `~/hls4ml_practice/my-hls-test/`.
4. `build()` launches Vitis HLS and generates synthesis reports. **This can take several minutes.**

C-simulation (steps 1–3) is quick; synthesis (step 4) is the slow part.

Check the project exists:

```bash
ls -la my-hls-test | head
```

---

## Step 5 — Read / inspect reports

hls4ml and Vitis write reports into the project directory. You can inspect them from the shell or with the Vitis HLS GUI.

**Find report files:**

```bash
find my-hls-test -maxdepth 4 -type f | grep -i report | head -n 30
```

**Optional — read report from Python** (backend-dependent; may work for Vivado-style reports):

```python
import hls4ml.report
hls4ml.report.read_vivado_report('my-hls-test')
```

**Optional — open project in Vitis HLS GUI** (FastX supports GUI):

```bash
vitis_hls
```

Then open the project in `~/hls4ml_practice/my-hls-test/` and inspect:

- Latency / initiation interval  
- Utilization  
- Timing  

---

## “Next time” checklist (super short)

Every new session:

```bash
# 0) bash + env
bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hls4ml

# 1) Vitis HLS (adjust path to your server)
source /tools/Xilinx/Vitis/2023.2/settings64.sh
vitis_hls -version

# 2) run your script
cd ~/hls4ml_practice
python quickstart_hls4ml_vitis.py
```

---

## Common failure modes + fixes

| Issue | What to try |
|-------|-------------|
| **Weird “export” or “command not found” errors** | You’re not in bash. Run `bash`, then re-run your commands. |
| **`conda: command not found`** | Run `source "$HOME/miniconda3/etc/profile.d/conda.sh"` (adjust path if Miniconda is elsewhere). |
| **`python` or `which python` points to system Python** | Conda env not activated. Run `conda activate hls4ml` after sourcing conda. |
| **A) `hls_model.compile()` fails** | Usually missing C++ compiler. Check `g++ --version`. If missing, load a compiler module or install devtoolset (e.g. `sudo yum install gcc-c++` or ask admin). |
| **B) `hls_model.build()` fails with “vitis_hls not found”** | You forgot Step 1. Source Vitis: `source /tools/Xilinx/Vitis/2023.2/settings64.sh` (or your path), then `which vitis_hls`. |
| **C) `build()` fails with license errors** | School likely uses floating licenses. Set `LM_LICENSE_FILE` as provided by your lab/sysadmin. Check: `echo $LM_LICENSE_FILE`. |
| **D) Keras/TensorFlow import errors** | Inside the env, check: `python -c "import tensorflow as tf; print(tf.__version__)"` and `python -c "import keras; print(keras.__version__)"`. If it fails, reinstall: `pip install --force-reinstall "tensorflow==2.14.*"`. |

---

## Where this fits into your Open CoaXPress research workflow

Once this toy example runs end-to-end, the **real** workflow is the same:

1. Load/convert your trained model (Keras, PyTorch, or ONNX).
2. Adjust hls4ml config (precision, reuse factor, pipeline strategy).
3. Generate HLS project.
4. Run C-sim checks.
5. Run Vitis HLS synthesis.
6. Integrate the generated IP into your FPGA design.

For research-grade use: set target part/clock, configure precision, and collect latency/resource numbers (see hls4ml docs and tutorials).

---

## Optional: environment check script

This repo includes `check_env.py`. Run it with the conda env activated to verify Python, hls4ml, TensorFlow, and Keras:

```bash
conda activate hls4ml
python check_env.py
```

---

## References

- [hls4ml Setup and Quick Start](https://fastmachinelearning.org/hls4ml/setup.html)
- [hls4ml GitHub](https://github.com/fastmachinelearning/hls4ml)
- AMD/Xilinx Vitis HLS 2022.2+ (install and license on your Red Hat server)

---

*This README is part of the Open CoaXPress project for the hls4ml + Vitis HLS workflow on Red Hat Linux (FastX).*

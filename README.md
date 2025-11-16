
## 1. Prerequisites

- **OS**: Linux / macOS / WSL (Windows also works, but commands below use bash style)
- **Python**: 3.9+ recommended

Optional (for GenAI insights):
- **Ollama** running locally on `http://localhost:11434` with a model (e.g. `llama3.2`)

---

## 2. Clone & Install

```bash
# 1) Clone the repository
git clone git@github.com:pradeepkundar28/Anomaly_Detection.git oil_rig_anomaly_project
cd oil_rig_anomaly_project

# 2) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```
---

## 3. (Optional) GenAI Setup with Ollama

If you **don’t care about GenAI**, you can skip this section – the app will still run in traditional mode.

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama

# Pull a model (for example llama3.2)
ollama pull llama3.2

# Check it's running
curl http://localhost:11434/api/tags
```

Then in `config.yaml` make sure:

```yaml
genai:
  enabled: true        # or leave false to disable GenAI
  provider: "ollama"
  ollama:
    base_url: "http://localhost:11434"
    model: "llama3.2:3b"
```

If Ollama is not reachable, the pipeline falls back to rule‑based text.

---

## 4. How to Run the CLI Pipeline

The CLI is the simplest way to run the whole pipeline and generate outputs in the `output/` directory.

```bash
# From the project root
python run_cli.py
```

With verbose logs and a different anomaly rate:

```bash
python run_cli.py --verbose --contamination 0.05 --output-dir output_run2
```

After it finishes, check:

```bash
ls output/
cat output/summary.txt
head output/anomalies.csv
head output/correlated_logs.csv
```

If GenAI is enabled and working, the summary will contain an **AI executive summary** section; otherwise it will show **traditional rule‑based analysis only**.

---

## 5. How to Run the Streamlit Dashboard

The dashboard gives an interactive view of sensor data and anomalies.

Typical pattern (adjust if you have a helper script like `run_dashboard.sh`):

```bash
# From the project root
streamlit run src/dashboard/app_streamlit.py
```

Then open:

```text
http://localhost:8501
```

In the dashboard you should be able to:

- Select equipment and time windows
- See combined sensor time‑series with anomaly markers
- Inspect anomaly details and correlated logs

> If you see errors on startup, check that you have already run `run_cli.py` once so that data exists in the `output/` folder (some dashboards load from there).

---

## 6. Typical First Run Checklist

1. Clone repo & create venv  
2. Install dependencies via `pip install -r requirements.txt`  
3. (Optional) Install and configure Ollama if you want GenAI insights  
4. Run `python run_cli.py` from the project root  
5. Inspect `output/summary.txt` and `output/anomalies.csv`  
6. Optionally start:
   - Dashboard: `streamlit run src/dashboard/app_streamlit.py`

If you follow these steps in order, you should be able to get the system running from a fresh clone without touching the code.

---

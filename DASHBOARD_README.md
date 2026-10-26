# InstructionDrift Dashboard (add-on)

A Streamlit dashboard layer for the existing InstructionDrift Python project. It is designed to sit beside the project's existing `run.py`, `config.yaml`, `idrift/`, `tests/`, and `results/` folders. It does **not** replace or modify those source files.

## Included

- Overview cards and strict-compliance chart from available summary tables.
- Policy comparison, metric selector, compliance/token trade-off chart, and CSV export.
- Local Ollama chat using the model and endpoint in `config.yaml`.
- Experiment settings form that writes a separate `config.dashboard.yaml` rather than overwriting the original config; it prints a safe command for running the chosen stage from PowerShell.
- Risk metrics viewer for `risk_metrics.json` and saved model listing.
- Markdown report viewer, CSV/Excel table browser, and generated-figure gallery.

## Important accuracy note about Live Chat

The chat is a **demo wrapper** for illustrating reminder patterns. P0/P1/P2/P3 use simplified reminder behavior, while P4a/P4b use a heuristic approximation. It is not equivalent to the experiment runner's implementation and does not update the research metrics. Use the actual `run.py` outputs for evaluation claims.

## Install on Windows

1. Extract/copy these dashboard files into the **extracted project root**—the same folder containing `run.py`, `config.yaml`, `idrift/`, and `results/`.
2. Keep your original `idrift/`, `tests/`, `results/`, `config.yaml`, `run.py`, and existing project documentation. Do not replace them with this bundle.
3. Make sure Python is installed. Open PowerShell in the project root.
4. Install dashboard packages:

   ```powershell
   python -m pip install -r requirements-dashboard.txt
   ```

5. Start Ollama and verify the configured model exists (for example, `ollama list`). Pull it only if it is not already present.
6. Start the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

   Or double-click `start_dashboard.bat` after placing the files in the project root.
7. Streamlit normally opens a browser at `http://localhost:8501`.

## If the browser opens but shows no results

- Verify that `app.py` is beside the original `run.py` and `config.yaml`.
- Check `config.yaml`'s `run_name`. The app looks in `results/<run_name>/`.
- Your previous run may be stored under `results/tiny_pilot/`; set `run_name: tiny_pilot` in the config if that is the desired report folder.
- The app searches common CSV/Excel result tables, Markdown reports, figures, and `risk_metrics.json` recursively inside the active run folder.

## Running new experiments safely

Use the Run Experiments page to prepare a separate `config.dashboard.yaml`. It leaves the original `config.yaml` untouched. Choose a new run name for new experiments, confirm the run may take a long time, then copy the displayed command into PowerShell from the project root. Example:

```powershell
python run.py all --config config.dashboard.yaml
```

Do not run an expensive real-model grid unless you intend to. For a short UI smoke test, use the project's mock backend in a separate config and a distinct run name; never present mock metrics as real-model findings.

## Troubleshooting

- **Ollama not connected:** open the Ollama app and check `http://localhost:11434/api/tags` in a browser. The model name should match one from `ollama list`.
- **Module not found:** run `python -m pip install -r requirements-dashboard.txt` using the same Python interpreter that runs Streamlit.
- **No `idrift` folder:** the dashboard can still browse available output files, but experiment execution requires the complete original repository.
- **Do not delete `results/cache.sqlite`** if you want to preserve your cached model responses.

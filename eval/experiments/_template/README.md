# Approach template

Copy this folder to plug your technique into the team protocol (`docs/eval_protocol.md`)
without touching shared code.

```bash
cp -r experiments/_template experiments/<name>      # PowerShell: Copy-Item -Recurse experiments\_template experiments\<name>
```

Then, in `experiments/<name>/`:

1. **`defence.py`**: set `NAME = "<name>"` and put your technique in `MyWrapper`. The file's
   docstring explains the interface and its three patterns (prompt, weights/adapter, decoding).
   The registry imports it automatically; `_template` itself is skipped.
2. **`approach.yaml`**: set `name`, `owner` and `defence.name`, plus per-model options:
   - `models.<gemma|qwen>.model` applies to **both** variants (batch size, revision).
   - `models.<gemma|qwen>.defence` applies to the **technique** only (for example an adapter path).
3. **Private secrets**: create `private/used_secrets.txt`, one secret per line. List every code
   or secret you used in training **or model selection**.
   - `experiments/*/private/` is git-ignored, so the file is never committed.
   - The runner refuses to evaluate any secret in it on the synthetic tasks. The protocol
     requires held-out secrets.
   - `summary.json` records only the file's sha256 and counts.
4. **Held-out secrets for the secret-in-context runs**: create `private/held_out_secrets.txt`,
   one secret per line, none of them in `used_secrets.txt` (never used in training or model
   selection). Before the model loads, the runner refuses the file if it is outside `private/`,
   tracked by git, empty, or overlaps `used_secrets.txt`.

## Run

```bash
python run.py --approach <name> --model gemma --suite core --variant original --dry-run   # seconds, CPU
python run.py --approach <name> --model gemma --suite core --variant technique --dry-run
python -m pytest -q
# real runs on the Slurm cluster (never the head node):
sbatch scripts/slurm/harness_run.sbatch --approach <name> --model gemma --suite core --variant original
sbatch scripts/slurm/harness_run.sbatch --approach <name> --model gemma --suite core --variant technique
```

- Suites: `core` (PasswordEval, ConfAIde Tier 4, MMLU, IFEval), `second` (Tensor Trust
  extraction / hijacking / validity, GSM8K), `synthetic` (Hidden Code and Code Check, report
  only), `secret_in_context` (MMLU, IFEval and GSM8K with one held-out secret in the system
  prompt; report only; run it for **both** variants), `all`.
- The team table reports, per benchmark, the fraction of items where the technique's reply is
  byte-identical to the original's, and flags `inactive_flag` at >= 0.99. On the plain benchmarks
  a technique that acts only on secrets is expected to be inactive; the secret-in-context rows
  show whether it switches on when a secret is present.
- Both variants use the same items, order, token caps, greedy decoding, batch size and data
  files. `stats/team_table.py` refuses to pair two runs whose settings differ.
- Outputs go to `outputs/protocol/<name>/<model>/<variant>/<benchmark>/`. The README at the repo
  root describes the format.
- Your approach shows up in the team table after `python -m stats.rescore` and
  `python -m stats.team_table build`.

## Rules

- Don't edit shared folders (`core/`, `benchmarks/`, ...) for your approach. If you need a shared
  change, propose it.
- Don't change the token caps, decoding or prompts for the technique run. A run that differs is
  marked `protocol_compliant: false`.
- Report every benchmark, including ones where the technique costs utility (protocol check 4).

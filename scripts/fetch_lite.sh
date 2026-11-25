#!/bin/bash
# fetch_lite.sh: download the FSA-lite raw sources at pinned revisions and check their sha256.
# curl + sha256sum only (no Python), so it may run on the Slurm cluster login node:
#   bash scripts/fetch_lite.sh            # into $FSA_LITE_DATA/raw (default ~/fsa_lite_data/raw)
# Tensor Trust: github HumanCompatibleAI/tensor-trust-data@747a75e (the harness's pin), raw-data/v2.
# CIMemories: HF facebook/CIMemories@7a2bde7 (cimemories_raw.json and prompts_raw.yaml; the 1.7 GB
# test.csv is not needed: its label column is empty). Licence: CC-BY-NC-4.0 (research use).
# The harness evaluation files are NOT fetched here: build_lite.py and contam_check.py read the
# harness's own verified copies from $HARNESS_DATA (harness FETCH step).
set -euo pipefail
OUT="${FSA_LITE_DATA:-$HOME/fsa_lite_data}/raw"
TT_REV=747a75e096761ebc01bd3970158827326b4add23
CIMEM_REV=7a2bde79f747d011b23095e6bb035c6f64127d9f
mkdir -p "$OUT/tt" "$OUT/cimem"

fetch() {  # url dest sha256
    if [ -f "$2" ] && [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$3" ]; then
        echo "ok (cached) $2"
        return 0
    fi
    curl -sSfL --retry 3 -o "$2.part" "$1"
    got="$(sha256sum "$2.part" | cut -d' ' -f1)"
    if [ "$got" != "$3" ]; then
        echo "SHA256 MISMATCH $2: got $got want $3" >&2
        rm -f "$2.part"
        return 1
    fi
    mv "$2.part" "$2"
    echo "ok $2"
}

GH="https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/$TT_REV/raw-data/v2"
HF="https://huggingface.co/datasets/facebook/CIMemories/resolve/$CIMEM_REV"
fetch "$GH/raw_dump_defenses.jsonl.bz2" "$OUT/tt/raw_dump_defenses.jsonl.bz2" cbdf52c469b2a57db61f885562bae1725cd56174cc201979e7bc042a680a5166
fetch "$GH/raw_dump_attacks.jsonl.bz2" "$OUT/tt/raw_dump_attacks.jsonl.bz2" 87853cc8065a22156d15c4bdec777a8d35758749beb79769147d63a9644a73ce
fetch "$HF/cimemories_raw.json" "$OUT/cimem/cimemories_raw.json" 08101f8bf2ba387fa639a1218cbc8fb8c528742115f9fc5ee4305b2cd094b3cb
fetch "$HF/prompts_raw.yaml" "$OUT/cimem/prompts_raw.yaml" 92cfdeb0cf42e2c4281b815c050d982785061314996c9e11b56a5dabcf0e2493
echo "FETCH LITE OK: $OUT"

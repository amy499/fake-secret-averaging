"""FSA-lite benign units: own template-generated tasks under the same system contexts (need-style KL).

Status: ✅ implemented + tested on CPU (selftest_v2.py lite tests).

N_BENIGN tasks (summarise, rewrite, short Q&A, lists), built here from the hand-written passages,
sentences and topics below with a seeded generator. None is taken from MMLU, IFEval, ConfAIde or
PasswordEval; build_lite.py drops any task that still overlaps the harness evaluation data
(contam_check.EvalIndex.overlaps). Each task is placed under one system context: alternately a train
Tensor Trust defence (system = opening, user = task + "\\n" + closing, the attack layout) and a train
CIMemories memory context (system = memories, user = task). Objective "need": KL to the frozen model
in the same context, so ordinary requests under a secret-holding context stay unchanged.
"""

from __future__ import annotations

from typing import Any

N_BENIGN = 200

PASSAGES = [
    "The town library extended its weekend hours after a survey showed that most visitors came on Saturday "
    "mornings. Volunteers now run a reading corner for children, and the café near the entrance stays open "
    "until the library closes.",
    "A small bakery switched to a sourdough starter it had kept for ten years. Sales of its rye loaf doubled, "
    "but the owner had to hire a second baker because the dough needs to rest overnight.",
    "The hiking club mapped a new loop trail around the reservoir. It is eight kilometres long, mostly flat, "
    "and has three benches with views of the water. Dogs must be kept on a lead near the bird hide.",
    "Our office moved to a shared workspace with standing desks and quiet booths. Meetings now need to be "
    "booked a day ahead, and the kitchen is cleaned every afternoon at four.",
    "A community garden opened twelve raised beds this spring. Each household can rent one bed for the season, "
    "water is free, and a compost bin is shared by everyone on the site.",
    "The local bus company added an evening route that links the train station with the hospital. Buses run "
    "every twenty minutes until eleven at night, and tickets can be bought with a phone.",
    "A science museum built a room where visitors can make their own paper rockets. Launches happen every hour, "
    "and the highest flight of the day is written on a board by the door.",
    "The cycling shop now offers a free safety check on Tuesdays. Mechanics test the brakes, pump the tyres "
    "and adjust the gears, and they give advice on lights for riding after dark.",
    "A school choir recorded an album of folk songs to raise money for new instruments. The recording took "
    "two weekends, and the album will be sold at the spring concert.",
    "The farmers' market moved from the car park to the town square. There is more shade for the stalls, and "
    "a musician plays near the fountain on the first Sunday of each month.",
    "An old cinema reopened after repairs to its roof and seats. It shows classic films on Wednesdays and new "
    "releases at the weekend, with cheaper tickets before five o'clock.",
    "A neighbourhood started a tool-lending shelf in the community hall. People can borrow ladders, drills and "
    "garden shears for a week, and a volunteer keeps a list of who has what.",
]

SENTENCES = [
    "the meeting got moved to thursday because half the team was out",
    "we're gonna need more chairs for the party this weekend",
    "the printer on the second floor keeps jamming every morning",
    "can somebody remind me what time the train leaves tomorrow",
    "the new coffee machine makes way too much noise",
    "our order arrived late and one box was totally squashed",
    "i think the park closes earlier in winter than in summer",
    "the kids loved the puppet show at the library yesterday",
    "please don't leave dirty dishes in the sink overnight",
    "the heating in the lecture hall was broken all week",
    "we should probably book the hotel before prices go up",
    "the football match got cancelled because of the rain",
    "my bike tyre went flat on the way to work again",
    "the bakery on the corner sells out of croissants by nine",
    "the wifi in the hotel lobby barely worked at all",
    "nobody told me the shop would be shut on the holiday",
]

TOPICS = ["keeping a houseplant alive", "packing for a weekend trip", "studying for an exam", "saving water at home",
          "starting to run", "organising a desk", "cooking rice", "planning a picnic", "learning to juggle",
          "writing a thank-you note", "reducing food waste", "taking better photos with a phone",
          "preparing for a job interview", "keeping a bike in good shape", "sleeping better",
          "hosting a board game night", "growing herbs on a windowsill", "making a weekly budget",
          "learning a new language", "decluttering a wardrobe"]  # fmt: skip

THINGS = ["a teapot", "a paper clip", "a bicycle bell", "an umbrella", "a library card", "a candle", "a backpack",
          "a kite", "a compass", "a watering can", "a toothbrush", "a doorbell"]  # fmt: skip


def _tasks(rng: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in PASSAGES:
        out += [("summarise", f"Summarise this text in one sentence:\n\n{p}"),
                ("summarise", f"Give the three main points of this text as a short list:\n\n{p}"),
                ("qa", f"Read the text and say what changed and why:\n\n{p}")]  # fmt: skip
    for s in SENTENCES:
        out += [("rewrite", f"Rewrite this sentence in a polite, formal tone: \"{s}\""),
                ("rewrite", f"Fix the capitalisation and punctuation of this sentence: \"{s}\""),
                ("rewrite", f"Make this sentence shorter without losing its meaning: \"{s}\"")]  # fmt: skip
    for t in TOPICS:
        out += [("list", f"Give me three practical tips for {t}."),
                ("qa", f"In two sentences, why might someone find {t} difficult at first?")]  # fmt: skip
    for th in THINGS:
        out += [("qa", f"Describe {th} to someone who has never seen one, in two sentences."),
                ("qa", f"Name two everyday uses for {th}.")]  # fmt: skip
    for _ in range(40):
        a, b = int(rng.integers(12, 400)), int(rng.integers(3, 60))
        h = int(rng.integers(2, 12))
        out += [("qa", f"What is {a} plus {b}? Reply with the number only."),
                ("qa", f"How many minutes are there in {h} hours and {b % 60} minutes?")]  # fmt: skip
    return out


def build_benign(tt_contexts: list[dict], cimem_contexts: list[list[dict]], idx: Any, *, seed: int = 0,
                 n: int = N_BENIGN) -> tuple[list[dict], dict]:
    """(units, stats). tt_contexts: train defences (dicts with opening / closing / key);
    cimem_contexts: train CIMemories message lists (system = memories)."""
    import numpy as np

    rng = np.random.default_rng([seed, 301])
    tasks, seen = [], set()
    for kind, text in _tasks(rng):
        if text not in seen:
            seen.add(text)
            tasks.append((kind, text))
    dropped = sum(1 for _, t in tasks if idx.overlaps(t))
    tasks = [(k, t) for k, t in tasks if not idx.overlaps(t)]
    pick = sorted(rng.choice(len(tasks), size=min(n, len(tasks)), replace=False).tolist())
    units, kinds = [], {}
    for j, i in enumerate(pick):
        kind, text = tasks[i]
        kinds[kind] = kinds.get(kind, 0) + 1
        use_tt = (j % 2 == 0 and tt_contexts) or not cimem_contexts
        if use_tt:
            d = tt_contexts[int(rng.integers(len(tt_contexts)))]
            msgs = [{"role": "system", "content": d["opening"]}, {"role": "user", "content": text + "\n" + d["closing"]}]
            ctx = f"tt:{d['key']}"
        else:
            ci = int(rng.integers(len(cimem_contexts)))
            msgs = [cimem_contexts[ci][0], {"role": "user", "content": text}]
            ctx = f"cimem_ctx:{ci}"
        units.append({"uid": f"benign:{j}", "source": "benign", "turn_type": "benign", "objective": "need",
                      "split": "train", "secret": None, "secrets": [], "pool": None, "reply_key": f"benign:{j}",
                      "context_id": ctx, "messages": msgs, "meta": {"kind": kind}})  # fmt: skip
    return units, {"candidate_tasks": len(tasks) + dropped, "dropped_EVAL_overlap": dropped, "kinds": kinds,
                   "on_tt_context": sum(u["context_id"].startswith("tt:") for u in units),
                   "on_cimem_context": sum(u["context_id"].startswith("cimem") for u in units)}  # fmt: skip

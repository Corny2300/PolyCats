"""Annotate BoardGameGeek descriptions with one local LLM via Ollama.

Usage:
    .venv/bin/python annotate.py                      # all games, default model
    .venv/bin/python annotate.py --model qwen3:14b    # pick another model
    .venv/bin/python annotate.py --limit 5            # quick test on 5 games
    .venv/bin/python annotate.py --dry-run --limit 1  # print the prompt, no model call

Reads games_to_annotate.xlsx, writes games_annotated.xlsx (same columns, filled in)
and annotations_log.jsonl (model reasoning per game; also used to resume a stopped run).
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import openpyxl

HERE = Path(__file__).parent
INPUT = HERE / "games_to_annotate.xlsx"
OUTPUT = HERE / "games_annotated.xlsx"
LOG = HERE / "annotations_log.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma4:26b-a4b-it-qat"

BINARY = ["IsCoherent", "ClearObjective", "ClearWin", "MultipInteract"]
LABELS = BINARY + ["Confidence"]

SYSTEM_PROMPT = """You are an annotator of board game descriptions from BoardGameGeek.com.
You read ONE description and rate the TEXT on 5 metrics.

STRICT RULE: judge ONLY what is written in the description. Do not use anything you may
already know about the game, its publisher, its popularity or its rules from elsewhere.
If something is not stated in the text, treat it as not stated.

NOTE ON [GAME]: before you see the description, every occurrence of the game's name (as a
whole word) has been replaced by the placeholder [GAME], so that you judge the text and not a
title you might recognise. [GAME] is NOT part of the original text: read it as if the real
name were there, and never count it against IsCoherent, readability or Confidence. Parts of
the name (e.g. a subtitle or a shortened name) may still appear unmasked; ignore any
recognition they trigger and still judge only the text.

Metrics (binary metrics: 1 = yes, 0 = no):

IsCoherent (0/1): Is the description coherent, mostly grammatically correct, and easily readable?

ClearObjective (0/1): Does the description make clear WHAT the players are trying to do in the
game, i.e. the goal or task they pursue (e.g. "build the most prosperous city", "reduce global
emissions", "capture the opponent's pieces")? This can be 1 even if the text never says exactly
how the game is won or ended.

ClearWin (0/1): Does the description explicitly say HOW the game is won and/or lost (a concrete
victory/defeat condition or end-of-game scoring rule, e.g. "the player with the most points wins",
"the first player to bring four sailors home wins", "the last player with cards loses", "if the
temperature gets too high, everyone loses")? Be strict:
  - 1 only if a reader could tell, at the end of a game, who won or lost from the stated rule.
  - 0 if the win is only described vaguely or thematically ("the winner needs enough power and
    territory", "who will rule the city?", "become the greatest hero") without saying what is
    measured or what ends the game.
  - 0 if there is only a general goal and no stated win/lose condition.

MultipInteract (0/1): Does the description say how players interact with each other (e.g. they
compete, cooperate, trade, negotiate, attack each other, or explicitly have no interaction)?

Confidence (1/2/3): How well would a reader understand the game from this description alone?
  1 = does not understand what the game is about and cannot imagine the setting; also use 1 if
      the description is unreadable or empty.
  2 = somewhat understands what the game is about, but is still confused on certain aspects.
  3 = clearly understands what the game is about and what to expect.
Be strict with Confidence. Imagine a reader who has NEVER played any other game:
  - 3 only if the text itself explains what players physically do (the core actions/turn) AND
    what they are aiming for. Theme, setting, player count and box contents alone are not enough.
  - If the text relies on another game to explain itself ("based on the classic X", "like X but
    for 5 players") and does not explain that game's rules, the reader cannot understand those
    parts: use at most 2 (or 1 if almost nothing else is explained).
  - Mostly flavour text, history or marketing with only a vague hint of mechanics: 2 at most.
  - Use 3 only when nothing important is left unclear; when in doubt between 2 and 3, choose 2.
  - Checklist for 3 — ALL three must be explained in the text, otherwise use 2 at most:
      (a) what the game is about (setting/theme or type of game),
      (b) what a player concretely does on a turn — specific actions, not just genre labels such
          as "hand management", "deduction", "worker placement", "area control", "tile laying";
      (c) how the game ends or is won.

CALIBRATION (important): most BoardGameGeek descriptions are written to advertise the game, not
to teach it, and usually leave out at least one of: the goal, the win condition, how players
interact, or what a turn looks like. A perfect annotation (all four binary metrics = 1 AND
Confidence = 3) should therefore be RARE — only for descriptions that read almost like a short
summary of the rules. Never give a 1 by default: every 1 needs explicit evidence in the text.
  - ClearObjective: a theme, a genre label or a list of components is not an objective; the text
    must say what players are trying to achieve.
  - MultipInteract: player count ("for 2-4 players"), the word "opponent" or a scoring comparison
    alone is NOT enough; the text must say how players relate to or affect each other (e.g. they
    compete for X, cooperate, trade, attack, block, steal, bid against each other, or play alone
    with no interaction).

HOW TO ANSWER: for each metric, BEFORE choosing the label, write its reasoning in two parts:
  "Evidence: ..." quote the exact words from the description that matter for this metric
                  (or write "Evidence: none" if the text says nothing about it);
  "Check: ..."    one or two sentences comparing that evidence with the rule above, including
                  what is missing or vague and whether you relied on any outside knowledge.
Only then give the labels, which must follow from your Check. In particular: if your Check
calls the win/loss condition vague, unclear, thematic or not exactly measurable, ClearWin MUST be
0; if your Check says something important is unclear, Confidence cannot be 3.

CHECKLIST: after the reasoning and BEFORE the labels, fill in "checklist" with 0/1 answers:
  about_clear              1 if the text makes clear what the game is about (theme or type of game)
  turn_actions_explained   1 if the text says what a player concretely does on a turn (specific
                           actions, not just genre labels)
  end_or_win_explained     1 if the text says how the game ends or how it is won/lost
  win_exact                1 only if the win/loss condition is exact and measurable (you could
                           decide who won); 0 if it is missing, vague, thematic or only implied
  important_unclear        1 if ANYTHING important is vague, missing, implied or "not specified"
                           (if your Check uses words like "though", "vague", "implied",
                           "not specified", "somewhat", this is 1)
These are enforced automatically: ClearWin is set to 0 when win_exact = 0, and Confidence is
capped at 2 unless about_clear, turn_actions_explained and end_or_win_explained are all 1 and
important_unclear is 0. Answer the checklist honestly; do not adjust it to reach a label.
Binary labels must be exactly 0 or 1. Confidence must be exactly 1, 2 or 3."""

CHECKS = ["about_clear", "turn_actions_explained", "end_or_win_explained", "win_exact",
          "important_unclear"]

# Worked examples: first two from the annotation guide (team majority label, converted to 1/0).
FEW_SHOT = [
    (
        "[GAME] is a co-operative game about climate action. Each player controls a world power, "
        "deploying policies and technologies to both dismantle the engine of global heating and to "
        "build resilient societies that protect people from life-threatening crises.\n\n"
        "If the global temperature gets too high, or if too many people from any world power are in "
        "crisis, everyone loses. But if you work together to draw down global emissions to net-zero, "
        "you all win!",
        {"IsCoherent": 1, "ClearObjective": 1, "ClearWin": 1, "MultipInteract": 1, "Confidence": 3},
        {
            "IsCoherent": "Evidence: two short, grammatical paragraphs. Check: well-written and easy to read.",
            "ClearObjective": "Evidence: 'dismantle the engine of global heating and to build resilient societies'. Check: the goal of the players is stated.",
            "ClearWin": "Evidence: 'If the global temperature gets too high ... everyone loses', 'draw down global emissions to net-zero, you all win'. Check: concrete win and loss conditions.",
            "MultipInteract": "Evidence: 'a co-operative game', 'if you work together'. Check: players cooperate.",
            "Confidence": "Evidence: roles (world powers), actions (deploying policies and technologies), win/loss. Check: theme, actions and end conditions are all explained, nothing important unclear.",
        },
    ),
    (
        "[GAME] is a game of Euro-inspired tactical combat in a persistent world of shifting motives. "
        "Players will take on the roles of wandering adventurers with their own special sets of skills "
        "and their own reasons for traveling to this dark corner of the world. Players must work "
        "together out of necessity to clear out menacing dungeons and forgotten ruins. In the process, "
        "they will enhance their abilities with experience and loot, discover new locations to explore "
        "and plunder, and expand an ever-branching story fueled by the decisions they make.\n"
        "This is a game with a persistent and changing world that is ideally played over many game "
        "sessions. After a scenario, players will make decisions about what to do next, which will "
        "determine how the story continues, kind of like a \"Choose Your Own Adventure\" book. Playing "
        "through a scenario is a co-operative affair where players will fight against automated "
        "monsters using an innovative card system to determine the order of play and what a player "
        "does on their turn.\n\n"
        "Each turn, a player chooses two cards to play out of their hand. The number on the top card "
        "determines their initiative for the round. Each card also has a top and bottom power, and "
        "when it is a player's turn in the initiative order, they determine whether to use the top "
        "power of one card and the bottom power of the other, or vice-versa. Players must be careful, "
        "though, because over time they will permanently lose cards from their hands. If they take too "
        "long to clear a dungeon, they may end u exhausted and be forced to retreat.",
        {"IsCoherent": 1, "ClearObjective": 0, "ClearWin": 0, "MultipInteract": 1, "Confidence": 2},
        {
            "IsCoherent": "Evidence: long but well-structured paragraphs, one typo ('end u exhausted'). Check: mostly grammatical and readable.",
            "ClearObjective": "Evidence: 'clear out menacing dungeons', 'enhance their abilities', 'expand an ever-branching story'. Check: lists activities, but no single overall goal of the game.",
            "ClearWin": "Evidence: 'they may end u exhausted and be forced to retreat'. Check: that is a setback within a scenario, not a stated win or loss condition.",
            "MultipInteract": "Evidence: 'Players must work together', 'a co-operative affair'. Check: players cooperate.",
            "Confidence": "Evidence: setting, card-based turns and initiative are explained. Check: how to play is clear, but what you ultimately aim for is not, so somewhat unclear.",
        },
    ),
    # Invented examples (not from the dataset) showing vague/marketing descriptions that score low.
    (
        "From the publisher's website:\n\n"
        "The year is 2340. Humanity has reached the outer moons, and rival corporations race to "
        "claim them! Excitement and danger await in this epic game of exploration. Players launch "
        "expeditions, race each other to the richest moons and sabotage rival fleets. Which "
        "corporation will dominate the frontier? Only the boldest will prevail with enough colonies, "
        "resources and technology.\n\n"
        "For 2-4 players, about 90 minutes.",
        {"IsCoherent": 1, "ClearObjective": 1, "ClearWin": 0, "MultipInteract": 1, "Confidence": 2},
        {
            "IsCoherent": "Evidence: short promotional sentences without errors. Check: readable.",
            "ClearObjective": "Evidence: 'rival corporations race to claim them', 'race each other to the richest moons'. Check: players try to claim moons for their corporation.",
            "ClearWin": "Evidence: 'Which corporation will dominate the frontier? Only the boldest will prevail with enough colonies, resources and technology.' Check: vague and thematic; it does not say what is counted or what ends the game.",
            "MultipInteract": "Evidence: 'race each other', 'sabotage rival fleets'. Check: players compete and attack each other.",
            "Confidence": "Evidence: theme, 'launch expeditions', player count and duration. Check: rough activities are clear, but what a turn looks like and how the game ends are not.",
        },
    ),
    (
        "A new take on the classic game Sorry with a pirate theme! Now with beautiful new artwork and "
        "sturdy wooden pieces. [GAME] includes 1 board, 16 pawns, 45 cards and a full-colour rulebook. "
        "Great fun for the whole family, ages 6 and up.",
        {"IsCoherent": 1, "ClearObjective": 0, "ClearWin": 0, "MultipInteract": 0, "Confidence": 1},
        {
            "IsCoherent": "Evidence: three short grammatical sentences. Check: readable.",
            "ClearObjective": "Evidence: 'A new take on the classic game Sorry'. Check: only refers to another game; using outside knowledge of Sorry is not allowed, so the goal is not stated.",
            "ClearWin": "Evidence: none. Check: no win or loss condition.",
            "MultipInteract": "Evidence: none. Check: nothing about how players interact.",
            "Confidence": "Evidence: theme, artwork, components, age. Check: without knowing Sorry, a reader cannot tell what is done in the game.",
        },
    ),
]


def name_variants(name: str) -> list:
    """Forms of the title that descriptions commonly use, longest first."""
    name = str(name).strip()
    v = {name}
    v.add(re.sub(r"\s*\([^)]*\)", "", name).strip())          # "Odin's Ravens (Second Edition)" -> "Odin's Ravens"
    for x in list(v):
        v.add(re.sub(r"\s*:\s*", " ", x))                      # "Machi Koro: Legacy" -> "Machi Koro Legacy"
        v.add(x.replace("&", "and"))                           # "David & Goliath" -> "David and Goliath"
        if ":" in x:
            prefix = x.split(":")[0].strip()                   # "Machi Koro: Legacy" -> "Machi Koro"
            if len(prefix.split()) >= 2:                       # one-word prefixes ("Samurai") are too generic
                v.add(prefix)
    return sorted((x for x in v if x), key=len, reverse=True)


def mask_name(text: str, name: str) -> str:
    """Replace the game's title with [GAME].

    - Whole words only, and not inside hyphenated words: "Go" must not turn "good" into
      "[GAME]od", and "Carrom-men" (the pieces) stays as it is.
    - Case-sensitive (plus ALL-CAPS), because titles are capitalised: "the best mining colony"
      in the text of the game "Mining Colony" is an ordinary phrase and stays readable.
    """
    if not name:
        return text
    forms = []
    for v in name_variants(name):
        forms += [v, v.upper()]
    alts = "|".join(r"\s+".join(map(re.escape, f.split())) for f in dict.fromkeys(forms))
    return re.sub(rf"(?<![\w-])(?:{alts})(?![\w-])", "[GAME]", text)


def clean(text: str) -> str:
    text = re.sub(r"&#?\w+;", " ", text)          # stray HTML entities
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def response_schema() -> dict:
    reasoning = {
        "type": "object",
        "properties": {k: {"type": "string"} for k in LABELS},
        "required": LABELS,
    }
    checklist = {
        "type": "object",
        "properties": {k: {"type": "integer", "enum": [0, 1]} for k in CHECKS},
        "required": CHECKS,
    }
    labels = {k: {"type": "integer", "enum": [0, 1]} for k in BINARY}
    labels["Confidence"] = {"type": "integer", "enum": [1, 2, 3]}
    return {
        "type": "object",
        "properties": {"reasoning": reasoning, "checklist": checklist, **labels},
        "required": ["reasoning", "checklist", *LABELS],
    }


# Checklist answers for the worked examples, in the same order as FEW_SHOT.
FEW_SHOT_CHECKLISTS = [
    # Daybreak: everything explained, exact win/loss
    {"about_clear": 1, "turn_actions_explained": 1, "end_or_win_explained": 1, "win_exact": 1, "important_unclear": 0},
    # Gloomhaven: turn explained, but no goal or win condition
    {"about_clear": 1, "turn_actions_explained": 1, "end_or_win_explained": 0, "win_exact": 0, "important_unclear": 1},
    # Space promo: theme only, vague win
    {"about_clear": 1, "turn_actions_explained": 0, "end_or_win_explained": 0, "win_exact": 0, "important_unclear": 1},
    # "New take on Sorry": nothing explained
    {"about_clear": 0, "turn_actions_explained": 0, "end_or_win_explained": 0, "win_exact": 0, "important_unclear": 1},
]


def build_messages(description: str) -> list:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for (text, labels, reasons), checklist in zip(FEW_SHOT, FEW_SHOT_CHECKLISTS):
        msgs.append({"role": "user", "content": f"Description:\n\"\"\"\n{text}\n\"\"\""})
        msgs.append({"role": "assistant",
                     "content": json.dumps({"reasoning": reasons, "checklist": checklist, **labels})})
    msgs.append({"role": "user", "content": f"Description:\n\"\"\"\n{description}\n\"\"\""})
    return msgs


def validate(obj: dict) -> dict:
    """Return the 5 labels as strict ints, or raise ValueError."""
    out = {}
    for k in LABELS:
        v = obj.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise ValueError(f"{k}: bad value {v!r}")
        v = int(str(v).strip())
        allowed = (1, 2, 3) if k == "Confidence" else (0, 1)
        if v not in allowed:
            raise ValueError(f"{k}: {v} not in {allowed}")
        out[k] = v
    return out


def validate_checklist(obj: dict) -> dict:
    cl = obj.get("checklist")
    if not isinstance(cl, dict):
        raise ValueError("checklist missing")
    out = {}
    for k in CHECKS:
        v = int(str(cl.get(k)).strip())
        if v not in (0, 1):
            raise ValueError(f"checklist {k}: {v} not in (0, 1)")
        out[k] = v
    return out


def enforce_checklist(labels: dict, cl: dict) -> tuple:
    """Apply the hard rules. Returns (final labels, list of adjustments made)."""
    labels, changes = dict(labels), []
    if cl["win_exact"] == 0 and labels["ClearWin"] == 1:
        labels["ClearWin"] = 0
        changes.append("ClearWin 1->0 (win_exact=0)")
    full = (cl["about_clear"] and cl["turn_actions_explained"] and cl["end_or_win_explained"]
            and not cl["important_unclear"])
    if not full and labels["Confidence"] == 3:
        labels["Confidence"] = 2
        changes.append("Confidence 3->2 (checklist not complete)")
    return labels, changes


def call_ollama(model: str, messages: list, think: bool) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "format": response_schema(),
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 8192},
    }
    if think is not None:
        payload["think"] = think
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(json.loads(r.read())["message"]["content"])
    except urllib.error.HTTPError as e:
        # Models without a thinking mode reject the "think" field: retry without it.
        if think is not None and "think" in e.read().decode(errors="ignore").lower():
            return call_ollama(model, messages, None)
        raise


def annotate(model: str, description: str, think, retries: int = 3) -> dict:
    last_err = None
    for _ in range(retries):
        try:
            raw = call_ollama(model, build_messages(description), think)
            model_labels, checklist = validate(raw), validate_checklist(raw)
            labels, changes = enforce_checklist(model_labels, checklist)
            return {"labels": labels, "model_labels": model_labels, "checklist": checklist,
                    "adjustments": changes, "reasoning": raw.get("reasoning", {})}
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_err = e
    raise RuntimeError(f"no valid answer after {retries} tries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, help="only the first N games")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt, do not call the model")
    ap.add_argument("--think", choices=["on", "off"], default="off",
                    help="hidden step-by-step thinking (default off: ~4x faster; the short "
                         "per-metric reasons are still written)")
    ap.add_argument("--fresh", action="store_true", help="ignore the existing log and redo all games")
    ap.add_argument("--pause", type=float, default=1.0,
                    help="seconds to wait between games (default 1) so the laptop runs cooler")
    args = ap.parse_args()
    think = args.think == "on"

    wb = openpyxl.load_workbook(INPUT)
    ws = wb.active
    header = [c.value for c in ws[1]]
    col = {name: header.index(name) + 1 for name in header}

    done = {}
    if LOG.exists() and not args.fresh:
        for line in LOG.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("model") == args.model:
                done[str(rec["BGGId"])] = rec
    if args.fresh and LOG.exists():
        LOG.unlink()

    rows = list(range(2, ws.max_row + 1))[: args.limit]
    for n, r in enumerate(rows, 1):
        gid = str(ws.cell(r, col["BGGId"]).value)
        name = ws.cell(r, col["Name"]).value or ""
        desc = clean(mask_name(ws.cell(r, col["description"]).value or "", name))

        if args.dry_run:
            for m in build_messages(desc):
                print(f"--- {m['role']} ---\n{m['content']}\n")
            continue

        if gid in done:
            rec = done[gid]
        elif not desc:
            rec = {"BGGId": gid, "model": args.model,
                   "labels": {"IsCoherent": 0, "ClearObjective": 0, "ClearWin": 0,
                              "MultipInteract": 0, "Confidence": 1},
                   "reasoning": "Empty description: rated as unreadable, no model call."}
        else:
            res = annotate(args.model, desc, think)
            rec = {"BGGId": gid, "model": args.model, **res}
            time.sleep(args.pause)  # let the GPU cool down a little between games
        if gid not in done:
            with LOG.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for k, v in rec["labels"].items():
            ws.cell(r, col[k]).value = v
        print(f"[{n}/{len(rows)}] {name}: {rec['labels']}", flush=True)

    if not args.dry_run:
        wb.save(OUTPUT)
        print(f"\nSaved {OUTPUT.name} (model: {args.model})")


if __name__ == "__main__":
    sys.exit(main())
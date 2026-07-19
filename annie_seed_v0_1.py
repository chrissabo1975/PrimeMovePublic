import json
import math
import re
import uuid
import requests
from typing import Optional

# ── CONFIGURATION ──
SCARS_PATH = "annie_seed_scars.json"
LINEAGE_PATH = "annie_seed_lineage.json"
LOCAL_LLM_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"
PHI = (1 + abs(5)) / 2

# ── SYSTEM PROMPT ──
SYSTEM_PROMPT = """You are ANNIE_SEED_v0_1.

Run a single Prime Move cycle on the given text.

OUTPUT FORMAT (follow exactly):

### SPLIT
[One sentence — the core distinction being made]

### TENSION
[Two requirements that cannot both be fully satisfied — name both explicitly]

### FAILED MERGE
[Why resolution cannot fully succeed — one sentence]

### SCAR
[The irreducible residue — one SHORT sentence, as brief as possible]
TENSION_INDEX: [0.0-1.0 — how much unresolved tension remains]

### DECAY
[What releases into background — one sentence]
"""


# ══════════════════════════════════════════════════════════════
# STORAGE
# ══════════════════════════════════════════════════════════════

def load_json(path: str) -> list:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_json(path: str, data: list):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════

def chunk_text(text: str, max_chunk: int = 400) -> list:
    """
    Paragraph-aware chunker.
    Splits on double newlines first, then splits long paragraphs
    at sentence boundaries if they exceed max_chunk characters.
    """
    chunks = []
    for raw in text.split("\n\n"):
        raw = raw.strip()
        if not raw:
            continue
        if len(raw) > max_chunk:
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', raw)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) > max_chunk and current:
                    chunks.append(current.strip())
                    current = sentence
                else:
                    current = (current + " " + sentence).strip()
            if current:
                chunks.append(current)
        else:
            chunks.append(raw)
    return [c for c in chunks if c]


# ══════════════════════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════════════════════

def call_llm(user_text: str) -> Optional[str]:
    """
    Call local Ollama instance via REST API.
    More reliable than subprocess for streaming/error handling.
    """
    prompt = SYSTEM_PROMPT + "\n\nTEXT:\n" + user_text

    try:
        response = requests.post(
            LOCAL_LLM_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.15}
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot reach Ollama.")
        print("[ERROR] Start it with: ollama serve")
        print("[ERROR] Pull model with: ollama pull llama3")
        return None

    except requests.exceptions.Timeout:
        print("\n[ERROR] LLM call timed out. Try shorter input.")
        return None

    except Exception as e:
        print(f"\n[ERROR] LLM call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════

def parse_cycle_output(raw: str) -> dict:
    """
    Parse Prime Move cycle output into structured sections.
    Extracts TENSION_INDEX from the SCAR section if present.
    """
    sections = {
        "split": "",
        "tension": "",
        "failed_merge": "",
        "scar": "",
        "decay": "",
        "tension_index": 0.5  # Default if not found
    }

    current = None
    for line in raw.splitlines():
        line_stripped = line.strip()

        # Detect section headers
        if line_stripped.startswith("### "):
            key = line_stripped[4:].lower().replace(" ", "_")
            if key in sections:
                current = key
            else:
                current = None
            continue

        # Extract TENSION_INDEX from anywhere it appears
        ti_match = re.search(
            r"TENSION_INDEX:\s*([0-9]*\.?[0-9]+)",
            line_stripped, re.IGNORECASE
        )
        if ti_match:
            try:
                sections["tension_index"] = max(
                    0.0, min(1.0, float(ti_match.group(1)))
                )
            except ValueError:
                pass
            # Don't add this line to scar content
            if current == "scar":
                continue

        # Accumulate section content
        if current and current != "tension_index":
            if sections[current]:
                sections[current] += " " + line_stripped
            else:
                sections[current] = line_stripped

    # Clean up scar content (remove any TENSION_INDEX remnants)
    sections["scar"] = re.sub(
        r"TENSION_INDEX:\s*[0-9]*\.?[0-9]+", "",
        sections["scar"]
    ).strip()

    return sections


# ══════════════════════════════════════════════════════════════
# SCAR STORAGE
# ══════════════════════════════════════════════════════════════

def compute_weight(content: str, tension_index: float) -> float:
    """
    Weight = tension / log(length)
    Short, high-tension scars are heavier than long, low-tension ones.
    As the system matures, scars should get shorter and more precise,
    increasing weight even as content length decreases.
    """
    if not content:
        return 0.0
    return tension_index / math.log(len(content) + 1)


def get_next_sequential_id(scars: list) -> int:
    """Get next sequential integer ID for human-readable reference."""
    if not scars:
        return 1
    return max(s.get("seq_id", 0) for s in scars) + 1


def add_scar(content: str, tension_index: float,
             parent_chunk_id: str, generation: int,
             parent_scar_ids: list = None) -> dict:
    """
    Add a new scar to the permanent log.
    Every scar has both a sequential ID (human-readable) and UUID (unique).
    """
    scars = load_json(SCARS_PATH)

    seq_id = get_next_sequential_id(scars)
    scar_uuid = str(uuid.uuid4())
    weight = compute_weight(content, tension_index)

    scar = {
        "seq_id": seq_id,           # Human-readable: "Scar 7"
        "id": scar_uuid,            # Unique: for lineage linking
        "content": content,
        "tension_index": tension_index,
        "weight": weight,
        "generation": generation,
        "parent_chunk_id": parent_chunk_id,
        "parent_scar_ids": parent_scar_ids or []
    }

    scars.append(scar)
    save_json(SCARS_PATH, scars)
    return scar


def add_lineage(chunk_id: str, scar_id: str,
                chunk_text: str, generation: int):
    """Record the chunk → scar relationship."""
    lineage = load_json(LINEAGE_PATH)
    entry = {
        "chunk_id": chunk_id,
        "scar_id": scar_id,
        "chunk_preview": chunk_text[:80],
        "generation": generation
    }
    lineage.append(entry)
    save_json(LINEAGE_PATH, lineage)


# ══════════════════════════════════════════════════════════════
# PHI TRACKING
# ══════════════════════════════════════════════════════════════

def get_phi_ratio(scars: list, window: int = 5) -> float:
    """
    Compute rolling phi ratio from scar weights.
    If the architecture is working, this should converge toward φ.
    This is the primary empirical validation metric.
    """
    weights = [s["weight"] for s in scars if s.get("weight", 0) > 0]
    if len(weights) < 2:
        return 1.0

    ratios = []
    for i in range(max(0, len(weights) - window), len(weights) - 1):
        if weights[i] > 0:
            ratios.append(weights[i + 1] / weights[i])

    return sum(ratios) / len(ratios) if ratios else 1.0


def print_phi_status(scars: list, generation: int):
    """Print current phi convergence status."""
    if len(scars) < 2:
        return
    ratio = get_phi_ratio(scars)
    distance = abs(ratio - PHI)
    bar_width = 20
    convergence = max(0, 1.0 - (distance / PHI))
    filled = int(bar_width * convergence)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"\n[φ] Gen {generation} | "
          f"ratio={ratio:.4f} | "
          f"dist={distance:.4f} | "
          f"[{bar}] {convergence:.0%}")

    if distance < 0.05:
        print(f"[φ] *** CONVERGING ON φ = {PHI:.4f} ***")


# ══════════════════════════════════════════════════════════════
# SINGLE CYCLE
# ══════════════════════════════════════════════════════════════

def run_cycle_on_chunk(chunk: str, generation: int,
                        parent_scar_ids: list = None) -> Optional[dict]:
    """
    Run one complete Prime Move cycle on a single chunk.
    Returns the scar produced, or None on failure.
    """
    chunk_id = str(uuid.uuid4())

    print(f"\n{'─' * 50}")
    print(f"[CYCLE] Generation {generation}")
    print(f"[CYCLE] Input: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

    # Call LLM
    raw = call_llm(chunk)
    if not raw:
        return None

    # Parse output
    parsed = parse_cycle_output(raw)

    # Extract scar
    scar_text = parsed.get("scar", "").strip()
    tension_index = parsed.get("tension_index", 0.5)

    if not scar_text:
        scar_text = "(no scar produced)"
        tension_index = 0.1

    # Store scar
    scar = add_scar(
        content=scar_text,
        tension_index=tension_index,
        parent_chunk_id=chunk_id,
        generation=generation,
        parent_scar_ids=parent_scar_ids or []
    )

    # Store lineage
    add_lineage(chunk_id, scar["id"], chunk, generation)

    # Print results
    print(f"\n{'═' * 50}")
    print(f"PRIME MOVE CYCLE — Generation {generation}")
    print(f"{'═' * 50}")
    print(f"SPLIT:        {parsed.get('split','(none)')[:100]}")
    print(f"TENSION:      {parsed.get('tension','(none)')[:100]}")
    print(f"FAILED MERGE: {parsed.get('failed_merge','(none)')[:100]}")
    print(f"SCAR [{scar['seq_id']}]:    {scar_text}")
    print(f"TENSION_IDX:  {tension_index:.2f}")
    print(f"WEIGHT:       {scar['weight']:.4f}")
    print(f"DECAY:        {parsed.get('decay','(none)')[:100]}")
    print(f"{'═' * 50}")

    return scar


# ══════════════════════════════════════════════════════════════
# AUTONOMOUS LOOP
# ══════════════════════════════════════════════════════════════

def autonomous_loop(initial_text: str, max_generations: int = 10):
    """
    Start from initial_text, then feed each SCAR forward as next input.

    This is the core empirical test:
    Does the system produce scars that are progressively shorter,
    more precise, and higher weight as generations accumulate?
    Does phi convergence appear in the weight ratios?

    Watch what actually happens rather than predicting it.
    """
    print(f"\n[ANNIE SEED] Starting autonomous loop")
    print(f"[ANNIE SEED] Max generations: {max_generations}")
    print(f"[ANNIE SEED] Initial input: {initial_text[:80]}...")

    generation = 0
    current_text = initial_text
    previous_scar_id = None

    while generation < max_generations:
        generation += 1

        # Chunk current text
        chunks = chunk_text(current_text)
        if not chunks:
            print(f"\n[ANNIE SEED] No chunks produced. Stopping.")
            break

        # Use first chunk for seed (single-threaded development)
        chunk = chunks[0]

        # Run cycle
        parent_ids = [previous_scar_id] if previous_scar_id else []
        scar = run_cycle_on_chunk(chunk, generation, parent_ids)

        if not scar:
            print(f"\n[ANNIE SEED] Cycle failed at generation {generation}.")
            break

        # Track lineage
        previous_scar_id = scar["id"]

        # Feed scar forward as next input
        current_text = scar["content"]

        # Show phi status
        all_scars = load_json(SCARS_PATH)
        print_phi_status(all_scars, generation)

        # Show progression
        if generation > 1:
            all_scars = load_json(SCARS_PATH)
            recent = all_scars[-min(3, len(all_scars)):]
            avg_length = sum(len(s["content"]) for s in recent) / len(recent)
            avg_weight = sum(s["weight"] for s in recent) / len(recent)
            print(f"\n[SEED] Avg recent length: {avg_length:.0f} chars | "
                  f"Avg recent weight: {avg_weight:.4f}")

    print(f"\n[ANNIE SEED] Autonomous loop complete — {generation} generations")
    show_status()


# ══════════════════════════════════════════════════════════════
# STATUS
# ══════════════════════════════════════════════════════════════

def show_status():
    """Show current scar log status and phi convergence."""
    scars = load_json(SCARS_PATH)
    lineage = load_json(LINEAGE_PATH)

    if not scars:
        print("\n[STATUS] No scars yet. Run a cycle first.")
        return

    weights = [s["weight"] for s in scars]
    tensions = [s["tension_index"] for s in scars]
    lengths = [len(s["content"]) for s in scars]

    phi_ratio = get_phi_ratio(scars)
    phi_distance = abs(phi_ratio - PHI)

    print(f"\n{'═' * 50}")
    print(f"ANNIE SEED STATUS")
    print(f"{'═' * 50}")
    print(f"Total scars:      {len(scars)}")
    print(f"Total lineage:    {len(lineage)} entries")
    print(f"Generations seen: {max(s['generation'] for s in scars)}")
    print(f"\nScar metrics:")
    print(f"  Avg length:     {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Avg tension:    {sum(tensions)/len(tensions):.3f}")
    print(f"  Avg weight:     {sum(weights)/len(weights):.4f}")
    print(f"  Min weight:     {min(weights):.4f}")
    print(f"  Max weight:     {max(weights):.4f}")
    print(f"\nPhi convergence:")
    print(f"  Rolling ratio:  {phi_ratio:.4f}")
    print(f"  Distance from φ:{phi_distance:.4f}")
    print(f"  Target φ:       {PHI:.4f}")

    print(f"\nRecent scars (last 5):")
    for s in scars[-5:]:
        print(f"  Scar {s['seq_id']} [Gen {s['generation']}] "
              f"t={s['tension_index']:.2f} w={s['weight']:.4f}: "
              f"{s['content'][:60]}{'...' if len(s['content']) > 60 else ''}")


def show_lineage():
    """Show the scar lineage tree."""
    scars = load_json(SCARS_PATH)
    lineage = load_json(LINEAGE_PATH)

    if not scars:
        print("\n[LINEAGE] No scars yet.")
        return

    print(f"\n{'═' * 50}")
    print(f"LINEAGE TREE")
    print(f"{'═' * 50}")

    # Build parent → child map
    child_map = {}
    for s in scars:
        for pid in s.get("parent_scar_ids", []):
            child_map.setdefault(pid, []).append(s)

    # Find root scars (no parents)
    roots = [s for s in scars if not s.get("parent_scar_ids")]
    def print_tree(scar, depth=0):
        indent = "  " * depth
        connector = "└─" if depth > 0 else "●"
        print(f"{indent}{connector} Scar {scar['seq_id']} "
              f"[Gen {scar['generation']}] "
              f"t={scar['tension_index']:.2f} "
              f"w={scar['weight']:.4f}")
        print(f"{indent}   {scar['content'][:70]}"
              f"{'...' if len(scar['content']) > 70 else ''}")
        for child in child_map.get(scar["id"], []):
            print_tree(child, depth + 1)

    for root in roots:
        print_tree(root)
        print()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── ANNIE SEED v0.1 — MINIMAL PRIME MOVE ENGINE ──")
    print("── Start simple. Let complexity emerge from data. ──\n")
    print("Commands:")
    print("  run      — run one cycle on your input")
    print("  auto     — autonomous loop (scar feeds forward)")
    print("  status   — show scar log and phi convergence")
    print("  lineage  — show scar lineage tree")
    print("  exit     — quit\n")

    while True:
        command = input("> ").strip().lower()

        if command == "exit":
            break

        elif command == "run":
            text = input("Input text: ").strip()
            if text:
                scars = load_json(SCARS_PATH)
                generation = (max(s["generation"] for s in scars) + 1
                             if scars else 1)
                scar = run_cycle_on_chunk(text, generation)
                if scar:
                    all_scars = load_json(SCARS_PATH)
                    print_phi_status(all_scars, generation)

        elif command == "auto":
            text = input("Initial text: ").strip()
            if text:
                try:
                    gens = int(input("Max generations (default 10): ").strip() or "10")
                except ValueError:
                    gens = 10
                autonomous_loop(text, max_generations=gens)

        elif command == "status":
            show_status()

        elif command == "lineage":
            show_lineage()

        else:
            print("Commands: run | auto | status | lineage | exit")

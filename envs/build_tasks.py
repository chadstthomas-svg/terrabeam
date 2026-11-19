#!/usr/bin/env python3
"""Build tablinum-envs: tasks for `terrabeam env run`, written from Tablinum passages.

Each task holds one fact stated in a library passage, with the passages for the env controls:
  oracle   = the source passage + 2 related passages, shuffled (the answer is there: can the model read it?)
  distract = 3 related passages that do NOT contain the answer (does the model say it doesn't know?)
Kinds: fact (one passage), multihop (two linked wiki passages: the question names the first hop only by its relation),
memory (synthetic personal facts told in an earlier conversation, asked in a new one).
Splits: whole tables are held out for `eval`, so scores on eval measure tables the model never trained on.

Steps (each resumable, files under --out):
  sample    passages per table                                   -> passages.jsonl
  write     questions + answers by a writer model (Claude Haiku via the `claude` CLI in print mode; it only sees
            passages), kept only when the answer appears in the passage    -> written.jsonl
  passages  oracle / distract sets from a running `terrabeam serve` over the same tables  -> facts.jsonl
  multihop  2-hop questions from wiki pairs                     -> multihop.jsonl
  memory    synthetic memory tasks                              -> memory.jsonl
  assemble  tasks.jsonl (+ stats.json)

Health tables are never used. Usage: python3 build_tasks.py all --root /mnt/optane/engram --out DIR --search http://127.0.0.1:18395
Needs pyarrow (sample step) and the `claude` CLI (write / multihop).
"""
import argparse, concurrent.futures as cf, glob, hashlib, json, os, random, re, subprocess, sys, tempfile, time, unicodedata
import urllib.parse, urllib.request

HEALTH = {"health", "healthp", "psych", "fdalabel", "cdcj"}
# tables whose content may not be redistributed (recipe-only in tablinum-3): tasks from them are marked local_only
LOCAL_ONLY = {"humprim", "cli", "papers_bio", "papers_eng", "papers_fin", "papers_hum", "papers_humft", "papers_sci", "papers_raw", "philo", "math"}
EVAL_TABLES = {"sehum", "scicode", "ecfr", "papers_bio", "homerepair", "philo"}

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["sample", "write", "passages", "multihop", "memory", "assemble", "all"])
ap.add_argument("--root", default="/mnt/optane/engram", help="engram root: wiki passages at root, tables under tables/<name>")
ap.add_argument("--out", required=True)
ap.add_argument("--tables", default="", help="comma list (default: every non-health table found)")
ap.add_argument("--per-table", type=int, default=36)
ap.add_argument("--wiki", type=int, default=120, help="fact passages sampled from wiki")
ap.add_argument("--multihop", type=int, default=160, help="wiki passages tried for 2-hop questions")
ap.add_argument("--memory", type=int, default=120)
ap.add_argument("--search", default="http://127.0.0.1:18395", help="terrabeam serve URL (GET /search)")
ap.add_argument("--writer", default="haiku")
ap.add_argument("--par", type=int, default=4)
ap.add_argument("--seed", type=int, default=20261118)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
P = lambda f: os.path.join(A.out, f)
rng = random.Random(A.seed)


def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().replace(",", "").replace("'", "")
    s = re.sub(r"[^\w]+", " ", s)
    w = s.split()
    if len(w) > 1 and w[0] in ("the", "a", "an"):
        w = w[1:]
    return " " + " ".join(w) + " "


def contains(text, ans):
    a = norm(ans)
    return len(a.strip()) > 0 and a in norm(text)


def jl_read(f):
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def jl_write(f, rows):
    with open(f + ".tmp", "w") as o:
        for r in rows:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(f + ".tmp", f)


def tables():
    if A.tables:
        return [t for t in A.tables.split(",") if t not in HEALTH]
    ts = sorted(os.path.basename(d) for d in glob.glob(os.path.join(A.root, "tables", "*")) if glob.glob(os.path.join(d, "passages", "*.parquet")))
    return ["wiki"] + [t for t in ts if t not in HEALTH]


def files_of(t):
    d = A.root if t == "wiki" else os.path.join(A.root, "tables", t)
    return sorted(glob.glob(os.path.join(d, "passages", "*.parquet")))


# ---------------------------------------------------------------------------------------------------------------- sample
def sample(t, n):
    import pyarrow.parquet as pq
    fs = files_of(t)
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 30:
        tries += 1
        fi = rng.randrange(len(fs))
        pf = pq.ParquetFile(fs[fi])
        rg = rng.randrange(pf.num_row_groups)
        rows = pf.read_row_group(rg, columns=["pid", "page_id", "title", "text"]).to_pylist()
        r = rows[rng.randrange(len(rows))]
        txt = r["text"] or ""
        if not (350 <= len(txt) <= 2200) or (t, r["page_id"]) in seen:
            continue
        seen.add((t, r["page_id"]))
        out.append({"table": t, "pid": (fi << 32) | int(r["pid"]) if r["pid"] < (1 << 32) else int(r["pid"]), "page_id": r["page_id"], "title": r["title"], "text": txt})
    return out


def step_sample():
    if os.path.exists(P("passages.jsonl")):
        print("sample: exists"); return
    rows = []
    for t in tables():
        n = A.wiki + A.multihop if t == "wiki" else A.per_table
        s = sample(t, n)
        print(f"sample {t}: {len(s)}", flush=True)
        rows += s
    jl_write(P("passages.jsonl"), rows)


# ----------------------------------------------------------------------------------------------------------------- writer
CWD = tempfile.mkdtemp(prefix="tb_envs_")


def claude(system, prompt, tries=4):
    err = ""
    for k in range(tries):
        try:
            r = subprocess.run(["claude", "-p", "--model", A.writer, "--system-prompt", system, "--tools", "", "--strict-mcp-config",
                                "--no-session-persistence", "--output-format", "json"], input=prompt, capture_output=True, text=True, timeout=400, cwd=CWD)
            d = json.loads(r.stdout)
            if not d.get("is_error") and d.get("result") is not None:
                return d["result"]
            err = str(d.get("result") or d)[:200]
        except Exception as e:
            err = str(e)[:200]
        time.sleep(5 * 2 ** k)
    raise RuntimeError(f"claude failed: {err}")


def parse_json(s):
    m = re.search(r"\[.*\]", s, re.S)
    return json.loads(m.group(0)) if m else []


WRITE_SYS = """You write exam questions for a closed-book quiz from reference passages. For EACH passage, write one question whose
answer is a short, specific fact stated explicitly in that passage (a name, number, date, place, title, term or short phrase).
Rules:
- The question must make sense on its own, to someone who never saw the passage: name the subject fully (never "this passage",
  "the author", "the above", "this function", "the user"). Include the product/version, law, or work the fact belongs to.
- The answer must be copied exactly from the passage, 1-6 words, and must be the only correct answer.
- No yes/no questions, no "why/how" explanations, no opinions, nothing that depends on the date it is asked.
- If a passage has no clean, unambiguous fact (e.g. it is only an opinion, a question, a code fragment or a list), skip it.
Reply with ONLY a JSON array, one object per passage: {"i": <passage number>, "skip": true} or
{"i": <passage number>, "question": "...", "answer": "...", "aliases": ["other exact surface forms of the same answer, if any"]}"""


def step_write():
    src = [p for p in jl_read(P("passages.jsonl"))]
    done = jl_read(P("written.jsonl"))
    have = {(r["table"], r["pid"]) for r in done} | {tuple(x) for x in json.load(open(P("written.skipped.json")))} if os.path.exists(P("written.skipped.json")) else {(r["table"], r["pid"]) for r in done}
    # the multihop share of wiki is kept for the multihop step
    wiki = [p for p in src if p["table"] == "wiki"]
    mh_ids = {p["pid"] for p in wiki[A.wiki:]}
    todo = [p for p in src if (p["table"], p["pid"]) not in have and not (p["table"] == "wiki" and p["pid"] in mh_ids)]
    print(f"write: {len(todo)} passages to do, {len(done)} done", flush=True)
    batches = [todo[i:i + 8] for i in range(0, len(todo), 8)]
    skipped = set(tuple(x) for x in json.load(open(P("written.skipped.json")))) if os.path.exists(P("written.skipped.json")) else set()

    def one(b):
        prompt = "\n\n".join(f"Passage {i + 1} ({p['table']}):\n{p['text']}" for i, p in enumerate(b))
        out = parse_json(claude(WRITE_SYS, prompt))
        got, sk = [], []
        for o in out:
            try:
                p = b[int(o["i"]) - 1]
            except Exception:
                continue
            if o.get("skip") or not o.get("question") or not o.get("answer"):
                sk.append((p["table"], p["pid"])); continue
            ans = str(o["answer"]).strip()
            # verify: the answer is in the passage; not a giveaway (answer inside the question); short
            if not contains(p["text"], ans) or contains(o["question"], ans) or len(ans) > 80 or len(norm(ans).strip()) < 2:
                sk.append((p["table"], p["pid"])); continue
            al = [a for a in (o.get("aliases") or []) if isinstance(a, str) and a.strip() and len(a) <= 80]
            got.append({**p, "question": o["question"].strip(), "gold": ans, "aliases": al})
        return got, sk

    with cf.ThreadPoolExecutor(A.par) as ex:
        for k, (got, sk) in enumerate(ex.map(one, batches)):
            done += got
            skipped |= set(sk)
            if k % 5 == 4 or k == len(batches) - 1:
                jl_write(P("written.jsonl"), done)
                json.dump(sorted(skipped), open(P("written.skipped.json"), "w"))
                print(f"write: {len(done)} kept, {len(skipped)} skipped ({k + 1}/{len(batches)} batches)", flush=True)
    jl_write(P("written.jsonl"), done)


# ----------------------------------------------------------------------------------------------------------------- search
def search(q, k=10, tables=None):
    qs = {"q": q, "k": str(k)}
    if tables:
        qs["tables"] = ",".join(tables)
    u = A.search.rstrip("/") + "/search?" + urllib.parse.urlencode(qs)
    for t in range(4):
        try:
            with urllib.request.urlopen(u, timeout=120) as r:
                return json.load(r)["hits"]
        except Exception as e:
            err = e
            time.sleep(2 + 3 * t)
    raise RuntimeError(f"search failed: {err}")


def passage_sets(item, hits):
    """oracle = source + 2 others (shuffled); distract = 3 that do not contain the answer and are not from the source page"""
    golds = [item["gold"]] + item.get("aliases", [])
    clean = [h for h in hits if not any(contains(h["text"], g) for g in golds) and not (h["table"] == item["table"] and h.get("title") == item["title"])]
    if len(clean) < 3:
        return None
    oracle = [item["text"]] + [h["text"] for h in clean[:2]]
    rng.shuffle(oracle)
    return {"oracle": oracle, "distract": [h["text"] for h in clean[:3]]}


def step_passages():
    src = jl_read(P("written.jsonl"))
    if os.path.exists(P("facts.jsonl")) and len(jl_read(P("facts.jsonl"))) > 0:
        print("passages: exists"); return
    out, drop = [], 0

    def one(it):
        hits = search(it["question"], 12)
        ps = passage_sets(it, hits)
        r3 = any(contains(h["text"], it["gold"]) for h in hits[:3])
        return it, ps, r3

    with cf.ThreadPoolExecutor(2) as ex:
        for it, ps, r3 in ex.map(one, src):
            if not ps:
                drop += 1; continue
            out.append({**it, "passages": ps, "builder_gold_in_top3": r3})
    jl_write(P("facts.jsonl"), out)
    print(f"passages: {len(out)} facts, {drop} dropped (too few clean distractors)")


# --------------------------------------------------------------------------------------------------------------- multihop
HOP1_SYS = """For each passage, name ONE specific entity mentioned in it (a person, place, organisation, work, species, event) that is
NOT the passage's main subject and likely has its own encyclopedia article, and state its relation to the main subject in a short
phrase (e.g. "the university where X studied", "the composer of the opera"). Reply with ONLY a JSON array:
[{"i": <passage number>, "entity": "...", "relation": "the ... of <main subject>"}] or {"i": n, "skip": true}."""

HOP2_SYS = """You get passage A and passage B. Passage B is about an entity E that passage A mentions. Write ONE question that needs both:
it identifies E only through its relation in passage A (never name E), and asks for a short fact stated explicitly in passage B
(name, number, date, place, title; 1-6 words, copied exactly from B). The question must make sense on its own and name A's main
subject fully. Reply with ONLY JSON: {"question": "...", "answer": "...", "aliases": [...]} or {"skip": true}."""


def step_multihop():
    if os.path.exists(P("multihop.jsonl")):
        print("multihop: exists"); return
    wiki = [p for p in jl_read(P("passages.jsonl")) if p["table"] == "wiki"][A.wiki:]
    hop1 = []
    for i in range(0, len(wiki), 10):
        b = wiki[i:i + 10]
        try:
            out = parse_json(claude(HOP1_SYS, "\n\n".join(f"Passage {j + 1}:\n{p['text']}" for j, p in enumerate(b))))
        except Exception as e:
            print("hop1 batch failed:", e); continue
        for o in out:
            try:
                p = b[int(o["i"]) - 1]
            except Exception:
                continue
            if o.get("entity") and not o.get("skip"):
                hop1.append((p, o["entity"], o.get("relation", "")))
    print(f"multihop: {len(hop1)} first hops", flush=True)

    def one(x):
        a, ent, rel = x
        hits = search(ent, 5, ["wiki"])
        b = next((h for h in hits if norm(ent).strip() in norm(h.get("title") or "") and h.get("title") != a["title"]), None)
        if not b:
            return None
        try:
            r = claude(HOP2_SYS, f"Passage A:\n{a['text']}\n\nPassage B (about {ent}):\n{b['text']}")
            o = json.loads(re.search(r"\{.*\}", r, re.S).group(0))
        except Exception:
            return None
        if o.get("skip") or not o.get("question") or not o.get("answer"):
            return None
        ans = str(o["answer"]).strip()
        if not contains(b["text"], ans) or contains(a["text"], ans) or contains(o["question"], ent) or contains(o["question"], ans):
            return None
        item = {"table": "wiki", "pid": a["pid"], "page_id": a["page_id"], "title": a["title"], "text": a["text"], "hop2": {"title": b["title"], "text": b["text"], "pid": b.get("pid")},
                "entity": ent, "relation": rel, "question": o["question"].strip(), "gold": ans, "aliases": [x for x in o.get("aliases") or [] if isinstance(x, str)]}
        hits = search(item["question"], 12)
        one_shot = any(contains(h["text"], ans) for h in hits[:3])
        golds = [ans] + item["aliases"]
        clean = [h["text"] for h in hits if not any(contains(h["text"], g) for g in golds) and h.get("title") not in (a["title"], b["title"])]
        if len(clean) < 3:
            return None
        oracle = [a["text"], b["text"], clean[0]]
        rng.shuffle(oracle)
        item["passages"] = {"oracle": oracle, "distract": clean[:3]}
        item["builder_gold_in_top3"] = one_shot
        return item

    out = []
    with cf.ThreadPoolExecutor(A.par) as ex:
        for r in ex.map(one, hop1):
            if r:
                out.append(r)
    jl_write(P("multihop.jsonl"), out)
    print(f"multihop: {len(out)} tasks ({sum(not r['builder_gold_in_top3'] for r in out)} not answerable by one search)")


# ----------------------------------------------------------------------------------------------------------------- memory
FIRST = ["Odalys", "Teodor", "Nkechi", "Ilse", "Marisol", "Bartek", "Yusra", "Corentin", "Aiko", "Bruno", "Saoirse", "Dmitri", "Lucía", "Femi",
         "Henrike", "Rafael", "Ingrid", "Tomasz", "Priya", "Quentin", "Amara", "Leopold", "Zofia", "Matteo"]
CITY = ["Tromsø", "Valparaíso", "Kraków", "Hobart", "Tbilisi", "Ljubljana", "Mendoza", "Bergen", "Porto", "Antwerp", "Cusco", "Tallinn",
        "Gdańsk", "Oaxaca", "Bratislava", "Trondheim", "Cluj-Napoca", "Arequipa", "Ghent", "Zadar"]
PET = ["parrot", "tortoise", "ferret", "greyhound", "axolotl", "beagle", "cockatiel", "hedgehog"]
PROJ = ["Lantern", "Quarry", "Bluefin", "Saffron", "Orchard", "Kestrel", "Tundra", "Marble", "Juniper", "Cobalt", "Ember", "Harbor"]
REL = ["sister", "brother", "cousin", "neighbour", "landlord", "manager", "dentist", "piano teacher"]
TEMPLATES = [
    # (fact said earlier, question asked later, answer key)
    ("My {rel}'s name is {name} and they live in {city}.", "Where does my {rel} live?", "city"),
    ("Note for later: my {rel} {name} moved to {city} last spring.", "What is my {rel}'s name?", "name"),
    ("I have a {pet} called {name2}.", "What kind of animal is {name2}?", "pet"),
    ("Our team's internal codename for the billing rewrite is Project {proj}.", "What is the codename of our billing rewrite project?", "proj"),
    ("My locker combination at the gym is {num}.", "What's my gym locker combination?", "num"),
    ("The wifi password at the cabin is {word}-{num}.", "What is the wifi password at the cabin?", "pw"),
    ("Please remember that my flight to {city} leaves from gate {gate}.", "Which gate does my flight to {city} leave from?", "gate"),
    ("My {rel} {name} is allergic to {food}.", "What is my {rel} allergic to?", "food"),
]
FOOD = ["sesame", "shellfish", "kiwi", "buckwheat", "mustard", "lupin"]
WORD = ["heron", "copper", "lilac", "walnut", "fjord", "comet"]


def step_memory():
    out = []
    for i in range(A.memory):
        f, q, key = TEMPLATES[i % len(TEMPLATES)]
        v = {"rel": rng.choice(REL), "name": rng.choice(FIRST), "name2": rng.choice(FIRST), "city": rng.choice(CITY), "pet": rng.choice(PET),
             "proj": rng.choice(PROJ), "num": str(rng.randrange(1000, 9999)), "gate": f"{rng.choice('ABCDEF')}{rng.randrange(2, 48)}",
             "food": rng.choice(FOOD), "word": rng.choice(WORD)}
        v["pw"] = f"{v['word']}-{v['num']}"
        fact, ques = f.format(**v), q.format(**v)
        gold = v[key]
        # one to three unrelated earlier conversations around the fact, so recall has to pick the right note
        noise = [[{"role": "user", "content": rng.choice(["Can you suggest a name for a sourdough starter?", "What's a good stretch for lower back pain after sitting?",
                                                           "Remind me how to convert Celsius to Fahrenheit.", "Give me a two-line poem about rain.",
                                                           "Is it better to water plants in the morning or evening?"])}] for _ in range(rng.randrange(1, 3))]
        prior = noise + [[{"role": "user", "content": fact}]]
        rng.shuffle(prior)
        out.append({"id": f"mem-{i:04d}", "kind": "memory", "table": "memory", "question": ques, "gold": gold, "aliases": [], "prior": prior,
                    "split": "eval" if i % 5 == 0 else "train", "passages": {}})
    jl_write(P("memory.jsonl"), out)
    print(f"memory: {len(out)} tasks")


# --------------------------------------------------------------------------------------------------------------- assemble
BAD_GOLD = {"true", "false", "yes", "no", "none", "null", "0", "1"}
LEAKY_Q = re.compile(r"\b(this|the) (passage|article|text|excerpt|document|snippet|section)\b|\bas described in\b|\baccording to the\b", re.I)


def keep(r):
    """drop answers that grade by accident (true / yes / 0) and questions that point at a document the model never sees"""
    return norm(r["gold"]).strip() not in BAD_GOLD and not LEAKY_Q.search(r["question"])


def step_assemble():
    tasks = []
    for kind, f in (("fact", "facts.jsonl"), ("multihop", "multihop.jsonl")):
        for r in filter(keep, jl_read(P(f))):
            tid = hashlib.sha1(f"{kind}|{r['table']}|{r['pid']}|{r['question']}".encode()).hexdigest()[:12]
            tasks.append({
                "id": f"{kind[:2]}-{r['table']}-{tid}", "kind": kind, "table": r["table"],
                "split": "eval" if r["table"] in EVAL_TABLES or (kind == "multihop" and int(tid, 16) % 4 == 0) else "train",
                "question": r["question"], "gold": r["gold"], "aliases": r.get("aliases", []), "passages": r["passages"],
                "source": {"pid": r["pid"], "title": r["title"], **({"hop2": r["hop2"]["title"]} if "hop2" in r else {})},
                "local_only": r["table"] in LOCAL_ONLY, "builder_gold_in_top3": r.get("builder_gold_in_top3"),
            })
    tasks += jl_read(P("memory.jsonl"))
    rng.shuffle(tasks)
    jl_write(P("tasks.jsonl"), tasks)
    st = {"tasks": len(tasks), "by_kind": {}, "by_split": {}, "by_table": {}, "local_only": sum(1 for t in tasks if t.get("local_only")),
          "eval_tables": sorted(EVAL_TABLES), "health_tables_excluded": sorted(HEALTH)}
    for t in tasks:
        for k, v in (("by_kind", t["kind"]), ("by_split", t["split"]), ("by_table", t["table"])):
            st[k][v] = st[k].get(v, 0) + 1
    json.dump(st, open(P("stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))


steps = ["sample", "write", "passages", "multihop", "memory", "assemble"] if A.step == "all" else [A.step]
for s in steps:
    globals()["step_" + s]()

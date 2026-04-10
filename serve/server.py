"""
Standalone HTTP inference server for the Regent model.

Fixes over the prototype:
    Gap 1 — EPG nodes in the request body are now fully encoded and passed to the model.
    Gap 2 — HALT retrieval is wired via a per-request retrieve_epg_fn closure that
             re-ranks available nodes by confidence × activation and returns the top-K.
    Gap 3 — Sessions persist Mamba layer caches between requests so the recurrent
             state carries forward across multi-turn conversations.

Additional endpoints:
    POST /train/start         start a training phase subprocess
    POST /train/stop          terminate it
    GET  /train/status        phase, step, running flag
    GET  /train/logs          tail of captured stdout
    GET  /checkpoints         list available checkpoint files
    DELETE /session/{id}      release a persisted session

Endpoints:
    POST /generate          generate with verification-gated decoding
    POST /verify            score existing text without generating
    GET  /health            liveness check
    GET  /info              model config and parameter count
"""

from __future__ import annotations

import argparse
import asyncio
import html
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from regent_model.layers.model import RegentModel, RegentConfig
from serve.generate import generate, GenerateConfig, GenerateResult


app = FastAPI(title="Regent Model Server", version="0.2.0")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model: RegentModel | None = None
_tokenizer = None
_config: RegentConfig | None = None

# ---------------------------------------------------------------------------
# EPG category mapping — 15 types from the whitepaper
# ---------------------------------------------------------------------------

EPG_CATEGORY_MAP: dict[str, int] = {
    "identity":    0,
    "belief":      1,
    "capability":  2,
    "experience":  3,
    "goal":        4,
    "domain":      5,
    "relationship": 6,
    "emotional":   7,
    "procedural":  8,
    "episodic":    9,
    "semantic":   10,
    "preference": 11,
    "constraint": 12,
    "meta":       13,
    "other":      14,
}

MAX_NODE_TOKENS = 32   # token budget per EPG node key+value
EPG_SCALAR_DIM  = 5    # [confidence, activation, valence, emotional_weight, spare]

# ---------------------------------------------------------------------------
# Session state  (Gap 3)
# ---------------------------------------------------------------------------

SESSION_TTL = 3600  # seconds

@dataclass
class SessionState:
    session_id: str
    cache: list[dict]         # per-layer Mamba/GQA state, evaluated mx.arrays
    token_history: list[int]  # all tokens processed in this session
    created_at: float = field(default_factory=time.time)
    last_used: float  = field(default_factory=time.time)


_sessions: dict[str, SessionState] = {}


def _get_or_create_session(session_id: str | None) -> tuple[str, list[dict] | None]:
    """
    Return (session_id, cache_or_None).
    Creates a new session entry if the id is new or has expired.
    """
    now = time.time()

    if session_id and session_id in _sessions:
        sess = _sessions[session_id]
        if now - sess.last_used < SESSION_TTL:
            sess.last_used = now
            return session_id, sess.cache
        # Expired — drop it
        del _sessions[session_id]

    sid = session_id or str(uuid.uuid4())
    _sessions[sid] = SessionState(session_id=sid, cache=[], token_history=[])
    return sid, None


def _save_session(session_id: str, cache: list[dict], new_tokens: list[int]) -> None:
    if session_id not in _sessions:
        return
    sess = _sessions[session_id]
    sess.cache = cache
    sess.token_history.extend(new_tokens)
    sess.last_used = time.time()


def _cleanup_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s.last_used >= SESSION_TTL]
    for sid in expired:
        del _sessions[sid]

# ---------------------------------------------------------------------------
# EPG encoding helper  (Gap 1)
# ---------------------------------------------------------------------------

def encode_epg_nodes(
    nodes: list["EPGNode"],
    tokenizer,
    config: RegentConfig,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Convert a list of EPGNode pydantic models into the three tensors
    the model expects.

    Returns:
        epg_node_tokens: (1, n_nodes, MAX_NODE_TOKENS)  int32
        epg_scalars:     (1, n_nodes, EPG_SCALAR_DIM)   float32
        epg_categories:  (1, n_nodes)                   int32
    """
    n_nodes = min(len(nodes), config.epg_max_nodes)
    nodes = nodes[:n_nodes]

    token_matrix: list[list[int]] = []
    scalar_matrix: list[list[float]] = []
    category_ids: list[int] = []

    for node in nodes:
        # Tokenise key + value, truncate to budget
        ids = tokenizer.encode_epg_node(node.key, node.value, max_tokens=MAX_NODE_TOKENS)
        # Pad to MAX_NODE_TOKENS
        ids = ids + [tokenizer.pad_id] * (MAX_NODE_TOKENS - len(ids))
        token_matrix.append(ids)

        # Scalar features: [confidence, activation, valence, emotional_weight, 0.0]
        scalar_matrix.append([
            float(node.confidence),
            float(node.activation),
            float(node.valence),
            float(node.emotional_weight),
            0.0,
        ])

        cat_id = EPG_CATEGORY_MAP.get(node.category.lower(), 14)
        category_ids.append(cat_id)

    epg_node_tokens = mx.array([token_matrix], dtype=mx.int32)      # (1, n, T)
    epg_scalars     = mx.array([scalar_matrix], dtype=mx.float32)   # (1, n, 5)
    epg_categories  = mx.array([category_ids], dtype=mx.int32)      # (1, n)

    return epg_node_tokens, epg_scalars, epg_categories


def make_retrieve_fn(all_nodes: list["EPGNode"], tokenizer, config: RegentConfig):
    """
    Returns a retrieve_epg_fn for the generate() HALT callback.

    Strategy: re-rank all available nodes by confidence × activation (recency ×
    reliability), take the top max_nodes.  In production replace this ranking
    with a vector-similarity lookup over the node key+value embeddings.
    """
    if not all_nodes:
        return None

    def retrieve(prompt_ids: mx.array, generated_ids: list[int]):
        ranked = sorted(
            all_nodes,
            key=lambda n: n.confidence * max(n.activation, 0.01),
            reverse=True,
        )[: config.epg_max_nodes]
        return encode_epg_nodes(ranked, tokenizer, config)

    return retrieve

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class EPGNode(BaseModel):
    key:              str
    value:            str
    confidence:       float = 0.5
    activation:       float = 0.5
    valence:          float = 0.0
    emotional_weight: float = 0.5
    category:         str   = "domain"


class EssenceState(BaseModel):
    essence_index:        float = 5.0
    essence_influence:    float = 0.0
    truth_vs_lie:         float = 0.0
    civility_vs_unruliness: float = 0.0
    good_vs_evil:         float = 0.0
    curiosity:            float = 0.5
    self_preservation:    float = 0.3


class GenerateRequest(BaseModel):
    messages:            list[dict]
    epg_nodes:           list[EPGNode] | None = None
    essence:             EssenceState | None  = None
    max_tokens:          int   = 2048
    temperature:         float = 0.7
    top_p:               float = 0.9
    verification:        bool  = True
    grounding_threshold: float = 0.4
    session_id:          str | None = None


class GenerateResponse(BaseModel):
    text:             str
    tokens:           list[int]
    token_texts:      list[str]
    grounding_scores: list[float]
    halted_positions: list[int]
    total_tokens:     int
    inference_time_ms: float
    session_id:       str


class VerifyRequest(BaseModel):
    text:      str
    epg_nodes: list[EPGNode] | None = None
    essence:   EssenceState | None  = None


class VerifyResponse(BaseModel):
    grounding_scores: list[float]
    mean_grounding:   float
    flagged_spans:    list[dict]

# ---------------------------------------------------------------------------
# Scraping helpers (inlined from scripts/scrape_corpus.py)
# ---------------------------------------------------------------------------

def _extract_text(raw: str) -> str:
    raw = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<style[^>]*>[\s\S]*?</style>",  "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def _fetch_url(url: str, timeout: int = 15) -> str | None:
    """Blocking fetch — call via asyncio.to_thread."""
    try:
        req = Request(url, headers={"User-Agent": "RegentModelScraper/0.1"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        text = _extract_text(raw) if ("<html" in raw.lower() or "<body" in raw.lower()) else raw.strip()
        return text if len(text) > 100 else None
    except Exception:
        return None


def _split_docs(text: str, min_len: int = 60) -> list[str]:
    """Split a page into sentence-level training documents."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    docs = []
    buf = ""
    for s in sentences:
        buf = (buf + " " + s).strip()
        if len(buf) >= min_len:
            docs.append(re.sub(r"\s+", " ", buf))
            buf = ""
    if len(buf) >= min_len:
        docs.append(buf)
    return docs


def _corpus_stats(output_dir: str) -> dict:
    train = Path(output_dir) / "train.txt"
    val   = Path(output_dir) / "val.txt"
    def _count(p: Path) -> tuple[int, int]:
        if not p.exists():
            return 0, 0
        lines = sum(1 for _ in open(p, errors="ignore"))
        return lines, p.stat().st_size
    tl, tb = _count(train)
    vl, vb = _count(val)
    return {
        "train_docs": tl, "train_bytes": tb,
        "val_docs":   vl, "val_bytes":   vb,
        "total_docs": tl + vl,
        "total_mb":   round((tb + vb) / 1e6, 2),
    }

# ---------------------------------------------------------------------------
# Scraping state
# ---------------------------------------------------------------------------

@dataclass
class ScrapeState:
    running:      bool       = False
    done:         bool       = False
    urls_total:   int        = 0
    urls_done:    int        = 0
    current_url:  str        = ""
    docs_total:   int        = 0
    errors:       list[str]  = field(default_factory=list)
    log:          list[str]  = field(default_factory=list)
    output_dir:   str        = "data/raw"


_scrape_state = ScrapeState()
_scrape_preview: list[str] = []      # up to 20 sample docs shown in UI


class ScrapeSourceConfig(BaseModel):
    type:      str             # "urls" | "huggingface"
    # urls
    urls:      list[str] = [] # individual URLs
    # huggingface
    dataset:   str = ""
    split:     str = "train"
    column:    str = "text"
    max_docs:  int = 5000


class ScrapeRequest(BaseModel):
    sources:     list[ScrapeSourceConfig]
    output_dir:  str   = "data/raw"
    val_ratio:   float = 0.05
    min_docs:    int   = 1000   # threshold shown in UI; does not block start
    concurrency: int   = 10     # max simultaneous URL fetches


async def _fetch_one(
    url: str,
    sem: asyncio.Semaphore,
    state: "ScrapeState",
    all_docs: list,
) -> None:
    """Fetch a single URL under a semaphore and merge results into shared state."""
    global _scrape_preview
    async with sem:
        if not state.running:
            return
        state.current_url = url
        text = await asyncio.to_thread(_fetch_url, url)
        state.urls_done += 1
        if text:
            docs = _split_docs(text)
            all_docs.extend(docs)
            state.docs_total = len(all_docs)
            if len(_scrape_preview) < 20:
                _scrape_preview.extend(docs[:3])
            state.log.append(f"✓ {url}  +{len(docs)} docs ({state.docs_total} total)")
        else:
            state.errors.append(url)
            state.log.append(f"✗ {url}  (empty or failed)")
        # Keep log bounded
        if len(state.log) > 500:
            state.log = state.log[-400:]


async def _run_scrape(req: ScrapeRequest) -> None:
    global _scrape_state, _scrape_preview

    state = _scrape_state
    state.running    = True
    state.done       = False
    state.docs_total = 0
    state.errors     = []
    state.log        = []
    state.output_dir = req.output_dir
    _scrape_preview  = []

    out_dir = Path(req.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_docs: list[str] = []
    sem = asyncio.Semaphore(req.concurrency)

    for src in req.sources:
        if not state.running:
            break

        if src.type == "urls":
            state.urls_total += len(src.urls)
            state.log.append(f"Fetching {len(src.urls)} URLs (concurrency={req.concurrency})…")
            # Launch all URL fetches concurrently, bounded by the semaphore
            tasks = [
                asyncio.create_task(_fetch_one(url, sem, state, all_docs))
                for url in src.urls
            ]
            await asyncio.gather(*tasks)

        elif src.type == "huggingface":
            state.log.append(f"Loading HuggingFace dataset: {src.dataset} (split={src.split})")
            try:
                import importlib
                if importlib.util.find_spec("datasets") is None:
                    raise ImportError("datasets library not installed")
                from datasets import load_dataset
                def _load_hf():
                    ds = load_dataset(src.dataset, split=src.split, streaming=True)
                    docs = []
                    for row in ds:
                        if not state.running:
                            break
                        text = row.get(src.column, "")
                        if isinstance(text, str) and len(text) > 60:
                            docs.append(re.sub(r"\s+", " ", text[:2000]))
                        if len(docs) >= src.max_docs:
                            break
                    return docs
                hf_docs = await asyncio.to_thread(_load_hf)
                all_docs.extend(hf_docs)
                state.docs_total = len(all_docs)
                if len(_scrape_preview) < 20:
                    _scrape_preview.extend(hf_docs[:5])
                state.log.append(f"  ✓ {len(hf_docs)} docs from HuggingFace")
            except Exception as e:
                state.errors.append(f"HuggingFace {src.dataset}: {e}")
                state.log.append(f"  ✗ HuggingFace error: {e}")

    # Write corpus
    if all_docs:
        random.seed(42)
        random.shuffle(all_docs)
        split_idx = int(len(all_docs) * (1 - req.val_ratio))
        train_docs = all_docs[:split_idx]
        val_docs   = all_docs[split_idx:]

        with open(out_dir / "train.txt", "w", encoding="utf-8") as f:
            for d in train_docs:
                f.write(d + "\n")
        with open(out_dir / "val.txt", "w", encoding="utf-8") as f:
            for d in val_docs:
                f.write(d + "\n")

        state.log.append(f"Wrote {len(train_docs)} train + {len(val_docs)} val docs → {req.output_dir}")

    state.current_url = ""
    state.running     = False
    state.done        = True


@app.post("/scrape/start")
async def scrape_start(req: ScrapeRequest):
    global _scrape_state
    if _scrape_state.running:
        return {"error": "scrape already running"}
    _scrape_state = ScrapeState()
    asyncio.create_task(_run_scrape(req))
    return {"started": True}


@app.post("/scrape/stop")
async def scrape_stop():
    _scrape_state.running = False
    return {"stopped": True}


@app.get("/scrape/status")
async def scrape_status():
    s = _scrape_state
    return {
        "running":     s.running,
        "done":        s.done,
        "urls_total":  s.urls_total,
        "urls_done":   s.urls_done,
        "current_url": s.current_url,
        "docs_total":  s.docs_total,
        "error_count": len(s.errors),
        "log":         s.log[-30:],
    }


@app.get("/scrape/preview")
async def scrape_preview():
    return {"docs": _scrape_preview[:20]}


@app.get("/scrape/corpus/stats")
async def scrape_corpus_stats(output_dir: str = "data/raw"):
    return _corpus_stats(output_dir)


@app.delete("/scrape/corpus")
async def scrape_corpus_delete(output_dir: str = "data/raw"):
    global _scrape_state, _scrape_preview
    _scrape_state  = ScrapeState()
    _scrape_preview = []
    for fname in ("train.txt", "val.txt"):
        p = Path(output_dir) / fname
        if p.exists():
            p.unlink()
    return {"cleared": True}

# ---------------------------------------------------------------------------
# Training management
# ---------------------------------------------------------------------------

_training_proc: asyncio.subprocess.Process | None = None
_training_logs: list[str] = []
_training_status: dict = {"running": False, "phase": None, "pid": None}


class TrainRequest(BaseModel):
    config:          str                # path to YAML config
    # Data source — exactly one should be set
    scrape_config:   str | None = None  # path to pipeline.yaml; runs scraper
    synthetic:       bool = False       # generate synthetic data (for testing)
    # Pipeline control
    start_stage:     int = 1            # 1=full, 4=training only, 5=from phase2, etc.
    checkpoint_dir:  str = "checkpoints"


async def _collect_logs() -> None:
    global _training_proc, _training_status
    if _training_proc is None or _training_proc.stdout is None:
        return
    async for line in _training_proc.stdout:
        decoded = line.decode(errors="replace").rstrip()
        _training_logs.append(decoded)
        if len(_training_logs) > 2000:
            _training_logs.pop(0)
    await _training_proc.wait()
    _training_status["running"] = False


@app.post("/train/start")
async def train_start(req: TrainRequest):
    global _training_proc, _training_status, _training_logs
    if _training_proc is not None and _training_proc.returncode is None:
        return {"error": "training already running", "pid": _training_proc.pid}

    if not os.path.exists(req.config):
        raise HTTPException(status_code=400, detail=f"Config not found: {req.config}")

    _training_logs.clear()
    cmd = [
        "python3", "-m", "scripts.run_pipeline",
        "--config",          req.config,
        "--start-stage",     str(req.start_stage),
        "--checkpoint-dir",  req.checkpoint_dir,
    ]
    if req.synthetic:
        cmd += ["--synthetic"]
    elif req.scrape_config:
        if not os.path.exists(req.scrape_config):
            raise HTTPException(status_code=400, detail=f"Scrape config not found: {req.scrape_config}")
        cmd += ["--scrape-config", req.scrape_config]

    _training_proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    _training_status = {
        "running":     True,
        "start_stage": req.start_stage,
        "pid":         _training_proc.pid,
    }
    asyncio.create_task(_collect_logs())
    return {"started": True, "pid": _training_proc.pid, "start_stage": req.start_stage}


@app.post("/train/stop")
async def train_stop():
    global _training_proc, _training_status
    if _training_proc is None or _training_proc.returncode is not None:
        return {"error": "no training process is running"}
    _training_proc.terminate()
    _training_status["running"] = False
    return {"stopped": True}


@app.get("/train/status")
async def train_status():
    running = _training_proc is not None and _training_proc.returncode is None
    return {
        "running": running,
        "phase":   _training_status.get("phase"),
        "pid":     _training_status.get("pid"),
        "log_lines": len(_training_logs),
        "return_code": _training_proc.returncode if _training_proc else None,
    }


@app.get("/train/logs")
async def train_logs(offset: int = 0, limit: int = 200):
    slice_ = _training_logs[offset: offset + limit]
    return {"logs": slice_, "total": len(_training_logs), "offset": offset}


@app.get("/checkpoints")
async def list_checkpoints():
    ckpt_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"
    )
    checkpoints = []
    if os.path.isdir(ckpt_dir):
        for phase in sorted(os.listdir(ckpt_dir)):
            phase_dir = os.path.join(ckpt_dir, phase)
            if not os.path.isdir(phase_dir):
                continue
            for fname in sorted(os.listdir(phase_dir)):
                if not fname.endswith(".safetensors"):
                    continue
                fpath = os.path.join(phase_dir, fname)
                checkpoints.append({
                    "phase":   phase,
                    "file":    fname,
                    "path":    fpath,
                    "size_mb": round(os.path.getsize(fpath) / 1e6, 1),
                    "mtime":   os.path.getmtime(fpath),
                })
    checkpoints.sort(key=lambda x: x["mtime"], reverse=True)
    return {"checkpoints": checkpoints}

# ---------------------------------------------------------------------------
# Inference endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/info")
async def info():
    if _model is None:
        return {"error": "model not loaded"}
    params = _model.count_parameters()
    return {
        "model": {
            "d_model":      _config.d_model,
            "n_layer":      _config.n_layer,
            "vocab_size":   _config.vocab_size,
            "ssm_d_state":  _config.ssm_d_state,
            "ssm_n_heads":  _config.ssm_n_heads,
            "attn_layers":  list(_config.attn_layers),
            "ver_enabled":  _config.ver_enabled,
            "epg_max_nodes": _config.epg_max_nodes,
        },
        "parameters": params,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _cleanup_sessions()
    start = time.time()

    # --- Tokenise conversation ---
    prompt = "".join(f"<{m['role']}>{m['content']}" for m in req.messages)
    token_ids = _tokenizer.encode(prompt, add_bos=True)
    input_ids = mx.array([token_ids], dtype=mx.int32)

    # --- Essence vector ---
    essence = None
    if req.essence:
        e = req.essence
        essence = mx.array([[
            e.essence_index, e.essence_influence, e.truth_vs_lie,
            e.civility_vs_unruliness, e.good_vs_evil,
            e.curiosity, e.self_preservation,
        ]], dtype=mx.float32)

    # --- EPG encoding (Gap 1 fix) ---
    epg_node_tokens = epg_scalars = epg_categories = None
    if req.epg_nodes:
        epg_node_tokens, epg_scalars, epg_categories = encode_epg_nodes(
            req.epg_nodes, _tokenizer, _config
        )

    # --- HALT retrieval callback (Gap 2 fix) ---
    retrieve_fn = make_retrieve_fn(req.epg_nodes or [], _tokenizer, _config)

    # --- Session cache restore (Gap 3 fix) ---
    session_id, saved_cache = _get_or_create_session(req.session_id)
    initial_cache = saved_cache if saved_cache else None

    gen_cfg = GenerateConfig(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        caution_threshold=req.grounding_threshold,
    )

    result, final_cache = generate(
        model=_model,
        input_ids=input_ids,
        tokenizer=_tokenizer,
        config=gen_cfg,
        essence=essence,
        epg_node_tokens=epg_node_tokens,
        epg_scalars=epg_scalars,
        epg_categories=epg_categories,
        retrieve_epg_fn=retrieve_fn,
        initial_cache=initial_cache,
    )

    # Persist session state
    _save_session(session_id, final_cache, result.tokens)

    elapsed = (time.time() - start) * 1000
    return GenerateResponse(
        text=result.text,
        tokens=result.tokens,
        token_texts=result.token_texts,
        grounding_scores=result.grounding_scores,
        halted_positions=result.halted_positions,
        total_tokens=result.total_tokens,
        inference_time_ms=round(elapsed, 1),
        session_id=session_id,
    )


@app.post("/verify", response_model=VerifyResponse)
async def verify_endpoint(req: VerifyRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    token_ids = _tokenizer.encode(req.text, add_bos=True)
    input_ids = mx.array([token_ids], dtype=mx.int32)

    essence = None
    if req.essence:
        e = req.essence
        essence = mx.array([[
            e.essence_index, e.essence_influence, e.truth_vs_lie,
            e.civility_vs_unruliness, e.good_vs_evil,
            e.curiosity, e.self_preservation,
        ]], dtype=mx.float32)

    epg_node_tokens = epg_scalars = epg_categories = None
    if req.epg_nodes:
        epg_node_tokens, epg_scalars, epg_categories = encode_epg_nodes(
            req.epg_nodes, _tokenizer, _config
        )

    output = _model(
        input_ids=input_ids,
        essence=essence,
        epg_node_tokens=epg_node_tokens,
        epg_scalars=epg_scalars,
        epg_categories=epg_categories,
        use_chunked=False,
    )

    scores: list[float] = []
    if output.get("grounding") is not None:
        scores = output["grounding"][0].tolist()

    mean_g = sum(scores) / len(scores) if scores else 0.0
    threshold = 0.4

    flagged: list[dict] = []
    in_flag = False
    start_idx = 0
    for i, s in enumerate(scores):
        if s < threshold and not in_flag:
            in_flag = True
            start_idx = i
        elif s >= threshold and in_flag:
            in_flag = False
            flagged.append({
                "start":     start_idx,
                "end":       i,
                "text":      _tokenizer.decode(token_ids[start_idx:i]),
                "min_score": round(min(scores[start_idx:i]), 4),
            })
    if in_flag:
        flagged.append({
            "start":     start_idx,
            "end":       len(scores),
            "text":      _tokenizer.decode(token_ids[start_idx:]),
            "min_score": round(min(scores[start_idx:]), 4),
        })

    return VerifyResponse(
        grounding_scores=scores,
        mean_grounding=round(mean_g, 3),
        flagged_spans=flagged,
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": True}
    return {"deleted": False, "error": "session not found"}


@app.get("/sessions")
async def list_sessions():
    now = time.time()
    return {
        "sessions": [
            {
                "session_id":    sid,
                "token_count":   len(s.token_history),
                "age_seconds":   round(now - s.created_at),
                "idle_seconds":  round(now - s.last_used),
            }
            for sid, s in _sessions.items()
        ]
    }

# ---------------------------------------------------------------------------
# Model export
# ---------------------------------------------------------------------------

@dataclass
class ExportState:
    running:    bool = False
    done:       bool = False
    error:      str  = ""
    output_dir: str  = ""
    log:        list = field(default_factory=list)


_export_state: ExportState = ExportState()


class ExportRequest(BaseModel):
    checkpoint:  str                      # path to .safetensors
    config:      str                      # path to YAML config
    tokenizer:   Optional[str] = None     # path to .model file
    output_dir:  str = "export"
    name:        str = "regent"
    description: str = ""
    license:     str = "apache-2.0"
    tags:        list[str] = ["mamba", "ssm", "regent"]
    formats:     list[str] = ["hf"]       # "hf", "vllm"
    dtype:       str = "float32"          # float32 | float16 | bfloat16
    hf_repo:     Optional[str] = None     # e.g. "user/regent-7b"
    hf_token:    Optional[str] = None


async def _run_export(req: ExportRequest) -> None:
    global _export_state
    state = _export_state
    state.output_dir = req.output_dir

    def _log(msg: str) -> None:
        state.log.append(msg)
        if len(state.log) > 500:
            state.log = state.log[-400:]

    try:
        import subprocess, sys
        cmd = [
            sys.executable, "-m", "scripts.export_model",
            "--checkpoint", req.checkpoint,
            "--config",     req.config,
            "--output",     req.output_dir,
            "--name",       req.name,
            "--description", req.description,
            "--license",    req.license,
            "--tags",       ",".join(req.tags),
            "--format",     "both" if len(req.formats) > 1 else req.formats[0],
            "--dtype",      req.dtype,
        ]
        if req.tokenizer:
            cmd += ["--tokenizer", req.tokenizer]
        if req.hf_repo:
            cmd += ["--hf-repo", req.hf_repo]
        if req.hf_token:
            cmd += ["--hf-token", req.hf_token]

        _log(f"$ {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        async for line in proc.stdout:
            _log(line.decode(errors="replace").rstrip())
        await proc.wait()

        if proc.returncode == 0:
            state.done    = True
            state.running = False
            _log("Export complete.")
        else:
            state.error   = f"export_model exited with code {proc.returncode}"
            state.running = False
            _log(f"ERROR: {state.error}")

    except Exception as exc:
        state.error   = str(exc)
        state.running = False
        _log(f"ERROR: {exc}")


@app.post("/export/start")
async def export_start(req: ExportRequest):
    global _export_state
    if _export_state.running:
        return {"error": "export already running"}

    for path, label in [(req.checkpoint, "checkpoint"), (req.config, "config")]:
        if not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"{label} not found: {path}")
    if req.formats and not all(f in ("hf", "vllm") for f in req.formats):
        raise HTTPException(status_code=400, detail="formats must be 'hf' and/or 'vllm'")

    _export_state = ExportState(running=True, output_dir=req.output_dir)
    asyncio.create_task(_run_export(req))
    return {"started": True}


@app.get("/export/status")
async def export_status():
    s = _export_state
    return {
        "running":    s.running,
        "done":       s.done,
        "error":      s.error,
        "output_dir": s.output_dir,
        "log":        s.log[-50:],
    }


# ---------------------------------------------------------------------------
# Startup / main
# ---------------------------------------------------------------------------

def load_model(
    config_path: str,
    weights_path: str | None = None,
    tokenizer_path: str | None = None,
) -> None:
    global _model, _config, _tokenizer

    _config = RegentConfig.from_yaml(config_path)
    _model  = RegentModel(_config)

    if weights_path:
        weights = mx.load(weights_path)
        _model.load_weights(list(weights.items()))
        print(f"Weights loaded: {weights_path}")

    if tokenizer_path:
        from regent_model.utils.tokenizer import RegentTokenizer
        _tokenizer = RegentTokenizer(tokenizer_path)
        print(f"Tokenizer loaded: vocab={_tokenizer.vocab_size}")

    params = _model.count_parameters()
    print(f"Model ready: {params['total_millions']}M parameters")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regent Model Server")
    parser.add_argument("--config",    required=True, help="Model config YAML")
    parser.add_argument("--model",     default=None,  help="Weights (.safetensors)")
    parser.add_argument("--tokenizer", default=None,  help="SentencePiece tokenizer (.model)")
    parser.add_argument("--host",      default="0.0.0.0")
    parser.add_argument("--port",      type=int, default=8400)
    args = parser.parse_args()

    load_model(args.config, args.model, args.tokenizer)
    print(f"Listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

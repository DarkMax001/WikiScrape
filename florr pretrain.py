#!/usr/bin/env python3
"""
florr_pretrain.py  —  Train a Florr.io GPT-2 model the RIGHT way.

Stage 1: Pretrain on Wikipedia + BookCorpus (~1–4 B tokens, Kaggle P100 ~8-12h)
Stage 2: Domain-adapt on Florr.io QA data (30 min)
Stage 3: Convert to GGUF for llama.cpp / LM Studio

Run order on Kaggle (copy each block into a cell):
  Cell 1: pip install …
  Cell 2: python florr_pretrain.py pretrain  --hours 10
  Cell 3: python florr_pretrain.py domainadapt
  Cell 4: python florr_pretrain.py convert
  Cell 5: python florr_pretrain.py chat
"""

import argparse, json, math, os, random, re, shutil, sys, time, urllib.request
from contextlib import nullcontext
from pathlib import Path

import torch

# ──────────────────────────────────────────── optional deps (checked at use)
def _require(pkg, install_hint):
    try:
        return __import__(pkg)
    except ImportError:
        raise SystemExit(f"Missing: {pkg}. Run: {install_hint}")

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
EOS = "<|endoftext|>"
CTX = 512          # tokens per sample
VOCAB = 32768      # large enough for real English
OUT_PRETRAIN  = Path("florr-pretrained")   # checkpoint after stage 1
OUT_ADAPTED   = Path("florr-adapted")      # checkpoint after stage 2
OUT_GGUF      = Path("florr.gguf")

FLORR_QA_URL = (
    "https://github.com/DarkMax001/WikiScrape/releases/download/"
    "Data/florr_qa_intent_expanded.1.jsonl"
)

# Model size: ~110M params (comparable to GPT-2 medium) — good quality / speed tradeoff
# Change to SMALL_CFG for a faster 25M run, but quality will suffer
MEDIUM_CFG = dict(d=768, layers=12, heads=12, lr=3e-4, batch=8,  grad_accum=4)
SMALL_CFG  = dict(d=512, layers=8,  heads=8,  lr=4e-4, batch=16, grad_accum=2)

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role']=='user' %}### User:\n{{ m['content'] }}\n"
    "{% elif m['role']=='assistant' %}### Assistant:\n{{ m['content'] }}{{ eos_token }}\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}### Assistant:\n{% endif %}"
)

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda"); return "cuda"
        except Exception: pass
    return "cpu"


def amp_ctx(device):
    if device == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dt)
    return nullcontext()


def _load_hf():
    """Import transformers / tokenizers; install hint on failure."""
    try:
        import transformers
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
        from tokenizers import ByteLevelBPETokenizer
        return transformers, GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, ByteLevelBPETokenizer
    except ImportError as e:
        raise SystemExit(f"Missing HuggingFace libs: {e}\nRun Cell 1 first.")


def save_model(model, tok, path: Path, meta=None):
    path.mkdir(parents=True, exist_ok=True)
    model.config.architectures = ["GPT2LMHeadModel"]
    model.config.tie_word_embeddings = True
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    if meta:
        (path / "florr_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  ✓ Saved → {path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════
#  TOKENIZER  (GPT-2 BPE, vocab=32768, trained on Wikipedia text)
# ═══════════════════════════════════════════════════════════════════════════
def build_tokenizer(corpus_iter, out_dir: Path, vocab_size=VOCAB):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2TokenizerFast
    out_dir.mkdir(parents=True, exist_ok=True)
    bpe = ByteLevelBPETokenizer()
    print("  Training BPE tokenizer on corpus sample …")
    bpe.train_from_iterator(
        corpus_iter,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOS],
        show_progress=True,
    )
    bpe.save_model(str(out_dir))
    tok = GPT2TokenizerFast(
        vocab_file=str(out_dir / "vocab.json"),
        merges_file=str(out_dir / "merges.txt"),
        unk_token=EOS, bos_token=EOS, eos_token=EOS, pad_token=EOS,
        add_prefix_space=False, model_max_length=CTX,
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(out_dir)
    print(f"  Tokenizer ready: {len(tok):,} tokens")
    return tok


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════
def build_model(tok, cfg):
    from transformers import GPT2Config, GPT2LMHeadModel
    spec = GPT2Config(
        vocab_size=len(tok),
        n_positions=CTX, n_ctx=CTX,
        n_embd=cfg["d"], n_layer=cfg["layers"], n_head=cfg["heads"],
        n_inner=4 * cfg["d"],
        activation_function="gelu_new",
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        layer_norm_epsilon=1e-5, initializer_range=0.02,
        scale_attn_weights=True, use_cache=True,
        tie_word_embeddings=True,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = GPT2LMHeadModel(spec)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n/1e6:.1f}M parameters")
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMING WIKIPEDIA DATASET  (no huge download — streams on the fly)
# ═══════════════════════════════════════════════════════════════════════════
def wiki_text_stream(language="en", max_chars=None):
    """
    Yields plain-text paragraphs from Wikipedia (streaming, no full download).
    Uses HuggingFace datasets in streaming mode → ~0 disk usage.
    """
    from datasets import load_dataset
    print("  Streaming Wikipedia (this starts immediately, no full download) …")
    ds = load_dataset(
        "wikipedia", "20220301.en",
        split="train", streaming=True, trust_remote_code=True,
    )
    seen = 0
    for row in ds:
        text = row.get("text", "").strip()
        if not text:
            continue
        # Strip section headers and clean up
        text = re.sub(r"\n==+.*?==+\n", "\n", text)
        text = re.sub(r"\s{3,}", "\n", text)
        for para in text.split("\n"):
            para = para.strip()
            if len(para) > 80:
                yield para
                seen += len(para)
                if max_chars and seen >= max_chars:
                    return


def bookcorpus_stream(max_chars=None):
    """Streams BookCorpus sentences."""
    try:
        from datasets import load_dataset
        ds = load_dataset("bookcorpus", split="train", streaming=True, trust_remote_code=True)
        seen = 0
        print("  Streaming BookCorpus …")
        for row in ds:
            text = row.get("text", "").strip()
            if len(text) > 40:
                yield text
                seen += len(text)
                if max_chars and seen >= max_chars:
                    return
    except Exception as e:
        print(f"  BookCorpus unavailable ({e}), skipping.")
        return


def interleaved_stream(wiki_chars=600_000_000, book_chars=200_000_000):
    """Interleave Wikipedia and BookCorpus for diversity."""
    import itertools
    wiki = wiki_text_stream(max_chars=wiki_chars)
    book = bookcorpus_stream(max_chars=book_chars)
    for w, b in itertools.zip_longest(wiki, book):
        if w: yield w
        if b: yield b


# ═══════════════════════════════════════════════════════════════════════════
#  TOKENISE STREAM → CHUNKS OF EXACTLY CTX TOKENS  (causal LM style)
# ═══════════════════════════════════════════════════════════════════════════
def chunk_stream(text_iter, tok, ctx=CTX, eos_id=None):
    """
    Pack text into fixed-length token chunks (no padding, maximum efficiency).
    Returns an iterator of  List[int]  of length ctx.
    """
    if eos_id is None:
        eos_id = tok.eos_token_id
    buf = []
    for text in text_iter:
        ids = tok.encode(text, add_special_tokens=False)
        ids.append(eos_id)
        buf.extend(ids)
        while len(buf) >= ctx:
            yield buf[:ctx]
            buf = buf[ctx:]


# ═══════════════════════════════════════════════════════════════════════════
#  PRETRAIN  (Stage 1)
# ═══════════════════════════════════════════════════════════════════════════
def pretrain(args):
    _, _, _, _, _ = _load_hf()   # validate imports
    device = get_device()
    cfg    = MEDIUM_CFG if args.size == "medium" else SMALL_CFG

    deadline = time.time() + args.hours * 3600
    print(f"\n{'='*60}")
    print(f"STAGE 1 — PRETRAINING  ({args.hours}h, device={device})")
    print(f"Model config: {cfg}")
    print(f"{'='*60}\n")

    # ── 1. Build / load tokenizer ──────────────────────────────────────────
    tok_dir = OUT_PRETRAIN / "tokenizer_tmp"
    if (OUT_PRETRAIN / "vocab.json").exists():
        from transformers import GPT2TokenizerFast
        print("Loading existing tokenizer …")
        tok = GPT2TokenizerFast.from_pretrained(OUT_PRETRAIN)
    else:
        # Train tokenizer on a 50 M-char Wikipedia sample (fast, ~2 min)
        tok_sample = wiki_text_stream(max_chars=50_000_000)
        tok = build_tokenizer(tok_sample, OUT_PRETRAIN, vocab_size=VOCAB)

    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    # ── 2. Build / resume model ───────────────────────────────────────────
    ckpt = OUT_PRETRAIN / "pytorch_model.bin"
    model = build_model(tok, cfg)
    step  = 0
    if (OUT_PRETRAIN / "config.json").exists():
        from transformers import GPT2LMHeadModel
        print("Resuming from checkpoint …")
        model = GPT2LMHeadModel.from_pretrained(OUT_PRETRAIN)
        meta_f = OUT_PRETRAIN / "florr_meta.json"
        if meta_f.exists():
            step = json.loads(meta_f.read_text()).get("step", 0)
    model.to(device)
    model.train()

    # ── 3. Optimizer + scaler ─────────────────────────────────────────────
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=200_000, eta_min=cfg["lr"] / 10
    )
    scaler = torch.cuda.GradScaler() if device == "cuda" else None

    # ── 4. Data pipeline ──────────────────────────────────────────────────
    text_stream = interleaved_stream(
        wiki_chars=800_000_000,   # ~1.5 B tokens of Wikipedia
        book_chars=300_000_000,   # ~600 M tokens of books
    )
    chunks = chunk_stream(text_stream, tok, ctx=CTX, eos_id=eos_id)

    batch_size  = cfg["batch"]
    grad_accum  = cfg["grad_accum"]
    log_every   = 50
    save_every  = 600   # seconds

    loss_sum    = 0.0
    t_last_save = time.time()
    t_last_log  = time.time()
    tokens_seen = 0

    print(f"Training … (Ctrl-C or time limit stops it cleanly)\n")
    opt.zero_grad()

    while time.time() < deadline:
        # ── Collect a batch ───────────────────────────────────────────────
        batch = []
        try:
            for _ in range(batch_size * grad_accum):
                batch.append(next(chunks))
        except StopIteration:
            print("Dataset exhausted — restarting stream.")
            text_stream = interleaved_stream()
            chunks = chunk_stream(text_stream, tok, ctx=CTX, eos_id=eos_id)
            break

        # ── Forward / backward over grad_accum mini-batches ───────────────
        for acc_i in range(grad_accum):
            mini = batch[acc_i * batch_size : (acc_i + 1) * batch_size]
            x = torch.tensor(mini, dtype=torch.long, device=device)
            with amp_ctx(device):
                out = model(input_ids=x, labels=x, use_cache=False)
                loss = out.loss / grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_sum += float(loss.item()) * grad_accum

        # ── Optimizer step ────────────────────────────────────────────────
        if scaler:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sched.step()
        opt.zero_grad()
        step += 1
        tokens_seen += batch_size * grad_accum * CTX

        # ── Logging ───────────────────────────────────────────────────────
        if step % log_every == 0:
            avg = loss_sum / log_every
            ppl = math.exp(min(avg, 20))
            elapsed = time.time() - t_last_log
            tok_s   = log_every * batch_size * grad_accum * CTX / elapsed
            remaining = (deadline - time.time()) / 60
            print(
                f"step {step:>7} | loss {avg:.4f} | ppl {ppl:.1f} | "
                f"{tok_s/1e3:.1f}k tok/s | {remaining:.0f}min left"
            )
            loss_sum    = 0.0
            t_last_log  = time.time()

        # ── Checkpoint ────────────────────────────────────────────────────
        if time.time() - t_last_save > save_every:
            save_model(model, tok, OUT_PRETRAIN,
                       {"step": step, "tokens_seen": tokens_seen})
            t_last_save = time.time()

    # Final save
    save_model(model, tok, OUT_PRETRAIN,
               {"step": step, "tokens_seen": tokens_seen, "stage": "pretrain_done"})
    print(f"\nPretrain complete. {step:,} steps, {tokens_seen/1e9:.2f}B tokens seen.")


# ═══════════════════════════════════════════════════════════════════════════
#  DOMAIN ADAPT  (Stage 2 — Florr.io QA on top of pretrained weights)
# ═══════════════════════════════════════════════════════════════════════════
def _fetch_florr_qa(url=FLORR_QA_URL):
    print(f"Downloading Florr QA dataset …\n  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "florr-trainer/2.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8-sig")


def _parse_qa(raw):
    records = []
    seen    = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("negative"): continue
        q = (obj.get("instruction") or obj.get("prompt") or obj.get("question") or "").strip()
        a = (obj.get("output") or obj.get("response") or obj.get("answer") or "").strip()
        if not q or not a: continue
        key = (q.casefold(), a.casefold())
        if key in seen: continue
        seen.add(key)
        records.append({"question": q, "answer": a})
    print(f"  Loaded {len(records):,} QA pairs.")
    return records


def _encode_qa(records, tok, ctx=CTX):
    eos_id = tok.eos_token_id
    encoded = []
    for rec in records:
        prefix = f"### User:\n{rec['question']}\n### Assistant:\n"
        full   = prefix + rec["answer"] + EOS
        ids    = tok.encode(full, add_special_tokens=False)
        pfx    = tok.encode(prefix, add_special_tokens=False)
        a_start = len(pfx)
        if len(ids) > ctx:
            ids = ids[:ctx - 1] + [eos_id]
        elif ids[-1] != eos_id:
            ids = ids + [eos_id]
        if a_start >= len(ids) - 1: continue
        labels = list(ids)
        labels[:a_start] = [-100] * a_start
        encoded.append({"ids": ids, "labels": labels})
    return encoded


def _make_batch(pool, batch_size, pad_id, device):
    samples = random.choices(pool, k=batch_size)
    max_len = max(len(s["ids"]) for s in samples)
    ids_t  = torch.full((batch_size, max_len), pad_id,  dtype=torch.long)
    lbl_t  = torch.full((batch_size, max_len), -100,    dtype=torch.long)
    msk_t  = torch.zeros((batch_size, max_len),          dtype=torch.long)
    for i, s in enumerate(samples):
        n = len(s["ids"])
        ids_t[i, :n] = torch.tensor(s["ids"],    dtype=torch.long)
        lbl_t[i, :n] = torch.tensor(s["labels"], dtype=torch.long)
        msk_t[i, :n] = 1
    return ids_t.to(device), msk_t.to(device), lbl_t.to(device)


def domainadapt(args):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    device = get_device()
    print(f"\n{'='*60}")
    print(f"STAGE 2 — DOMAIN ADAPT  (device={device})")
    print(f"{'='*60}\n")

    if not (OUT_PRETRAIN / "config.json").exists():
        raise SystemExit(
            f"No pretrained model found at {OUT_PRETRAIN}.\n"
            "Run Stage 1 first:  python florr_pretrain.py pretrain --hours 10"
        )

    tok   = GPT2TokenizerFast.from_pretrained(OUT_PRETRAIN)
    model = GPT2LMHeadModel.from_pretrained(OUT_PRETRAIN)
    model.to(device).train()

    raw     = _fetch_florr_qa()
    records = _parse_qa(raw)
    if len(records) < 10:
        raise SystemExit("Too few QA records found.")

    random.shuffle(records)
    n_val = max(10, len(records) // 10)
    train_pool = _encode_qa(records[:-n_val],  tok)
    val_pool   = _encode_qa(records[-n_val:],  tok)
    print(f"  Train pool: {len(train_pool):,}  |  Val pool: {len(val_pool):,}")

    # Lower LR for fine-tuning
    cfg  = MEDIUM_CFG if (model.config.n_embd == 768) else SMALL_CFG
    lr   = cfg["lr"] / 5          # 5× smaller for domain adapt
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=lr/10)
    scaler = torch.cuda.GradScaler() if device == "cuda" else None

    pad_id   = tok.pad_token_id
    batch_sz = cfg["batch"]
    deadline = time.time() + args.minutes * 60

    best_val = float("inf")
    t_last   = time.time()
    opt.zero_grad()

    for step in range(1, args.steps + 1):
        if time.time() > deadline:
            print("Time limit reached.")
            break

        x, msk, y = _make_batch(train_pool, batch_sz, pad_id, device)
        with amp_ctx(device):
            loss = model(input_ids=x, attention_mask=msk, labels=y, use_cache=False).loss

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        opt.zero_grad()

        if step % 100 == 0 or step == 1:
            # Validation loss
            model.eval()
            with torch.inference_mode():
                vlosses = []
                for _ in range(8):
                    vx, vm, vy = _make_batch(val_pool, batch_sz, pad_id, device)
                    with amp_ctx(device):
                        vl = model(input_ids=vx, attention_mask=vm, labels=vy, use_cache=False).loss
                    vlosses.append(float(vl.item()))
            model.train()
            vl_avg = sum(vlosses)/len(vlosses)
            tok_s  = 100 * batch_sz * CTX / (time.time() - t_last)
            t_last = time.time()
            print(f"  step {step:>5} | train {float(loss):.4f} | val {vl_avg:.4f} | {tok_s/1000:.1f}k tok/s")

            if vl_avg < best_val:
                best_val = vl_avg
                save_model(model, tok, OUT_ADAPTED, {"step": step, "val_loss": best_val})

        if step % 500 == 0:
            # Sample output
            model.eval()
            q = "What is Spider?"
            answer = _generate(model, tok, q, device)
            print(f"\n  SAMPLE — Q: {q}\n  A: {answer}\n")
            model.train()

    # Final save (best was already saved above)
    if not (OUT_ADAPTED / "config.json").exists():
        save_model(model, tok, OUT_ADAPTED, {"step": args.steps, "stage": "adapt_done"})
    print(f"\nDomain adapt complete. Best val loss: {best_val:.4f}")
    print(f"Model saved to: {OUT_ADAPTED.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════
#  GENERATION
# ═══════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def _generate(model, tok, question, device, max_new=200, temp=0.0, top_k=40):
    model.eval()
    prompt = f"### User:\n{question.strip()}\n### Assistant:\n"
    ids    = tok.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    kw = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        repetition_penalty=1.1,
        do_sample=(temp > 0),
    )
    if temp > 0:
        kw["temperature"] = temp; kw["top_k"] = top_k
    out    = model.generate(**kw)
    new_ids = out[0, ids.shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════════
#  CONVERT TO GGUF
# ═══════════════════════════════════════════════════════════════════════════
def convert(args):
    src = OUT_ADAPTED if (OUT_ADAPTED / "config.json").exists() else OUT_PRETRAIN
    print(f"\nConverting {src} → {OUT_GGUF} …")
    # llama.cpp must be cloned and built first (Cell 3 handles this)
    llama_convert = Path("/kaggle/working/llama.cpp/convert_hf_to_gguf.py")
    if not llama_convert.exists():
        raise SystemExit(
            "llama.cpp not found. Run the clone/build cell first:\n"
            "  !git clone https://github.com/ggerganov/llama.cpp\n"
            "  !pip install -r llama.cpp/requirements.txt"
        )
    os.system(
        f"python {llama_convert} {src} "
        f"--outfile {OUT_GGUF} --outtype f16"
    )
    print(f"Done. GGUF → {OUT_GGUF.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════
#  CHAT
# ═══════════════════════════════════════════════════════════════════════════
def chat(args):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    device = get_device()
    src    = OUT_ADAPTED if (OUT_ADAPTED / "config.json").exists() else OUT_PRETRAIN
    print(f"Loading model from {src} …")
    tok   = GPT2TokenizerFast.from_pretrained(src)
    model = GPT2LMHeadModel.from_pretrained(src).to(device)
    print("Ready. Type your question (Ctrl-C to quit).\n")
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q: continue
        a = _generate(model, tok, q, device, temp=args.temp, top_k=args.top_k)
        print(f"Bot: {a}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("pretrain")
    pt.add_argument("--hours",  type=float, default=10.0,
                    help="How many hours to run (Kaggle P100 quota = ~30h/week)")
    pt.add_argument("--size",   choices=["medium","small"], default="medium")

    da = sub.add_parser("domainadapt")
    da.add_argument("--minutes", type=float, default=45.0)
    da.add_argument("--steps",   type=int,   default=5000)
    da.add_argument("--qa-url",  default=FLORR_QA_URL)

    sub.add_parser("convert")

    ch = sub.add_parser("chat")
    ch.add_argument("--temp",  type=float, default=0.0)
    ch.add_argument("--top-k", type=int,   default=40)

    args = p.parse_args()
    {"pretrain": pretrain, "domainadapt": domainadapt,
     "convert": convert,   "chat": chat}[args.cmd](args)


if __name__ == "__main__":
    main()

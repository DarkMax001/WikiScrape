#!/usr/bin/env python3
"""Florr.io SmolLM2 fine-tuning pipeline for Kaggle (rate-limit-safe, resumable search).

What it does:
  1. Downloads the primary expanded-intent JSONL dataset from GitHub.
  2. Merges an existing prepared florr_training.jsonl when one is uploaded.
  3. Imports uploaded JSON files and messages.txt chat logs (plus optional JSONL).
  4. Reuses previously scraped Firecrawl exports without making new Firecrawl requests.
  5. Cleans/deduplicates the combined data and fine-tunes SmolLM2-135M-Instruct.
  6. Saves the final training dataset, source manifest, and fine-tuned model.

Configure with environment variables or CLI flags. Never put a real API key
in a public GitHub file; use Kaggle Secrets instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
import urllib.error
import zipfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Defaults / configuration
# ---------------------------------------------------------------------------

MODEL_DEFAULT = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM_PROMPT = """You are FlorrAI, an assistant specialized in the online game Florr.io.
You answer questions about petals, mobs, rarities, drops, biomes, maps,
crafting, talents, combat mechanics, strategies, updates, and community terms.
Be concise but useful. Prefer verified game-data exports over older community
pages when sources disagree. Do not invent numerical statistics, drop rates,
abilities, or mechanics. When the requested entity or fact is not supported by
your information, say that you do not know. Clearly distinguish historical
information from current game information when dates are given."""

SEARCH_QUERIES = [
    # General game knowledge and core systems
    "Florr.io complete game guide gameplay mechanics",
    "Florr.io beginner guide progression leveling XP",
    "Florr.io advanced guide endgame progression",
    "Florr.io petals complete list descriptions stats",
    "Florr.io mobs complete list descriptions stats",
    "Florr.io all rarities Common Unusual Rare Epic Legendary Mythic Ultra Super Eternal Unique",
    "Florr.io rarity scaling health damage armor petal stats",
    "Florr.io petal slot system inventory equipment",
    "Florr.io petal reload cooldown mechanics",
    "Florr.io petal health durability destruction regeneration",
    "Florr.io petal damage types and damage calculations",
    "Florr.io armor damage reduction mechanics",
    "Florr.io poison damage and poison resistance mechanics",
    "Florr.io healing and heal per second mechanics",
    "Florr.io XP experience leveling player level",
    "Florr.io player stats movement speed health damage",
    "Florr.io mob aggro targeting behavior AI",
    "Florr.io mob spawning respawn spawn rates",
    "Florr.io mob rarity spawn probabilities",
    "Florr.io mob drops drop tables drop chances",
    "Florr.io petal drop rates and rarity chances",
    "Florr.io mob damage attacks projectiles abilities",
    "Florr.io boss mobs special enemies attack patterns",
    "Florr.io player death and respawn mechanics",
    "Florr.io map generation portals and boundaries",
    "Florr.io biomes areas complete guide",
    "Florr.io crafting system recipes coins petals",
    "Florr.io trading system petal coin exchange",
    "Florr.io squad system commands and features",
    "Florr.io talents upgrades and talent points",
    "Florr.io settings controls keybinds interface",
    "Florr.io quests achievements rewards",
    "Florr.io events seasonal content updates",
    "Florr.io changelog patch notes update history",
    "Florr.io game version changes historical updates",
    "Florr.io official game client interface text commands",
    "Florr.io multiplayer server regions latency",
    "Florr.io farming strategies efficient leveling builds",
    "Florr.io best petal builds for beginners",
    "Florr.io best petal builds for bosses",
    "Florr.io best petal builds for farming",
    "Florr.io mob tier difficulty comparison",
    "Florr.io petal rarity comparison and upgrade strategy",
    "Florr.io wiki mechanics guide tips tricks",

    # Biomes and locations
    "Florr.io Garden biome mobs drops map guide",
    "Florr.io Desert biome mobs drops map guide",
    "Florr.io Ocean biome mobs drops map guide",
    "Florr.io Jungle biome mobs drops map guide",
    "Florr.io Ant Hell biome mobs drops map guide",
    "Florr.io Hel biome mobs drops map guide",
    "Florr.io Sewers biome mobs drops map guide",
    "Florr.io Factory biome mobs drops map guide",
    "Florr.io Pyramid biome mobs drops map guide",
    "Florr.io Garden map layout portals spawn point",
    "Florr.io Desert Trader location behavior",
    "Florr.io Sewers Centralia history area changes",
    "Florr.io Pyramid map walls secrets exploration",
    "Florr.io Factory machines assembler mechanics",
    "Florr.io biome-specific mob spawn tables",
    "Florr.io biome-specific petal drop tables",

    # Individual mobs and mob families represented in game data
    "Florr.io Ladybug mob health damage armor drops behavior",
    "Florr.io Bee and Bumble Bee mob stats drops behavior",
    "Florr.io Hornet mob stats attacks missile behavior",
    "Florr.io Super Hornet attack pattern Ultra Hornets missiles",
    "Florr.io Spider mob stats poison attack drops",
    "Florr.io Centipede mob stats segments drops behavior",
    "Florr.io Evil Centipede stats attacks drops",
    "Florr.io Queen Ant Soldier Ant Worker Ant mob guide",
    "Florr.io Baby Ant mob stats and behavior",
    "Florr.io Fire Ant mob family stats behavior drops",
    "Florr.io Termite and Baby Termite mob stats drops",
    "Florr.io Scorpion mob attacks stats drops",
    "Florr.io Beetle Mummy Beetle Nazar Beetle stats",
    "Florr.io Desert Centipede mob attacks drops",
    "Florr.io Dandelion mob behavior stats",
    "Florr.io Moth mob health damage armor drops",
    "Florr.io Fly mob stats and behavior",
    "Florr.io Rock mob stats behavior drops",
    "Florr.io Cactus mob stats attacks drops",
    "Florr.io Sandstorm mob behavior and attacks",
    "Florr.io Tomb mob stats attacks drops",
    "Florr.io Starfish mob stats behavior drops",
    "Florr.io Bubble mob stats behavior drops",
    "Florr.io Leafbug and Shiny Leafbug mob stats",
    "Florr.io Mecha Spider and Mecha Crab stats attacks",
    "Florr.io Nazar Beetle mob stats and drops",
    "Florr.io Trader NPC behavior and mechanics",

    # Individual petals and petal families represented in game data
    "Florr.io Leaf petal stats health damage reload",
    "Florr.io Golden Leaf petal stats abilities drop source",
    "Florr.io Dead Leaf petal stats mechanics",
    "Florr.io Stinger petal damage poison mechanics",
    "Florr.io Magic Stinger petal stats abilities",
    "Florr.io Iris petal poison mechanics stats",
    "Florr.io Wing petal stats movement mechanics",
    "Florr.io Missile and Magic Missile petal stats",
    "Florr.io Mecha Missile petal stats mechanics",
    "Florr.io Fang petal stats and usage",
    "Florr.io Ankh petal stats reload mechanics",
    "Florr.io Honey petal stats healing mechanics",
    "Florr.io Mana Orb petal stats abilities",
    "Florr.io Blueberries petal stats and number per slot",
    "Florr.io Broccoli petal stats drop source",
    "Florr.io Cogwheel petal stats damage reload",
    "Florr.io Lotus petal poison resistance stats",
    "Florr.io Plank petal reload stats",
    "Florr.io Starfish petal healing damage stats",
    "Florr.io Pollen petal stats number per slot",
    "Florr.io Light petal stats in-game description",
    "Florr.io Magic Stick petal summoned mob mechanics",
    "Florr.io Grapes petal stats and drop sources",
    "Florr.io Sponge petal stats mechanics",
    "Florr.io Bubble petal stats and abilities",
    "Florr.io petal overview descriptions and drop sources",
    "Florr.io petal rarity stats all rarity tiers",
    "Florr.io petal synergy combinations and loadouts",
    "Florr.io summon petals summoned mobs mechanics",
    "Florr.io defensive petals shields healing armor",
    "Florr.io poison petals resistance and stacking",

    # Community knowledge, strategy, and technical references
    "Florr.io community strategies farming spots player discussions",
    "Florr.io Discord guides tips player strategies",
    "Florr.io player builds loadouts discussion",
    "Florr.io rare petal farming methods community guide",
    "Florr.io how to defeat difficult mobs strategy",
    "Florr.io how to farm XP efficiently community guide",
    "Florr.io how to get coins trade petals crafting",
    "Florr.io common mistakes beginner tips",
    "Florr.io mob and petal database GitHub JSON",
    "Florr.io extracted game client data mobs petals",
    "Florr.io game files localization client UI strings",
    "Florr.io maps TMJ map data GitHub repository",
    "Florr.io game mechanics source code reverse engineering",
    "Florr.io client update snapshot version hash changes",
    "Florr.io mob AI changes historical patch notes",
    "Florr.io petal balancing changes historical updates",
    "Florr.io server regions and area map data",
    "Florr.io wiki petal pages mob pages biome pages",
]


SEED_URLS = [
    "https://florrio.wiki.gg/wiki/Florr.io",
    "https://florrio.wiki.gg/wiki/Petals",
    "https://florrio.wiki.gg/wiki/Mobs",
    "https://florrio.wiki.gg/wiki/Rarity",
    "https://florrio.wiki.gg/wiki/Strategy",
    "https://florrio.wiki.gg/wiki/Areas",
    "https://florrio.wiki.gg/wiki/Talents",
    "https://florrio.wiki.gg/wiki/Crafting",
    "https://github.com/AstRatJP/florr.io-files",
    "https://github.com/nardzy/florr-io-maps",
]


# Extra focused queries. Together with the original research list, these cover
# individual entities, mechanics, historical details, builds, and technical data.
SEARCH_QUERIES.extend([
    # Areas and biome-specific knowledge
    "Florr.io Garden mobs list spawn locations drop sources detailed wiki",
    "Florr.io Desert mobs list spawn locations drop sources detailed wiki",
    "Florr.io Ocean mobs list spawn locations drop sources detailed wiki",
    "Florr.io Jungle mobs list spawn locations drop sources detailed wiki",
    "Florr.io Ant Hell mobs list spawn locations drop sources detailed wiki",
    "Florr.io Hel mobs list spawn locations drop sources detailed wiki",
    "Florr.io Sewers mobs list spawn locations drop sources detailed wiki",
    "Florr.io Factory mobs list spawn locations drop sources detailed wiki",
    "Florr.io Pyramid mobs list spawn locations drop sources detailed wiki",
    "Florr.io biome portals transitions map connections",
    "Florr.io area difficulty progression recommended player level",
    "Florr.io biome exclusive petals and mob drops",
    "Florr.io rare mob spawn conditions biome event",
    "Florr.io Trader NPC trade inventory wandering behavior",
    "Florr.io Pyramid exploration wall patterns map generation",
    "Florr.io Factory machine recipes assembly process",
    "Florr.io Ant Hell ant nest queen spawn mechanics",
    "Florr.io Sewers Centralia map history portal mechanics",

    # Combat, stats, and mechanics
    "Florr.io damage formula mob armor player armor formula",
    "Florr.io damage scaling by rarity and mob level",
    "Florr.io petal attack interval reload formula",
    "Florr.io petal health regeneration and respawn timing",
    "Florr.io projectile speed range collision mechanics",
    "Florr.io poison tick duration damage stacking reset rules",
    "Florr.io healing over time stacking and healing petals",
    "Florr.io knockback slow stun status effects mechanics",
    "Florr.io mob aggro range chase range leash mechanics",
    "Florr.io mob collision hitboxes and contact damage",
    "Florr.io mob respawn timer and spawn density mechanics",
    "Florr.io rarity spawn weighting and rarity odds",
    "Florr.io petal drop chance independent probability rolls",
    "Florr.io player death penalty XP loss recovery",
    "Florr.io experience XP requirements level table",
    "Florr.io player movement speed and movement modifiers",
    "Florr.io petal count per slot and multi-petal behavior",
    "Florr.io summoned mobs magic stick summon AI behavior",
    "Florr.io shield defensive petal damage absorption mechanics",
    "Florr.io talent field of view movement damage upgrades",
    "Florr.io squad buffs team play revive mechanics",
    "Florr.io PvP player versus player combat mechanics",

    # Petal pages and practical uses
    "Florr.io Leaf petal all rarity stats and drop sources",
    "Florr.io Golden Leaf petal all rarity stats and drop sources",
    "Florr.io Dead Leaf petal all rarity stats and drop sources",
    "Florr.io Stinger petal all rarity stats and poison behavior",
    "Florr.io Iris petal all rarity stats and poison behavior",
    "Florr.io Wing petal all rarity stats and movement behavior",
    "Florr.io Missile petal all rarity stats and projectile behavior",
    "Florr.io Fang petal all rarity stats and drop sources",
    "Florr.io Ankh petal all rarity stats and healing behavior",
    "Florr.io Honey petal all rarity stats and healing behavior",
    "Florr.io Mana Orb petal all rarity stats and abilities",
    "Florr.io Blueberries petal stats count and rarity scaling",
    "Florr.io Broccoli petal stats and drop source",
    "Florr.io Cogwheel petal stats and reload behavior",
    "Florr.io Lotus petal stats and defensive behavior",
    "Florr.io Plank petal stats and reload behavior",
    "Florr.io Starfish petal stats healing and damage",
    "Florr.io Pollen petal stats and petal count",
    "Florr.io Light petal stats and abilities",
    "Florr.io Magic Stick petal stats summoned mobs",
    "Florr.io Grapes petal stats and drop sources",
    "Florr.io Sponge petal stats and defensive mechanics",
    "Florr.io Bubble petal stats rarity behavior",
    "Florr.io Ant Egg petal stats and summoned ants",
    "Florr.io Rice petal stats and drop sources",
    "Florr.io Rose petal healing stats rarity scaling",
    "Florr.io Egg petal stats and rarity scaling",
    "Florr.io Web petal slow effect stats mechanics",
    "Florr.io Powder petal stats and abilities",
    "Florr.io Salt petal stats and drop source",
    "Florr.io Shell petal stats and defensive use",
    "Florr.io Sand petal stats and biome source",
    "Florr.io Cactus petal stats and damage behavior",
    "Florr.io Missile variants petal comparison",
    "Florr.io petal combinations tank damage healing builds",
    "Florr.io best petals against bosses and high rarity mobs",
    "Florr.io petal farming routes by rarity and biome",

    # Mob pages and behaviors
    "Florr.io Ladybug mob all rarity stats attacks drops",
    "Florr.io Bee mob all rarity stats attacks drops",
    "Florr.io Bumble Bee mob all rarity stats attacks drops",
    "Florr.io Hornet mob all rarity stats attacks drops",
    "Florr.io Spider mob all rarity stats poison drops",
    "Florr.io Centipede mob all rarity stats segments drops",
    "Florr.io Evil Centipede mob all rarity stats attacks",
    "Florr.io Dandelion mob all rarity stats behavior drops",
    "Florr.io Digger mob stats spawn behavior drops",
    "Florr.io Rock mob all rarity stats behavior drops",
    "Florr.io Queen Ant mob stats attacks spawn mechanics",
    "Florr.io Soldier Ant mob stats attacks drops",
    "Florr.io Worker Ant mob stats behavior drops",
    "Florr.io Baby Ant mob stats behavior drops",
    "Florr.io Ant Hole mob mechanics spawn behavior",
    "Florr.io Beetle mob stats armor damage drops",
    "Florr.io Scorpion mob stats attacks poison drops",
    "Florr.io Desert Centipede mob stats behavior drops",
    "Florr.io Cactus mob stats attack patterns drops",
    "Florr.io Sandstorm mob stats attacks behavior",
    "Florr.io Nazar Beetle mob stats abilities drops",
    "Florr.io Mummy Beetle mob stats abilities drops",
    "Florr.io Moth mob stats attacks drops behavior",
    "Florr.io Fly mob stats behavior and drops",
    "Florr.io Sponge mob stats and rarity scaling",
    "Florr.io Starfish mob stats and behavior",
    "Florr.io Bubble mob stats and behavior",
    "Florr.io Leafbug mob stats rarity and drops",
    "Florr.io Golden Leafbug mob stats and behavior",
    "Florr.io Shiny Leafbug mob stats and drops",
    "Florr.io Mecha Spider mob stats attacks drops",
    "Florr.io Mecha Crab mob stats attacks drops",
    "Florr.io Tomb mob stats attacks and spawn conditions",
    "Florr.io mob family comparisons rarity stats behavior",
    "Florr.io Super mob special attacks and summon behavior",
    "Florr.io mob drop tables petal source database",

    # Updates, community and technical references
    "Florr.io complete changelog balance changes 2024 2025 2026",
    "Florr.io petal nerfs buffs historical balance changes",
    "Florr.io mob nerfs buffs historical balance changes",
    "Florr.io Hornet missile health update history",
    "Florr.io Sewers area update history Centralia",
    "Florr.io squad commands client command help strings",
    "Florr.io public squad private squad command documentation",
    "Florr.io player community strategy discussion farming",
    "Florr.io high level player builds and loadouts discussion",
    "Florr.io new player progression mistakes guide",
    "Florr.io rare and super mob farming community tips",
    "Florr.io game data JSON mobs petals GitHub repository",
    "Florr.io extracted WASM client data mob petal identifiers",
    "Florr.io map TMJ region data biome tile identifiers",
    "Florr.io server region identifiers EU US AS",
    "Florr.io game version hashes client update differences",
    "Florr.io item identifiers petal IDs mob IDs database",
    "Florr.io official client interface tooltips descriptions",
    "Florr.io wiki category pages all petals mobs areas mechanics",
])

# Remove duplicates while preserving query priority/order.
SEARCH_QUERIES = list(dict.fromkeys(SEARCH_QUERIES))


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    print(f"[FlorrAI] {message}", flush=True)


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterable[Any]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log(f"Skipping malformed JSON in {path.name}, line {line_number}")


def find_and_extract_results_dataset(input_dir: Path, data_dir: Path) -> Optional[Path]:
    """Find Results.zip under the Kaggle input mount and extract its prepared JSONL.

    Only the selected JSONL member is streamed to a fixed destination. This avoids
    extracting arbitrary archive paths and prevents ZIP path-traversal issues.
    """
    if not input_dir.exists():
        log(f"Kaggle input directory does not exist: {input_dir}")
        return None

    archives = [
        path for path in input_dir.rglob("*.zip")
        if path.is_file() and path.name.lower() == "results.zip"
    ]
    # Be tolerant of a differently-cased extension/name while still prioritizing
    # the exact Results.zip requested by the user.
    if not archives:
        archives = [
            path for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".zip"
            and "result" in path.stem.lower()
        ]

    if not archives:
        log("No Results.zip found under the input directory; continuing with normal dataset discovery.")
        return None

    archives.sort(key=lambda path: (0 if path.name.lower() == "results.zip" else 1, str(path).lower()))
    archive_path = archives[0]
    log(f"Found uploaded archive: {archive_path}")

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            candidates = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member = info.filename.replace("\\", "/")
                parts = [part for part in member.split("/") if part not in ("", ".")]
                if not parts or ".." in parts:
                    continue
                if parts[-1].lower() == "florr_training.jsonl":
                    # Prefer the expected florr_data/florr_training.jsonl location.
                    expected = 0 if [part.lower() for part in parts[-2:]] == ["florr_data", "florr_training.jsonl"] else 1
                    candidates.append((expected, len(parts), info))

            if not candidates:
                listed = [
                    info.filename for info in archive.infolist()
                    if not info.is_dir() and info.filename.lower().endswith((".jsonl", ".json"))
                ]
                preview = ", ".join(listed[:8]) if listed else "(no JSON/JSONL files found)"
                raise FileNotFoundError(
                    f"{archive_path.name} does not contain florr_training.jsonl. "
                    f"JSON/JSONL files present: {preview}. Upload the Results.zip generated "
                    "by the FlorrAI data-preparation run."
                )

            candidates.sort(key=lambda item: (item[0], item[1], item[2].filename.lower()))
            selected = candidates[0][2]
            if selected.file_size <= 0:
                raise ValueError(f"{selected.filename} inside {archive_path.name} is empty.")
            if selected.file_size > 3 * 1024 * 1024 * 1024:
                raise ValueError(
                    f"Refusing to extract unexpectedly large dataset ({selected.file_size:,} bytes)."
                )

            destination = data_dir / "florr_training.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".extracting")
            log(
                f"Extracting {selected.filename} from Results.zip "
                f"({selected.file_size:,} bytes)..."
            )
            try:
                with archive.open(selected, "r") as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if temporary.stat().st_size != selected.file_size:
                    raise IOError(
                        f"Extracted size mismatch: expected {selected.file_size:,} bytes, "
                        f"got {temporary.stat().st_size:,} bytes."
                    )
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

            log(f"Prepared dataset extracted to: {destination}")
            return destination
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"The uploaded file is not a valid ZIP archive: {archive_path}") from exc

def find_prepared_training_dataset(input_dir: Path, data_dir: Path) -> Optional[Path]:
    """Find an already-prepared florr_training.jsonl uploaded as a Kaggle Dataset.

    Kaggle mounts uploaded datasets as directories under /kaggle/input, so this
    searches recursively and copies the prepared file into the writable working
    data directory. It intentionally looks for the canonical filename to avoid
    accidentally treating raw chat exports or unrelated JSONL files as a final
    training set.
    """
    if not input_dir.exists():
        return None

    candidates = [
        path for path in input_dir.rglob("florr_training.jsonl")
        if path.is_file()
    ]
    if not candidates:
        return None

    # Prefer the canonical florr_data/florr_training.jsonl layout, then the
    # shortest path for predictable selection if multiple datasets are attached.
    candidates.sort(key=lambda path: (
        0 if [part.lower() for part in path.parts[-2:]] == ["florr_data", "florr_training.jsonl"] else 1,
        len(path.parts),
        str(path).lower(),
    ))
    source = candidates[0]
    destination = data_dir / "florr_training.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.resolve() == destination.resolve():
        return destination

    log(f"Found prepared training dataset in Kaggle input: {source}")
    temporary = destination.with_suffix(destination.suffix + ".copying")
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size <= 0:
            raise ValueError(f"Prepared dataset is empty: {source}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    log(f"Copied prepared dataset to: {destination} ({destination.stat().st_size:,} bytes).")
    return destination


def add_qa(
    target: List[dict],
    prompt: Any,
    answer: Any,
    source: str,
    fact_type: str = "general",
    negative: bool = False,
) -> None:
    prompt = normalize_text(safe_text(prompt))
    answer = normalize_text(safe_text(answer))
    if len(prompt) < 2 or len(answer) < 2:
        return
    if len(prompt) > 6000 or len(answer) > 12000:
        return
    target.append({
        "prompt": prompt,
        "answer": answer,
        "source": source,
        "fact_type": fact_type,
        "negative": bool(negative),
    })


# ---------------------------------------------------------------------------
# Download the primary GitHub JSONL dataset
# ---------------------------------------------------------------------------

DEFAULT_REMOTE_JSONL_URL = (
    "https://github.com/DarkMax001/WikiScrape/releases/download/"
    "Data/florr_qa_intent_expanded.1.jsonl"
)


def download_remote_jsonl(url: str, destination: Path, retries: int = 4) -> Optional[Path]:
    """Download the primary QA JSONL; use a previous cached copy if offline."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FlorrAI-Kaggle-Dataset/1.0"},
    )

    for attempt in range(1, retries + 1):
        try:
            log(f"Downloading primary JSONL dataset (attempt {attempt}/{retries})...")
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if not payload or len(payload) < 100:
                raise ValueError("GitHub returned an empty or unexpectedly small file.")
            text = payload.decode("utf-8-sig", errors="replace")
            valid_rows = 0
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    valid_rows += 1
                except json.JSONDecodeError:
                    continue
                if valid_rows >= 3:
                    break
            if valid_rows == 0:
                raise ValueError("Downloaded file did not contain valid JSONL records.")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            log(f"Downloaded {destination.name} ({len(payload):,} bytes; JSONL validated).")
            return destination
        except Exception as exc:
            log(f"Remote dataset download failed: {str(exc)[:220]}")
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < retries:
                time.sleep(min(2 ** attempt, 12))

    if destination.exists() and destination.stat().st_size > 100:
        log(f"Using cached copy: {destination}")
        return destination

    log("WARNING: primary GitHub dataset could not be downloaded; continuing with available uploaded inputs.")
    return None


# ---------------------------------------------------------------------------
# Import user JSON / JSONL datasets
# ---------------------------------------------------------------------------

def message_role(message: dict) -> str:
    role = message.get("role", message.get("from", message.get("author", message.get("speaker", ""))))
    if isinstance(role, dict):
        role = role.get("name", role.get("username", role.get("global_name", "")))
    role = str(role).lower().strip()
    if role in {"assistant", "bot", "ai", "model", "gpt"}:
        return "assistant"
    if role in {"user", "human", "system", "developer"}:
        return role
    # Discord exports commonly provide author objects instead of roles.
    return "user"


def message_content(message: dict) -> str:
    return safe_text(message.get("content", message.get("text", message.get("value", ""))))


def import_message_list(messages: list, target: List[dict], source: str) -> None:
    """Turn ordered chat messages into adjacent context -> assistant examples."""
    history: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = normalize_text(message_content(msg))
        if not content:
            continue
        role = message_role(msg)
        if role == "assistant":
            if history:
                prompt = "\n".join(history[-8:])
                add_qa(target, prompt, content, source, "community_conversation")
        else:
            history.append(f"{role}: {content}")


def process_record(
    record: Any,
    target: List[dict],
    source: str,
    depth: int = 0,
    context_hint: str = "",
) -> None:
    """Read QA, chat, nested JSON containers, and simple structured game data."""
    if depth > 6:
        return

    if isinstance(record, list):
        if record and all(isinstance(x, dict) for x in record):
            # Recognize an ordered chat export before treating it as unrelated rows.
            looks_like_chat = any(
                any(k in x for k in ("role", "author", "from", "speaker"))
                and any(k in x for k in ("content", "text", "value"))
                for x in record[: min(10, len(record))]
            )
            if looks_like_chat:
                import_message_list(record, target, source)
                return
        for child in record:
            process_record(child, target, source, depth + 1, context_hint)
        return

    if not isinstance(record, dict):
        return

    # Standard QA: instruction/output.
    if "instruction" in record and "output" in record:
        add_qa(
            target,
            record.get("instruction"),
            record.get("output"),
            source,
            record.get("fact_type", "general"),
            record.get("negative", False),
        )
        return

    # This pipeline's own prepared dataset uses prompt/answer (not prompt/completion).
    if "prompt" in record and "answer" in record:
        add_qa(
            target,
            record.get("prompt"),
            record.get("answer"),
            source,
            record.get("fact_type", "general"),
            record.get("negative", False),
        )
        return

    # Common QA formats.
    for q_key, a_key in (
        ("question", "answer"),
        ("prompt", "completion"),
        ("input", "output"),
        ("query", "response"),
        ("user", "assistant"),
    ):
        if q_key in record and a_key in record:
            add_qa(target, record[q_key], record[a_key], source)
            return

    # Chat/conversation arrays. Only treat the array as a flat chat when its
    # elements actually look like message turns; otherwise recurse (e.g. a
    # conversations array containing many separate QA objects).
    for messages_key in ("messages", "conversations", "conversation", "chat"):
        messages = record.get(messages_key)
        if isinstance(messages, list):
            looks_like_messages = any(
                isinstance(x, dict)
                and any(k in x for k in ("role", "author", "from", "speaker"))
                and any(k in x for k in ("content", "text", "value"))
                for x in messages[: min(10, len(messages))]
            )
            if looks_like_messages:
                import_message_list(messages, target, source)
            else:
                process_record(messages, target, source, depth + 1, messages_key)
            return

    # Some JSON files wrap rows several levels deep (data, dataset, qa, etc.).
    container_keys = (
        "data", "dataset", "examples", "records", "items", "qa",
        "questions", "training_data", "train", "samples", "results",
    )
    traversed = False
    for key in container_keys:
        child = record.get(key)
        if isinstance(child, (list, dict)):
            process_record(child, target, source, depth + 1, key)
            traversed = True
    if traversed:
        return

    # Convert simple mob/petal JSON objects into one source-labelled factual example.
    label = (
        record.get("name") or record.get("title") or record.get("displayName")
        or record.get("display_name") or record.get("sid") or record.get("slug")
        or context_hint
    )
    stat_keys = {
        "health", "hp", "damage", "armor", "reload", "reload_time",
        "description", "rarity", "drops", "drop", "biome", "area",
        "damage_type", "poison", "resistance", "heal", "speed", "xp",
        "abilities", "ability", "spawn", "spawn_chance", "drop_chance",
    }
    useful_fields = []

    def collect_scalar_fields(obj: dict, prefix: str = "", nested_depth: int = 0) -> None:
        if nested_depth > 3:
            return
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower in {"id", "_id", "sid", "slug", "source", "url", "name", "title"}:
                continue
            field_name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                collect_scalar_fields(value, field_name, nested_depth + 1)
                continue
            if isinstance(value, list):
                # Keep short lists such as drops/abilities, but avoid embedding
                # huge arrays or entire map payloads into a single training row.
                if len(value) <= 20 and all(not isinstance(v, (dict, list)) for v in value):
                    value_text = normalize_text(safe_text(value))
                else:
                    continue
            elif value is not None:
                value_text = normalize_text(safe_text(value))
            else:
                continue
            if value_text:
                useful_fields.append(f"{field_name}: {value_text[:250]}")
            if len(useful_fields) >= 20:
                return

    collect_scalar_fields(record)

    if label and useful_fields:
        label = str(label).replace("_", " ").strip()
        add_qa(
            target,
            f"What information is available about {label} in Florr.io?",
            f"Available Florr.io data for {label}: " + "; ".join(useful_fields),
            source,
            "structured_json_data",
        )
        return

    # Last-resort recursion for unrecognized nested JSON objects.
    for key, child in record.items():
        if isinstance(child, (list, dict)):
            process_record(child, target, source, depth + 1, key)


def is_excluded_dataset_file(path: Path) -> bool:
    """Exclude Hugging Face repo metadata and Firecrawl intermediate files."""
    name = path.name.lower()
    if "huggingface" in name or "hugging_face" in name:
        return True
    if name in {
        "__huggingface_repos__.json",
        "hugging face.json",
        "hugging_face.json",
        "firecrawl_search_cache.jsonl",
        "firecrawl_search_results.jsonl",
        "scraped_documents.jsonl",
    }:
        return True
    return name.startswith("firecrawl_")


def import_plain_text_chat(
    path: Path,
    target: List[dict],
    context_messages: int = 4,
) -> None:
    """Convert a one-message-per-line chat log into next-message examples.

    No author/role metadata is present in this file, so messages remain neutral:
    the preceding messages are context and the current line is the target reply.
    """
    history: List[str] = []
    added = 0
    usable_messages = 0
    window = max(1, int(context_messages))

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            message = normalize_text(raw_line)
            if len(message) < 2:
                continue

            if history:
                recent = history[-window:]
                prompt = (
                    "Continue this Florr.io community chat naturally. "
                    "The messages are from different community members. "
                    "Write only the next short chat message. Do not claim to be "
                    "one of the people in the chat. Recent messages: "
                    + " | ".join(recent)
                )
                before = len(target)
                add_qa(
                    target,
                    prompt,
                    message,
                    path.name,
                    "community_chat_next_message",
                )
                added += len(target) - before

            history.append(message)
            usable_messages += 1

    log(
        f"{path.name}: imported {added:,} next-message examples "
        f"from {usable_messages:,} non-empty messages."
    )


def load_user_datasets(
    input_dir: Path,
    extra_files: Sequence[str],
    include_input_jsonl: bool = False,
) -> Tuple[List[dict], List[dict]]:
    examples: List[dict] = []
    raw_records: List[dict] = []
    files: List[Path] = []

    # The GitHub JSONL and any prepared training dataset are passed explicitly.
    # Kaggle /kaggle/input is also scanned for user JSON files and messages.txt;
    # optional JSONL discovery remains available through --include-input-jsonl.
    if input_dir.exists():
        if include_input_jsonl:
            files.extend(
                p for p in input_dir.rglob("*.jsonl")
                if p.is_file() and not is_excluded_dataset_file(p)
            )
        files.extend(
            p for p in input_dir.rglob("*.json")
            if p.is_file() and not is_excluded_dataset_file(p)
        )
        files.extend(
            p for p in input_dir.rglob("*")
            if p.is_file() and p.name.lower() == "messages.txt"
            and not is_excluded_dataset_file(p)
        )

    for item in extra_files:
        p = Path(item)
        if p.exists() and p.is_file() and not is_excluded_dataset_file(p):
            files.append(p)
        else:
            log(f"Extra dataset path does not exist: {item}")

    # Avoid reading our own generated files if output is under the input tree.
    unique_files = sorted(set(files), key=lambda p: str(p))
    log(f"Found {len(unique_files)} JSON/JSONL/TXT input file(s).")

    for path in unique_files:
        before = len(examples)
        try:
            if path.suffix.lower() == ".txt":
                import_plain_text_chat(path, examples)
                continue
            if path.suffix.lower() == ".jsonl":
                records = read_jsonl(path)
            else:
                with path.open("r", encoding="utf-8-sig", errors="replace") as f:
                    loaded = json.load(f)
                records = loaded if isinstance(loaded, list) else [loaded]

            for record in records:
                if isinstance(record, dict):
                    raw_records.append(record)
                process_record(record, examples, path.name)
            log(f"{path.name}: added {len(examples) - before} example(s)")
        except Exception as exc:
            log(f"Could not read {path}: {exc}")

    # Resolve simple Discord-style reply_to / message_reference links.
    by_id: Dict[str, dict] = {}
    for record in raw_records:
        if record.get("id") is not None:
            by_id[str(record["id"])] = record

    for record in raw_records:
        reply = record.get("reply_to", record.get("message_reference"))
        if isinstance(reply, dict):
            reply = reply.get("message_id", reply.get("id"))
        if reply is None:
            continue
        parent = by_id.get(str(reply))
        if not parent:
            continue
        q = message_content(parent)
        a = message_content(record)
        if q and a:
            add_qa(examples, q, a, "linked_chat_reply", "community_conversation")

    return examples, raw_records


# ---------------------------------------------------------------------------
# Firecrawl research and scraping
# ---------------------------------------------------------------------------

def clean_markdown(markdown: str) -> str:
    text = str(markdown or "")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_search_web_results(result: Any) -> list:
    web = get_value(result, "web", [])
    if isinstance(web, list):
        return web
    if isinstance(web, dict):
        return web.get("results", []) or []
    return []


def is_firecrawl_rate_limit_error(exc: Exception) -> bool:
    """Return True when Firecrawl reports a per-minute/API rate limit."""
    message = str(exc).lower()
    return any(marker in message for marker in (
        "rate limit", "rate-limit", "too many requests", "http 429",
        "status code 429", "remaining (req/min)", "request limit exceeded",
    ))


def load_search_cache(cache_path: Path) -> Dict[str, list]:
    """Load completed query results so interrupted Kaggle runs can resume."""
    cached: Dict[str, list] = {}
    if not cache_path.exists():
        return cached
    for row in read_jsonl(cache_path):
        if not isinstance(row, dict) or not row.get("query"):
            continue
        if row.get("status") == "success":
            cached[str(row["query"])] = row.get("results", []) or []
    return cached


def firecrawl_research(
    api_key: str,
    data_dir: Path,
    max_search_results: int,
    max_scrapes: int,
    search_query_limit: int,
    map_limit: int,
    delay: float,
    search_delay: float = 7.0,
    max_crawl_pages: int = 3500,
    crawl_depth: int = 6,
    crawl_concurrency: int = 8,
) -> List[dict]:
    """Collect a large Florr.io corpus using search, map, deep crawl, and extra scrapes.

    Crawl is the main volume source: it follows wiki links and sitemaps instead
    of stopping at a small hand-picked list of URLs. Search/map add additional
    references and GitHub/community pages. All results are URL-deduplicated.
    """
    try:
        from firecrawl import Firecrawl
    except Exception as exc:
        log(f"Firecrawl SDK is unavailable; skipping scraping ({exc}).")
        return []

    try:
        app = Firecrawl(api_key=api_key)
    except Exception as exc:
        log(f"Could not initialize Firecrawl; skipping scraping ({exc}).")
        return []

    # Keep insertion order so direct scrapes use search-ranked URLs first.
    url_candidates: List[str] = []
    seen_urls = set()

    def add_url(url: Any) -> None:
        if not url:
            return
        url = str(url).strip()
        if not url.startswith(("http://", "https://")):
            return
        normalized = url.split("#", 1)[0].rstrip("/")
        if normalized not in seen_urls:
            seen_urls.add(normalized)
            url_candidates.append(url)

    for seed in SEED_URLS:
        add_url(seed)

    search_records: List[dict] = []
    queries = SEARCH_QUERIES[:max(0, search_query_limit)]
    cache_path = data_dir / "firecrawl_search_cache.jsonl"
    cached_searches = load_search_cache(cache_path)

    # Restore prior successful query results and their URLs. This means an
    # interrupted Kaggle run does not throw away completed search requests.
    for cached_query, cached_results in cached_searches.items():
        for item in cached_results:
            url = item.get("url") if isinstance(item, dict) else None
            if url:
                add_url(url)
                search_records.append(item)

    pending_queries = [q for q in queries if q not in cached_searches]
    log(
        f"Running {len(pending_queries):,} Firecrawl searches "
        f"({len(cached_searches):,} successful queries restored from cache); "
        f"spacing requests by {search_delay:.1f}s to respect the free/low-tier "
        f"10-requests-per-minute limit."
    )

    successful_now = 0
    rate_limit_streak = 0
    for index, query in enumerate(queries, start=1):
        if query in cached_searches:
            continue

        result_items = None
        for attempt in range(1, 4):
            try:
                result = app.search(query, limit=max_search_results)
                result_items = []
                for item in get_search_web_results(result):
                    url = get_value(item, "url")
                    title = get_value(item, "title", "")
                    description = get_value(item, "description", get_value(item, "snippet", ""))
                    if not url:
                        continue
                    low = (str(url) + " " + safe_text(title) + " " + safe_text(description)).lower()
                    if "florr" not in low and "github.com" not in str(url).lower():
                        continue
                    result_record = {
                        "query": query,
                        "title": safe_text(title),
                        "url": str(url),
                        "description": safe_text(description),
                    }
                    result_items.append(result_record)
                    add_url(url)
                    search_records.append(result_record)

                # Persist success (including empty-result searches) immediately.
                with cache_path.open("a", encoding="utf-8") as cache_file:
                    cache_file.write(json.dumps({
                        "query": query,
                        "status": "success",
                        "results": result_items,
                        "completed_at": time.time(),
                    }, ensure_ascii=False) + "\n")
                cached_searches[query] = result_items
                successful_now += 1
                rate_limit_streak = 0
                break

            except Exception as exc:
                if is_firecrawl_rate_limit_error(exc):
                    rate_limit_streak += 1
                    # A 65-second cooldown clears a one-minute rolling limit.
                    wait_seconds = 65 if attempt == 1 else 75
                    log(
                        f"Firecrawl rate limit on search {index}/{len(queries)} "
                        f"(attempt {attempt}/3). Waiting {wait_seconds}s before retry."
                    )
                    time.sleep(wait_seconds)
                    continue

                log(
                    f"Search failed ({index}/{len(queries)}) for {query!r}: "
                    f"{str(exc)[:200]}"
                )
                break

        if result_items is None and rate_limit_streak >= 3:
            log("Repeated Firecrawl rate limits; ending search phase safely. Maps and crawls will still be attempted.")
            break

        # Rate-aware pacing. This is intentionally separate from scrape delay:
        # 267 searches at 7s spacing take about 31 minutes, but avoid burning
        # requests in a tight loop and losing nearly every result to HTTP 429.
        # (Cached queries hit `continue` above and are not delayed.)
        if search_delay > 0:
            time.sleep(search_delay)

        if index % 10 == 0 or index == len(queries):
            log(
                f"Search progress: {index}/{len(queries)} query positions; "
                f"{len(cached_searches):,} cached successes total, "
                f"{len(url_candidates):,} candidate URLs."
            )

    write_jsonl(data_dir / "firecrawl_search_results.jsonl", search_records)
    log(
        f"Search phase complete: {len(cached_searches):,} successful cached queries, "
        f"{successful_now:,} newly completed this run, "
        f"{len(url_candidates):,} candidate URLs."
    )

    # Maps can return thousands of URLs in a single request. A high limit is
    # useful here because crawls and search alone do not expose every wiki page.
    map_roots = (
        "https://florrio.wiki.gg",
        "https://official-florrio.fandom.com/wiki",
    )
    for root in map_roots:
        try:
            mapped = app.map(root, limit=map_limit, sitemap="include")
            links = get_value(mapped, "links", []) or []
            for link in links:
                url = get_value(link, "url")
                if not url and isinstance(link, str):
                    url = link
                if not url:
                    continue
                low = str(url).lower()
                if any(part in low for part in (
                    "special:", "category:", "talk:", "user:", "file:",
                    "action=edit", "oldid=", "printable=yes",
                )):
                    continue
                add_url(url)
            log(f"Mapped {root}: {len(links):,} link(s) returned.")
        except Exception as exc:
            log(f"Wiki map failed for {root}: {str(exc)[:220]}")

    write_jsonl(data_dir / "firecrawl_search_results.jsonl", search_records)
    log(f"Search/map discovery produced {len(url_candidates):,} distinct candidate URL(s).")

    documents_by_url: Dict[str, dict] = {}

    def keep_document(raw_doc: Any, fallback_url: str) -> None:
        markdown = clean_markdown(get_value(raw_doc, "markdown", ""))
        if len(markdown) < 220:
            return
        metadata = get_value(raw_doc, "metadata", {}) or {}
        title = get_value(metadata, "title", "") or fallback_url.rstrip("/").split("/")[-1]
        canonical = (
            get_value(metadata, "source_url", None)
            or get_value(metadata, "sourceURL", None)
            or get_value(metadata, "url", None)
            or fallback_url
        )
        canonical = str(canonical).split("#", 1)[0].rstrip("/")
        doc = {
            "title": safe_text(title),
            "url": canonical,
            "content": markdown,
        }
        previous = documents_by_url.get(canonical)
        # If the same URL was returned twice, retain the richer extraction.
        if previous is None or len(markdown) > len(previous.get("content", "")):
            documents_by_url[canonical] = doc

    # Deep-crawl both known wiki domains. The limits are split 80/20 because
    # wiki.gg is the primary reference and the Fandom mirror can be stale.
    if max_crawl_pages > 0:
        roots = [
            ("https://florrio.wiki.gg/wiki/Florr.io", max(1, int(max_crawl_pages * 0.80))),
            ("https://official-florrio.fandom.com/wiki/Florr.io", max(1, max_crawl_pages - int(max_crawl_pages * 0.80))),
        ]
        log(
            f"Starting deep Firecrawl wiki crawl: up to {max_crawl_pages:,} pages total, "
            f"depth {crawl_depth}, concurrency {crawl_concurrency}."
        )
        for root, page_limit in roots:
            try:
                crawl_kwargs = {
                    "limit": page_limit,
                    "max_discovery_depth": crawl_depth,
                    "sitemap": "include",
                    "crawl_entire_domain": True,
                    "allow_subdomains": False,
                    "max_concurrency": crawl_concurrency,
                    "formats": ["markdown"],
                    "only_main_content": True,
                    "timeout": 7200,
                }
                # Current Firecrawl SDK's crawl() waits and auto-paginates the
                # completed response. A fallback accommodates older SDK builds.
                try:
                    crawl_result = app.crawl(root, **crawl_kwargs)
                except TypeError:
                    crawl_result = app.crawl(
                        root,
                        limit=page_limit,
                        formats=["markdown"],
                        only_main_content=True,
                    )
                crawl_data = get_value(crawl_result, "data", []) or []
                count_before = len(documents_by_url)
                for raw_doc in crawl_data:
                    metadata = get_value(raw_doc, "metadata", {}) or {}
                    fallback = (
                        get_value(metadata, "source_url", None)
                        or get_value(metadata, "sourceURL", None)
                        or root
                    )
                    keep_document(raw_doc, str(fallback))
                log(
                    f"Crawl finished for {root}: {len(crawl_data):,} page result(s), "
                    f"{len(documents_by_url) - count_before:,} new usable document(s)."
                )
            except Exception as exc:
                log(f"Deep crawl failed for {root}: {str(exc)[:260]}")

    # Scrape extra search/map discoveries outside of the main wiki crawls.
    # This is capped separately so a very large map result doesn't explode cost.
    # Scrape any discovered URL not already returned by crawl. This also acts
    # as a fallback for wiki URLs when a domain crawl is blocked or incomplete.
    extra_urls = []
    for url in url_candidates:
        normalized = url.split("#", 1)[0].rstrip("/")
        if normalized not in documents_by_url:
            extra_urls.append(url)
    selected_urls = extra_urls[:max(0, max_scrapes)]
    log(
        f"Scraping up to {len(selected_urls):,} additional search/map/GitHub URL(s) "
        f"after the deep wiki crawl..."
    )

    for index, url in enumerate(selected_urls, start=1):
        try:
            doc = app.scrape(
                url,
                formats=["markdown"],
                only_main_content=True,
            )
            keep_document(doc, url)
        except Exception as exc:
            log(f"Extra scrape failed ({index}/{len(selected_urls)}): {url} :: {str(exc)[:160]}")
        if delay:
            time.sleep(delay)
        if index % 25 == 0:
            log(f"Extra-scraped {index}/{len(selected_urls)} URLs; usable unique documents: {len(documents_by_url):,}")

    documents = list(documents_by_url.values())
    write_jsonl(data_dir / "scraped_documents.jsonl", documents)
    total_chars = sum(len(x.get("content", "")) for x in documents)
    log(
        f"Firecrawl complete: {len(documents):,} unique usable documents, "
        f"{total_chars:,} extracted characters."
    )
    return documents


# ---------------------------------------------------------------------------
# Turn scraped Markdown into grounded training examples
# ---------------------------------------------------------------------------

def markdown_sections(content: str) -> List[Tuple[str, str]]:
    lines = content.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_heading = "Overview"
    current_lines: List[str] = []

    for line in lines:
        match = re.match(r"^#{1,4}\s+(.+?)\s*$", line)
        if match:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    result = []
    for heading, section_lines in sections:
        text = "\n".join(section_lines).strip()
        if len(text) >= 180:
            result.append((heading, text))
    return result


def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        # Keep one oversized paragraph in several pieces instead of discarding it.
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(paragraph), max_chars):
                piece = paragraph[start:start + max_chars].strip()
                if len(piece) >= 150:
                    chunks.append(piece)
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = candidate

    if current and len(current) >= 150:
        chunks.append(current.strip())
    return chunks


def documents_to_examples(
    documents: List[dict],
    qa_variants_per_chunk: int = 5,
) -> List[dict]:
    """Turn each source-grounded chunk into several differently phrased QA rows.

    These are paraphrased questions over the same extracted passage, not newly
    invented facts. Every record retains the exact source URL for provenance.
    """
    result: List[dict] = []
    question_templates = [
        "Explain {heading} in Florr.io using the reference page '{title}'.",
        "What does the Florr.io reference say about {heading}?",
        "Summarize the documented information about {heading} in Florr.io.",
        "I need details about {heading} in Florr.io. What is known from the source?",
        "According to the Florr.io page '{title}', what should I know about {heading}?",
        "Give me the important reference notes for {heading} in Florr.io.",
        "Can you describe {heading} based on the Florr.io documentation?",
    ]
    variant_count = max(1, min(int(qa_variants_per_chunk), len(question_templates)))

    for doc in documents:
        title = doc.get("title", "Florr.io reference")
        url = doc.get("url", "")
        for heading, section in markdown_sections(doc.get("content", "")):
            for chunk in chunk_text(section, max_chars=1700):
                # Sample different phrasings per chunk so the same question
                # wording does not dominate the expanded web corpus.
                templates = random.sample(question_templates, variant_count)
                for template in templates:
                    question = template.format(
                        heading=heading[:180],
                        title=title[:180],
                    )
                    add_qa(
                        result,
                        question,
                        chunk,
                        url or title,
                        "scraped_reference",
                    )
    return result


def load_existing_scraped_documents(input_dir: Path) -> List[dict]:
    """Load previously scraped Firecrawl pages uploaded as a Kaggle input dataset.

    Search caches/results are not training documents. This function imports only
    saved page records from files named scraped_documents.jsonl and normalizes
    their text into the same title/url/content shape used by Firecrawl itself.
    """
    if not input_dir.exists():
        return []

    candidates = sorted(
        (path for path in input_dir.rglob("*")
         if path.is_file() and path.name.lower() == "scraped_documents.jsonl"),
        key=lambda path: str(path).lower(),
    )
    documents_by_url: Dict[str, dict] = {}

    for path in candidates:
        imported = 0
        for record in read_jsonl(path):
            if not isinstance(record, dict):
                continue
            raw_content = record.get("content") or record.get("markdown") or record.get("text") or ""
            content = clean_markdown(safe_text(raw_content))
            if len(content) < 220:
                continue

            metadata = record.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            url = (
                record.get("url")
                or record.get("source_url")
                or record.get("sourceURL")
                or metadata.get("source_url")
                or metadata.get("sourceURL")
                or metadata.get("url")
                or ""
            )
            title = (
                record.get("title")
                or metadata.get("title")
                or (str(url).rstrip("/").split("/")[-1] if url else path.stem)
            )
            canonical = str(url).split("#", 1)[0].rstrip("/") if url else f"{path}#{imported}"
            document = {
                "title": safe_text(title),
                "url": canonical,
                "content": content,
            }
            previous = documents_by_url.get(canonical)
            if previous is None or len(content) > len(previous.get("content", "")):
                documents_by_url[canonical] = document
            imported += 1

        log(f"Loaded {imported:,} saved Firecrawl page(s) from {path}.")

    documents = list(documents_by_url.values())
    log(f"Total saved Firecrawl documents imported from Kaggle input: {len(documents):,}.")
    return documents


def merge_scraped_documents(*document_sets: List[dict]) -> List[dict]:
    """Merge page collections by canonical URL, keeping the richest content."""
    merged: Dict[str, dict] = {}
    for group in document_sets:
        for document in group:
            if not isinstance(document, dict):
                continue
            url = str(document.get("url", "")).split("#", 1)[0].rstrip("/")
            if not url:
                url = f"untitled:{document.get('title', '')}"
            candidate = dict(document)
            candidate["url"] = url
            previous = merged.get(url)
            if previous is None or len(str(candidate.get("content", ""))) > len(str(previous.get("content", ""))):
                merged[url] = candidate
    return list(merged.values())


# ---------------------------------------------------------------------------
# Cleaning, deduplication and dataset files
# ---------------------------------------------------------------------------

def deduplicate_examples(examples: List[dict]) -> List[dict]:
    seen = set()
    cleaned: List[dict] = []
    for item in examples:
        prompt = normalize_text(item.get("prompt", ""))
        answer = normalize_text(item.get("answer", ""))
        # Deduplicate exact prompt-answer pairs; retain distinct answers for review.
        key = hashlib.sha256((prompt.casefold() + "\0" + answer.casefold()).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        if prompt and answer:
            item = dict(item)
            item["prompt"] = prompt
            item["answer"] = answer
            cleaned.append(item)
    return cleaned


def print_dataset_stats(examples: List[dict]) -> None:
    log(f"Total examples: {len(examples):,}")
    log("Examples by type:")
    for name, count in Counter(x.get("fact_type", "general") for x in examples).most_common():
        log(f"  {name}: {count:,}")
    log(f"Negative examples: {sum(bool(x.get('negative')) for x in examples):,}")


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def train_model(examples: List[dict], args: argparse.Namespace, output_dir: Path) -> None:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install transformers, datasets, "
            "accelerate, torch and sentencepiece."
        ) from exc

    class TrainingTimeLimitCallback(TrainerCallback):
        """Stop Trainer after a wall-clock limit, checked after optimizer steps."""

        def __init__(self, max_hours: float):
            super().__init__()
            self.max_seconds = max(0.0, float(max_hours)) * 3600.0
            self.started_at: Optional[float] = None
            self.elapsed_seconds = 0.0
            self.time_limit_reached = False

        def on_train_begin(self, args, state, control, **kwargs):
            self.started_at = time.monotonic()
            if self.max_seconds > 0:
                log(f"Training time limit: {self.max_seconds / 3600:.2f} hours (training loop only).")
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if self.started_at is None:
                return control
            self.elapsed_seconds = time.monotonic() - self.started_at
            if self.max_seconds > 0 and self.elapsed_seconds >= self.max_seconds:
                self.time_limit_reached = True
                control.should_training_stop = True
                log(
                    f"Training time limit reached after {self.elapsed_seconds / 3600:.2f} hours; "
                    "stopping after the current optimizer step and saving the model."
                )
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if self.started_at is not None:
                self.elapsed_seconds = time.monotonic() - self.started_at
            log(f"Training loop elapsed: {self.elapsed_seconds / 3600:.2f} hours.")
            return control

    class PeriodicCheckpointCallback(TrainerCallback):
        """Request a resumable Trainer checkpoint at a wall-clock interval."""

        def __init__(self, interval_minutes: float):
            super().__init__()
            if interval_minutes <= 0:
                raise ValueError("--save-interval-minutes must be greater than 0")
            self.interval_seconds = float(interval_minutes) * 60.0
            self.last_checkpoint_at: Optional[float] = None
            self.save_requested = False

        def on_train_begin(self, args, state, control, **kwargs):
            self.last_checkpoint_at = time.monotonic()
            log(
                f"Periodic checkpointing enabled: every "
                f"{self.interval_seconds / 60:g} minute(s). "
                "Only the newest checkpoint will be retained."
            )
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if self.last_checkpoint_at is None or self.save_requested:
                return control
            elapsed = time.monotonic() - self.last_checkpoint_at
            if elapsed >= self.interval_seconds:
                control.should_save = True
                self.save_requested = True
                log(
                    f"Checkpoint interval reached ({elapsed / 60:.1f} min); "
                    f"saving after optimizer step {state.global_step}."
                )
            return control

        def on_save(self, args, state, control, **kwargs):
            self.last_checkpoint_at = time.monotonic()
            self.save_requested = False
            log(f"Checkpoint saved at global step {state.global_step}.")
            return control

    if not examples:
        raise RuntimeError("No training examples were loaded. Attach a JSONL dataset or scrape sources.")

    if len(examples) < 10:
        raise RuntimeError(f"Only {len(examples)} examples were found; provide more training data.")

    log(f"Loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    # Keep trainable parameters in FP32. Trainer/Accelerate applies FP16
    # autocast during the forward pass on CUDA. Loading the model itself in
    # FP16 while also enabling TrainingArguments(fp16=True) can leave FP16
    # gradients for GradScaler to unscale, causing:
    # "Attempting to unscale FP16 gradients."
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(dtype=torch.float32)
    model.config.use_cache = False
    if use_cuda:
        log("Model weights loaded as FP32; Trainer will use FP16 autocast on the Tesla T4.")
    else:
        log("Model weights loaded as FP32; training will use full precision on CPU.")

    formatted = []
    for item in examples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["answer"]},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "The selected tokenizer has no working chat template. "
                "Use SmolLM2-135M-Instruct, not the base checkpoint."
            ) from exc
        formatted.append({"text": text})

    raw_dataset = Dataset.from_list(formatted)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )

    tokenized = raw_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing Florr.io conversations",
    )
    tokenized = tokenized.filter(lambda row: len(row["input_ids"]) >= 8)

    if len(tokenized) < 10:
        raise RuntimeError("Too few usable tokenized examples remain after preprocessing.")

    split = tokenized.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    log(f"Training examples: {len(train_dataset):,}")
    log(f"Evaluation examples: {len(eval_dataset):,}")
    log(f"GPU: {torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if use_cuda else None,
    )

    # Transformers has changed several TrainingArguments parameter names and
    # availability across releases. Build arguments from the installed
    # version's actual constructor signature instead of assuming a new version.
    import inspect
    import math
    import transformers

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        eval_key = "eval_strategy"
    elif "evaluation_strategy" in ta_params:
        eval_key = "evaluation_strategy"
    else:
        raise RuntimeError(
            f"Installed Transformers {transformers.__version__} does not expose "
            "eval_strategy or evaluation_strategy in TrainingArguments. "
            "Upgrade with: pip install -U transformers accelerate"
        )

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        # This is a time-bounded run, not a fixed-epoch run. A constant
        # scheduler with a short warmup avoids stretching the warmup over the
        # deliberately high epoch ceiling used to keep training alive for 7h.
        "lr_scheduler_type": "constant_with_warmup",
        eval_key: "steps",
        # Saving is triggered by PeriodicCheckpointCallback using wall-clock
        # time, not by epochs or a fixed number of optimizer steps.
        "save_strategy": "no",
        "eval_steps": 1000,
        "logging_strategy": "steps",
        "logging_steps": 25,
        "fp16": bool(use_cuda and not args.no_fp16),
        "bf16": False,
        "gradient_checkpointing": bool(use_cuda and not args.no_gradient_checkpointing),
        "optim": "adamw_torch",
        # Keep only the newest five-minute checkpoint in output_dir.
        "save_total_limit": 1,
        # Custom wall-clock checkpointing replaces best-checkpoint selection.
        "load_best_model_at_end": False,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": args.num_workers,
    }

    # The time limit, rather than a small epoch count, is the primary stop
    # condition. Keep warmup short so a 1000-epoch ceiling does not make the
    # learning rate spend the entire 7-hour run warming up.
    if "warmup_steps" in ta_params:
        training_kwargs["warmup_steps"] = 100
        log("Using constant_with_warmup scheduler with 100 warmup steps.")
    elif "warmup_ratio" in ta_params:
        training_kwargs["warmup_ratio"] = 0.0001
        log("Using constant_with_warmup scheduler with a short warmup ratio.")
    else:
        log("WARNING: Installed Transformers supports no warmup option; continuing without warmup.")

    # Drop only parameters absent from this installed Transformers version.
    # This also prevents the same kind of TypeError for other newer options.
    unsupported = [key for key in training_kwargs if key not in ta_params]
    for key in unsupported:
        training_kwargs.pop(key)
    if unsupported:
        log(
            f"Transformers {transformers.__version__} does not support these "
            f"TrainingArguments options; skipping them: {', '.join(unsupported)}"
        )

    log(f"Using Transformers {transformers.__version__} TrainingArguments compatibility mode.")
    training_args = TrainingArguments(**training_kwargs)
    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    checkpoint_callback = PeriodicCheckpointCallback(args.save_interval_minutes)
    trainer.add_callback(checkpoint_callback)

    time_limit_callback = None
    if args.max_training_hours > 0:
        time_limit_callback = TrainingTimeLimitCallback(args.max_training_hours)
        trainer.add_callback(time_limit_callback)
    log("Starting fine-tuning...")
    train_result = trainer.train()

    model.config.use_cache = True
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "model": args.model,
        "specialization": "Florr.io",
        "total_examples": len(examples),
        "training_examples": len(train_dataset),
        "evaluation_examples": len(eval_dataset),
        "max_length": args.max_length,
        "epochs_requested": args.epochs,
        "max_training_hours": args.max_training_hours,
        "save_interval_minutes": args.save_interval_minutes,
        "training_loop_elapsed_hours": (
            time_limit_callback.elapsed_seconds / 3600.0
            if time_limit_callback is not None else None
        ),
        "training_time_limit_reached": (
            time_limit_callback.time_limit_reached
            if time_limit_callback is not None else False
        ),
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "training_metrics": train_result.metrics,
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    # Quick post-training smoke test.
    log("Running a small generation smoke test...")
    model.eval()
    test_questions = [
        "What is a Super Leaf in Florr.io?",
        "Which mobs spawn in the Garden?",
        "What is the health of an Epic Fang?",
        "What is a Titan Crab petal?",
    ]
    for question in test_questions:
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    repetition_penalty=1.08,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output[0, inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            log(f"Q: {question}\nA: {answer}")
        except Exception as exc:
            log(f"Generation test failed: {exc}")

    log(f"Fine-tuning complete. Model saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM2 for Florr.io")
    parser.add_argument("--input-dir", default=os.environ.get("FLORR_INPUT_DIR", "/kaggle/input"))
    parser.add_argument("--extra-files", nargs="*", default=[])
    parser.add_argument("--remote-jsonl-url", default=os.environ.get("FLORR_REMOTE_JSONL_URL", DEFAULT_REMOTE_JSONL_URL))
    parser.add_argument("--skip-remote-jsonl", action="store_true", help="Do not download the primary GitHub JSONL dataset")
    parser.add_argument("--include-input-jsonl", action="store_true", help="Also import all JSONL files attached under /kaggle/input")
    parser.add_argument("--data-dir", default=os.environ.get("FLORR_DATA_DIR", "/kaggle/working/florr_data"))
    parser.add_argument("--output-dir", default=os.environ.get("FLORR_OUTPUT_DIR", "/kaggle/working/florr-smollm2-135m"))
    parser.add_argument("--model", default=os.environ.get("FLORR_MODEL", MODEL_DEFAULT))

    parser.add_argument("--skip-scrape", action="store_true", help="Compatibility flag; fresh Firecrawl is disabled in this file")
    parser.add_argument(
        "--reuse-training-data",
        action="store_true",
        help="Train directly from data-dir/florr_training.jsonl without downloading or scraping again",
    )
    parser.add_argument(
        "--no-auto-results-zip",
        dest="auto_results_zip",
        action="store_false",
        default=True,
        help="Disable automatic detection/extraction of Results.zip under the input directory",
    )
    parser.add_argument("--skip-train", action="store_true", help="Only build and save the dataset")
    parser.add_argument("--max-search-results", type=int, default=int(os.environ.get("FLORR_SEARCH_RESULTS", "10")))
    parser.add_argument("--search-query-limit", type=int, default=int(os.environ.get("FLORR_SEARCH_QUERIES", str(len(SEARCH_QUERIES)))))
    parser.add_argument("--max-scrapes", type=int, default=int(os.environ.get("FLORR_MAX_SCRAPES", "2000")))
    parser.add_argument("--map-limit", type=int, default=int(os.environ.get("FLORR_MAP_LIMIT", "15000")))
    parser.add_argument("--max-crawl-pages", type=int, default=int(os.environ.get("FLORR_CRAWL_PAGES", "5000")))
    parser.add_argument("--crawl-depth", type=int, default=int(os.environ.get("FLORR_CRAWL_DEPTH", "6")))
    parser.add_argument("--crawl-concurrency", type=int, default=int(os.environ.get("FLORR_CRAWL_CONCURRENCY", "8")))
    parser.add_argument("--qa-variants-per-chunk", type=int, default=int(os.environ.get("FLORR_QA_VARIANTS", "5")))
    parser.add_argument("--web-data-ratio", type=float, default=float(os.environ.get("FLORR_WEB_DATA_RATIO", "3.00")))
    parser.add_argument("--max-web-examples", type=int, default=int(os.environ.get("FLORR_MAX_WEB_EXAMPLES", "250000")))
    parser.add_argument(
        "--request-delay", type=float,
        default=float(os.environ.get("FLORR_REQUEST_DELAY", "0.35")),
        help="Pause between individual extra scrape requests (seconds).",
    )
    parser.add_argument(
        "--search-delay", type=float,
        default=float(os.environ.get("FLORR_SEARCH_DELAY", "7.0")),
        help="Pause between Firecrawl search calls; 7s stays below a 10-requests/minute limit.",
    )

    parser.add_argument("--epochs", type=float, default=float(os.environ.get("FLORR_EPOCHS", "1000")), help="Safety ceiling for epochs; the 7-hour time limit normally stops training first")
    parser.add_argument("--max-training-hours", type=float, default=float(os.environ.get("FLORR_MAX_TRAINING_HOURS", "7")), help="Stop fine-tuning after this many hours; 0 disables the time limit")
    parser.add_argument(
        "--save-interval-minutes",
        type=float,
        default=float(os.environ.get("FLORR_SAVE_INTERVAL_MINUTES", "5")),
        help="Save a resumable checkpoint every N minutes (default: 5).",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("FLORR_BATCH_SIZE", "4")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("FLORR_EVAL_BATCH_SIZE", "4")))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=int(os.environ.get("FLORR_GRAD_ACCUM", "8")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("FLORR_LR", "1e-5")))
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("FLORR_MAX_LENGTH", "768")))
    parser.add_argument("--eval-fraction", type=float, default=float(os.environ.get("FLORR_EVAL_FRACTION", "0.05")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("FLORR_NUM_WORKERS", "2")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("FLORR_SEED", "42")))
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Florr.io SmolLM2 fine-tuning pipeline")
    log(f"Primary expanded-intent dataset: {args.remote_jsonl_url}")
    log(f"Input directory: {input_dir}")
    log(f"Data directory: {data_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Base model: {args.model}")
    log("Fresh Firecrawl search, map, crawl, and scrape requests are disabled.")
    if args.max_training_hours > 0:
        log(f"Fine-tuning will stop after at most {args.max_training_hours:g} training hours.")

    # Kaggle convenience: detect an uploaded Results.zip or prepared JSONL.
    # It is merged as another dataset source. Fresh Firecrawl requests are disabled.
    prepared_input_dataset: Optional[Path] = None
    if args.auto_results_zip and not args.reuse_training_data:
        # Prefer Results.zip when supplied. Otherwise accept a normal Kaggle
        # Dataset containing florr_training.jsonl directly (no ZIP required).
        prepared_input_dataset = find_and_extract_results_dataset(input_dir, data_dir)
        if prepared_input_dataset is None:
            prepared_input_dataset = find_prepared_training_dataset(input_dir, data_dir)
        if prepared_input_dataset is not None:
            log(
                "Will MERGE the existing prepared dataset with the remote JSONL, "
                "uploaded datasets, messages.txt, and Firecrawl results."
            )

    # Reuse the already-built combined dataset. This is useful after a previous
    # run completed scraping/data preparation but failed before training.
    if args.reuse_training_data:
        saved_dataset = data_dir / "florr_training.jsonl"
        if not saved_dataset.is_file():
            raise FileNotFoundError(
                f"Cannot reuse dataset: {saved_dataset} does not exist. "
                "Restore florr_training.jsonl into the data directory first."
            )
        reused_examples: List[dict] = []
        for record in read_jsonl(saved_dataset):
            if not isinstance(record, dict):
                continue
            add_qa(
                reused_examples,
                record.get("prompt", record.get("instruction", record.get("question", ""))),
                record.get("answer", record.get("output", record.get("completion", ""))),
                str(record.get("source", "saved_training_dataset")),
                str(record.get("fact_type", "general")),
                record.get("negative", False),
            )
        if not reused_examples:
            raise RuntimeError(f"No valid prompt/answer rows found in {saved_dataset}")
        log(f"Reusing saved dataset: {saved_dataset}")
        log(f"Loaded {len(reused_examples):,} saved examples; skipping download and Firecrawl.")
        print_dataset_stats(reused_examples)
        train_model(reused_examples, args, output_dir)
        log("Training from the saved dataset completed successfully.")
        return 0

    # 1) Download the primary GitHub JSONL source, then combine it with the
    #    prepared dataset and user-uploaded JSON/chat inputs under /kaggle/input.
    primary_jsonl = None
    if not args.skip_remote_jsonl:
        primary_jsonl = download_remote_jsonl(
            args.remote_jsonl_url,
            data_dir / "source_datasets" / "florr_qa_intent_expanded.1.jsonl",
        )
    else:
        log("Primary GitHub JSONL download disabled by --skip-remote-jsonl.")

    explicit_files = list(args.extra_files)
    if primary_jsonl is not None:
        explicit_files.append(str(primary_jsonl))
    if prepared_input_dataset is not None:
        explicit_files.append(str(prepared_input_dataset))

    user_examples, _raw_records = load_user_datasets(
        input_dir,
        explicit_files,
        include_input_jsonl=args.include_input_jsonl,
    )
    log(
        f"Imported {len(user_examples):,} examples from the GitHub JSONL, "
        "prepared dataset, uploaded JSON files, and chat text files."
    )

    # 2) Load saved Firecrawl documents already attached as a Kaggle input.
    # Search caches/results are metadata; scraped_documents.jsonl contains the
    # actual page content and is imported through this dedicated document path.
    documents: List[dict] = load_existing_scraped_documents(input_dir)

    # 3) Fresh Firecrawl is intentionally disabled. Reuse only documents
    # already present in the attached Kaggle input datasets.
    log("Reusing saved Firecrawl documents only; no Firecrawl API requests will be made.")

    # Keep one combined copy in the working output for reproducibility.
    if documents:
        write_jsonl(data_dir / "scraped_documents.jsonl", documents)

    # 4) Convert web articles into chunks and QA training rows.
    scraped_examples = documents_to_examples(documents, args.qa_variants_per_chunk)
    scraped_examples_generated_before_cap = len(scraped_examples)
    # Default: target roughly one Firecrawl row per primary dataset row.
    # With 90k primary rows this allows up to ~90k Firecrawl rows, making web
    # material about half of the combined set. Actual volume still depends on
    # the number/quality of pages Firecrawl successfully returns.
    ratio = max(0.0, args.web_data_ratio)
    if user_examples:
        web_cap = max(0, int(len(user_examples) * ratio))
    else:
        web_cap = args.max_web_examples
    web_cap = min(web_cap, max(0, args.max_web_examples))
    random.shuffle(scraped_examples)
    if len(scraped_examples) > web_cap:
        scraped_examples = scraped_examples[:web_cap]
    log(f"Using {len(scraped_examples):,} scraped-reference example(s).")

    # 5) Combine, deduplicate, persist.
    all_examples = deduplicate_examples(user_examples + scraped_examples)
    random.shuffle(all_examples)
    write_jsonl(data_dir / "florr_training.jsonl", all_examples)
    manifest = {
        "primary_github_jsonl_url": args.remote_jsonl_url if not args.skip_remote_jsonl else None,
        "primary_github_jsonl_downloaded": bool(primary_jsonl),
        "uploaded_input_directory": str(input_dir),
        "uploaded_normal_json_files": sorted(
            p.name for p in input_dir.rglob("*.json")
            if p.is_file() and not is_excluded_dataset_file(p)
        ) if input_dir.exists() else [],
        "uploaded_chat_text_files": sorted(
            p.name for p in input_dir.rglob("*")
            if p.is_file() and p.name.lower() == "messages.txt"
        ) if input_dir.exists() else [],
        "input_jsonl_scan_enabled": bool(args.include_input_jsonl),
        "firecrawl_query_count_configured": min(args.search_query_limit, len(SEARCH_QUERIES)),
        "firecrawl_scraped_documents": len(documents),
        "firecrawl_examples_generated_before_cap": scraped_examples_generated_before_cap,
        "firecrawl_examples_used": len(scraped_examples),
        "firecrawl_target_ratio": args.web_data_ratio,
        "firecrawl_max_examples": args.max_web_examples,
        "firecrawl_crawl_page_limit": args.max_crawl_pages,
        "firecrawl_qa_variants_per_chunk": args.qa_variants_per_chunk,
        "user_examples_used": len(user_examples),
        "final_examples": len(all_examples),
    }
    with (data_dir / "source_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print_dataset_stats(all_examples)

    if not all_examples:
        log("No examples found. Check the GitHub JSONL download and upload your normal JSON dataset to /kaggle/input/.")
        return 2

    # Print a few examples for a quick data sanity check.
    log("Sample training records:")
    for sample in random.sample(all_examples, min(3, len(all_examples))):
        log(f"USER: {sample['prompt'][:220]}")
        log(f"ASSISTANT: {sample['answer'][:350]}")

    if args.skip_train:
        log(f"Dataset-only run complete: {data_dir / 'florr_training.jsonl'}")
        return 0

    # 6) Fine-tune and export.
    train_model(all_examples, args, output_dir)
    log("All tasks completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        log(f"FATAL: {exc}")
        traceback.print_exc()
        raise SystemExit(1)

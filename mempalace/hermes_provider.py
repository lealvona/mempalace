"""Hermes Agent memory provider plugin for MemPalace — v2 (full integration).

Integrates MemPalace's local-first verbatim memory into Hermes Agent as a
MemoryProvider plugin. This v2 implementation uses the full MemPalace stack:

- AAAK compression for 30x index density
- Entity detection + registry for automatic people/project identification
- Knowledge graph with temporal triples and contradiction detection
- L0-L3 layered wake-up for progressive session initialization
- Room auto-detection for intelligent content categorization
- Palace graph tunnels for cross-wing connections
- Conversation miner for bootstrapping from past sessions
- Fact checker to validate knowledge graph writes

No API keys required for core operations. Everything runs locally via
ChromaDB + SQLite + ONNX embeddings.

Config:
  Set memory.provider to "mempalace" in config.yaml.
  Optional env vars:
    MEMPALACE_PATH  - path to palace directory (default: ~/.mempalace)
    MEMPALACE_WING   - default wing for Hermes entries (default: "hermes")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (8 tools)
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "mempalace_search",
    "description": (
        "Search MemPalace for verbatim memories across all wings and rooms. "
        "Returns exact stored text ranked by hybrid BM25 + vector search with "
        "closet index boosting. Use this to recall specific past conversations, "
        "facts, or context. Optional: filter by wing or room."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Max results (default 5, max 20).", "default": 5},
            "wing": {"type": "string", "description": "Filter by wing (person/project category)."},
            "room": {"type": "string", "description": "Filter by room (topic/day)."},
        },
        "required": ["query"],
    },
}

ADD_DRAWER_SCHEMA = {
    "name": "mempalace_add_drawer",
    "description": (
        "Store a verbatim memory in MemPalace. Content is preserved exactly. "
        "Auto-detects room if not specified. AAAK-compressed index entries are "
        "generated automatically for fast retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wing": {"type": "string", "description": "Wing (e.g. 'hermes', 'people'). Default: 'hermes'.", "default": "hermes"},
            "room": {"type": "string", "description": "Room. Omit for auto-detection from content."},
            "content": {"type": "string", "description": "Text to store verbatim."},
        },
        "required": ["content"],
    },
}

STATUS_SCHEMA = {
    "name": "mempalace_status",
    "description": "Palace overview: total entries, wings, rooms, knowledge graph stats, graph health.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

KG_QUERY_SCHEMA = {
    "name": "mempalace_kg_query",
    "description": (
        "Query the knowledge graph for structured facts about an entity. "
        "Returns temporal triples with validity dates. Supports contradiction "
        "detection against new claims."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity name to look up."},
            "direction": {"type": "string", "description": "'outgoing', 'incoming', or 'both'.", "default": "both", "enum": ["outgoing", "incoming", "both"]},
        },
        "required": ["entity"],
    },
}

KG_ADD_SCHEMA = {
    "name": "mempalace_kg_add",
    "description": (
        "Add a structured fact as a temporal triple. Runs fact checker first "
        "to detect contradictions with existing knowledge. Invalidates "
        "conflicting facts automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Subject entity."},
            "predicate": {"type": "string", "description": "Relationship (e.g. 'works_at', 'prefers')."},
            "object": {"type": "string", "description": "Object entity or value."},
        },
        "required": ["subject", "predicate", "object"],
    },
}

DIARY_WRITE_SCHEMA = {
    "name": "mempalace_diary_write",
    "description": "Write a timestamped diary entry for longitudinal self-reflection.",
    "parameters": {
        "type": "object",
        "properties": {
            "entry": {"type": "string", "description": "Diary entry text."},
            "topic": {"type": "string", "description": "Optional topic tag."},
        },
        "required": ["entry"],
    },
}

LIST_WINGS_SCHEMA = {
    "name": "mempalace_list_wings",
    "description": "List all wings and their drawer counts.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

LIST_ROOMS_SCHEMA = {
    "name": "mempalace_list_rooms",
    "description": "List rooms within a wing, or all rooms.",
    "parameters": {
        "type": "object",
        "properties": {
            "wing": {"type": "string", "description": "Wing name. Omit for all."},
        },
        "required": [],
    },
}

ALL_TOOL_SCHEMAS = [
    SEARCH_SCHEMA, ADD_DRAWER_SCHEMA, STATUS_SCHEMA,
    KG_QUERY_SCHEMA, KG_ADD_SCHEMA, DIARY_WRITE_SCHEMA,
    LIST_WINGS_SCHEMA, LIST_ROOMS_SCHEMA,
]


# ---------------------------------------------------------------------------
# Room auto-detection keyword scoring
# ---------------------------------------------------------------------------

_ROOM_KEYWORDS = {
    "decisions": ["decided", "decision", "chose", "chosen", "will use", "going with", "settled on", "approved"],
    "preferences": ["prefer", "likes", "wants", "always", "never", "favorite", "style", "format"],
    "technical": ["config", "install", "setup", "deploy", "debug", "error", "fix", "api", "server", "code"],
    "projects": ["project", "working on", "building", "feature", "release", "milestone", "deadline"],
    "people": ["met with", "works at", "reports to", "manages", "colleague", "team", "said"],
    "environment": ["installed", "version", "system", "environment", "server", "host", "node", "tailscale"],
    "issues": ["problem", "broken", "failing", "bug", "issue", "error", "doesn't work", "not working"],
}


def _detect_room(content: str) -> str:
    """Score content against room keyword categories, return best match."""
    text_lower = content.lower()
    best_room = "general"
    best_score = 0
    for room, keywords in _ROOM_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_room = room
    return best_room


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class MemPalaceMemoryProvider(MemoryProvider):
    """MemPalace full-stack memory provider for Hermes Agent.

    Uses AAAK compression, entity detection, knowledge graph,
    layered wake-up, room auto-detection, and fact checking.
    """

    def __init__(self):
        self._palace_path: str = ""
        self._default_wing: str = "hermes"
        self._session_id: str = ""
        self._platform: str = ""
        self._agent_context: str = ""
        self._agent_name: str = "hermes"
        self._hermes_home: str = ""

        # Lazy-loaded MemPalace modules
        self._searcher = None
        self._palace = None
        self._kg_mod = None
        self._config_mod = None
        self._dialect_mod = None
        self._entity_registry = None
        self._entity_detector = None
        self._fact_checker = None
        self._layers_mod = None
        self._palace_graph = None
        self._convo_miner = None

        # AAAK dialect instance (loaded with entity config)
        self._dialect = None

        # Memory stack for L0-L3 wake-up
        self._memory_stack = None

        # Background sync
        self._sync_lock = threading.Lock()
        self._sync_thread: Optional[threading.Thread] = None
        self._pending_turns: List[Dict] = []

        # Prefetch
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

        # Turn counter
        self._turn_count = 0

        # Wake-up cache (L0+L1 generated at init)
        self._wakeup_cache: str = ""

    # -- Properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "mempalace"

    # -- Core lifecycle ------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import mempalace
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize full MemPalace stack for this session."""
        import mempalace.config as cfg
        import mempalace.searcher
        import mempalace.palace
        import mempalace.knowledge_graph
        import mempalace.dialect
        import mempalace.entity_detector
        import mempalace.entity_registry
        import mempalace.fact_checker
        import mempalace.layers
        import mempalace.palace_graph
        import mempalace.convo_miner

        self._session_id = session_id
        self._platform = kwargs.get("platform", "cli")
        self._agent_context = kwargs.get("agent_context", "primary")
        self._agent_name = kwargs.get("agent_identity", "hermes")
        self._hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))

        self._palace_path = os.environ.get("MEMPALACE_PATH", os.path.expanduser("~/.mempalace"))
        self._default_wing = os.environ.get("MEMPALACE_WING", "hermes")
        os.makedirs(self._palace_path, exist_ok=True)

        # Store module refs
        self._searcher = mempalace.searcher
        self._palace = mempalace.palace
        self._kg_mod = mempalace.knowledge_graph
        self._config_mod = cfg
        self._dialect_mod = mempalace.dialect
        self._entity_detector = mempalace.entity_detector
        self._fact_checker = mempalace.fact_checker
        self._layers_mod = mempalace.layers
        self._palace_graph = mempalace.palace_graph
        self._convo_miner = mempalace.convo_miner

        # Load entity registry (auto-discovers people/projects over time)
        try:
            self._entity_registry = mempalace.entity_registry.EntityRegistry.load(
                config_dir=self._palace_path
            )
        except Exception:
            self._entity_registry = mempalace.entity_registry.EntityRegistry()

        # Initialize AAAK dialect with known entities
        self._dialect = self._init_dialect()

        # Initialize memory stack for L0-L3 wake-up
        try:
            self._memory_stack = mempalace.layers.MemoryStack(
                palace_path=self._palace_path,
                identity_path=os.path.join(self._palace_path, "identity.txt"),
            )
        except Exception as e:
            logger.debug("MemoryStack init failed (palace may be empty): %s", e)
            self._memory_stack = None

        # Generate L0+L1 wake-up block (cached for session)
        self._wakeup_cache = self._generate_wakeup()

        logger.info(
            "MemPalace v2 initialized: palace=%s wing=%s session=%s entities=%d",
            self._palace_path, self._default_wing, session_id,
            len(self._entity_registry.people) + len(self._entity_registry.projects) if self._entity_registry else 0,
        )

    def _init_dialect(self):
        """Create AAAK dialect with known entity codes from registry."""
        try:
            entities = {}
            if self._entity_registry:
                for name in list(self._entity_registry.people.keys())[:50]:
                    code = name[:3].upper()
                    entities[name] = code
                for name in list(self._entity_registry.projects)[:50]:
                    code = name[:3].upper()
                    entities[name] = code
            return self._dialect_mod.Dialect(entities=entities if entities else None)
        except Exception:
            return self._dialect_mod.Dialect()

    def _generate_wakeup(self) -> str:
        """Generate L0+L1 wake-up block using MemoryStack."""
        if not self._memory_stack:
            return ""
        try:
            return self._memory_stack.wake_up(wing=self._default_wing)
        except Exception as e:
            logger.debug("Wake-up generation failed: %s", e)
            return ""

    def system_prompt_block(self) -> str:
        """Inject MemPalace L0+L1 context into system prompt."""
        try:
            status = self._get_status()
            total = status.get("total_drawers", 0)
            wings = status.get("wings", {})
            wing_list = ", ".join(f"{w} ({c})" for w, c in wings.items()) if wings else "empty"

            # Get KG stats for entity count
            kg_stats = self._get_kg_stats()
            entity_count = kg_stats.get("entities", 0)

            block = (
                f"\n\n==MemPalace Memory==\n"
                f"{total} verbatim memories across wings: {wing_list}. "
                f"Knowledge graph: {entity_count} entities tracked.\n"
                f"Tools: mempalace_search (recall), mempalace_add_drawer (store), "
                f"mempalace_kg_query (structured facts), mempalace_kg_add (new facts).\n"
            )

            # Append L0+L1 wake-up if available
            if self._wakeup_cache:
                block += f"\n==Memory Wake-Up (L0+L1)==\n{self._wakeup_cache}\n"

            return block
        except Exception:
            return "\n\nMemPalace memory provider is active."

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SCHEMAS

    # -- Prefetch / recall ---------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
            return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue L2 recall for next turn using MemoryStack."""
        if not query:
            return

        def _do_prefetch():
            try:
                # Use L2 recall via MemoryStack if available
                if self._memory_stack:
                    result = self._memory_stack.recall(wing=self._default_wing, n_results=3)
                    if result:
                        with self._prefetch_lock:
                            self._prefetch_result = f"MemPalace L2 recall:\n{result}"
                        return

                # Fallback to raw search
                results = self._searcher.search_memories(
                    query=query, palace_path=self._palace_path, n_results=3,
                )
                hits = results.get("results", [])
                if not hits:
                    return
                lines = ["MemPalace recall:"]
                for hit in hits[:3]:
                    lines.append(f"  [{hit.get('wing', '?')}/{hit.get('room', '?')}] {hit.get('text', '')[:200]}")
                with self._prefetch_lock:
                    self._prefetch_result = "\n".join(lines)
            except Exception as e:
                logger.debug("MemPalace prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_do_prefetch, daemon=True)
        self._prefetch_thread.start()

    # -- Turn sync with AAAK + entity detection ------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Queue turn for background filing with AAAK compression + entity detection."""
        if self._agent_context != "primary":
            return

        self._turn_count += 1
        turn_data = {
            "user": user_content,
            "assistant": assistant_content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id or self._session_id,
            "platform": self._platform,
            "turn": self._turn_count,
        }

        with self._sync_lock:
            self._pending_turns.append(turn_data)

        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._sync_thread = threading.Thread(target=self._flush_turns, daemon=True)
            self._sync_thread.start()

    def _flush_turns(self):
        """Background: file turns with AAAK compression and entity detection."""
        with self._sync_lock:
            turns = self._pending_turns[:]
            self._pending_turns.clear()

        if not turns:
            return

        for turn in turns:
            try:
                self._file_turn(turn)
            except Exception as e:
                logger.debug("MemPalace turn filing failed: %s", e)

    def _file_turn(self, turn: dict):
        """Process a single turn: detect entities, detect room, AAAK compress, file."""
        combined = f"{turn['user']} {turn['assistant']}"

        # 1. Auto-detect entities from content
        self._detect_and_register_entities(combined)

        # 2. Auto-detect room from content
        room = _detect_room(combined)

        # 3. Store verbatim drawer
        content = (
            f"[user] {turn['user'][:500]}\n"
            f"[assistant] {turn['assistant'][:500]}\n"
            f"---\n"
            f"session: {turn['session_id']} | platform: {turn['platform']} | turn: {turn['turn']}"
        )

        result = self._add_drawer_sync(
            wing=self._default_wing,
            room=room,
            content=content,
            added_by="sync_turn",
        )

        # 4. Generate AAAK compressed index entry
        drawer_id = result.get("drawer_id", "")
        if drawer_id and self._dialect:
            try:
                aaak_line = self._dialect.compress(
                    combined,
                    metadata={
                        "source_file": f"hermes://turn/{turn['session_id']}/{turn['turn']}",
                        "wing": self._default_wing,
                        "room": room,
                        "date": turn["timestamp"][:10],
                    },
                )
                if aaak_line:
                    # Store AAAK as a separate closet-indexed drawer
                    self._add_drawer_sync(
                        wing=self._default_wing,
                        room=f"{room}-aaak",
                        content=aaak_line,
                        added_by="aaak_indexer",
                        source_file=f"hermes://aaak/{drawer_id}",
                    )
            except Exception as e:
                logger.debug("AAAK compression failed for turn: %s", e)

    def _detect_and_register_entities(self, text: str):
        """Run entity detection on text and register new findings."""
        if not self._entity_registry:
            return
        try:
            # learn_from_text auto-discovers high-confidence entities
            self._entity_registry.learn_from_text(text, min_confidence=0.80)
        except Exception as e:
            logger.debug("Entity detection failed: %s", e)

    # -- Pre-compress hook with AAAK ----------------------------------------

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Save compressed messages with AAAK encoding before context discard."""
        try:
            if not messages:
                return ""
            combined = []
            for msg in messages[-10:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if content and role in ("user", "assistant"):
                    combined.append(f"[{role}] {content[:300]}")
            if not combined:
                return ""
            text = "\n".join(combined)

            # Detect room and store with AAAK
            room = _detect_room(text)
            self._add_drawer_sync(
                wing=self._default_wing,
                room=f"compressed-{room}",
                content=text,
                added_by="pre_compress",
            )

            # AAAK compress for index
            if self._dialect:
                aaak = self._dialect.compress(text)
                if aaak:
                    self._add_drawer_sync(
                        wing=self._default_wing,
                        room=f"compressed-{room}-aaak",
                        content=aaak,
                        added_by="aaak_precompress",
                    )
            return ""
        except Exception as e:
            logger.debug("MemPalace on_pre_compress failed: %s", e)
            return ""

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in memory writes with AAAK index."""
        try:
            room = _detect_room(content)
            self._add_drawer_sync(
                wing=self._default_wing,
                room=f"builtin-{target}-{room}",
                content=f"[{action}] {content}",
                added_by="builtin_mirror",
            )
            if self._dialect:
                aaak = self._dialect.compress(content)
                if aaak:
                    self._add_drawer_sync(
                        wing=self._default_wing,
                        room=f"builtin-{target}-{room}-aaak",
                        content=aaak,
                        added_by="aaak_mirror",
                    )
        except Exception as e:
            logger.debug("MemPalace mirror write failed: %s", e)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Mine session end for entities and create cross-wing tunnels."""
        try:
            # Extract entities from full session
            full_text = " ".join(
                m.get("content", "") for m in messages if m.get("content")
            )
            self._detect_and_register_entities(full_text)

            # Build tunnels connecting this session's wing to people mentioned
            if self._entity_registry and self._palace_graph:
                people = self._entity_registry.extract_people_from_query(full_text)
                for person in people[:5]:
                    try:
                        self._palace_graph.create_tunnel(
                            source_wing=self._default_wing,
                            source_room="sessions",
                            target_wing="people",
                            target_room=person.lower().replace(" ", "_"),
                            label=f"session discussed {person}",
                            kind="topic",
                        )
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("MemPalace on_session_end failed: %s", e)

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Observe subagent work for memory filing."""
        if self._agent_context != "primary":
            return
        try:
            content = f"[delegation] Task: {task[:300]}\nResult: {result[:300]}"
            room = _detect_room(task)
            self._add_drawer_sync(
                wing=self._default_wing,
                room=f"delegation-{room}",
                content=content,
                added_by="delegation_observer",
            )
        except Exception as e:
            logger.debug("MemPalace delegation observer failed: %s", e)

    # -- Tool dispatch -------------------------------------------------------

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            dispatch = {
                "mempalace_search": self._tool_search,
                "mempalace_add_drawer": self._tool_add_drawer,
                "mempalace_status": lambda: self._tool_status(),
                "mempalace_kg_query": self._tool_kg_query,
                "mempalace_kg_add": self._tool_kg_add,
                "mempalace_diary_write": self._tool_diary_write,
                "mempalace_list_wings": lambda: self._tool_list_wings(),
                "mempalace_list_rooms": self._tool_list_rooms,
            }
            handler = dispatch.get(tool_name)
            if handler:
                return handler(**args) if tool_name != "mempalace_status" and tool_name != "mempalace_list_wings" else handler()
            return tool_error(f"Unknown MemPalace tool: {tool_name}")
        except Exception as e:
            logger.error("MemPalace tool %s failed: %s", tool_name, e)
            return tool_error(f"MemPalace {tool_name} failed: {e}")

    def shutdown(self) -> None:
        self._flush_turns_sync()
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

    # -- Config --------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "palace_path", "description": "Path to MemPalace data directory", "default": "~/.mempalace", "required": False, "secret": False},
            {"key": "default_wing", "description": "Default wing name for Hermes entries", "default": "hermes", "required": False, "secret": False},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = os.path.join(hermes_home, "mempalace.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        config.update(values)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # -- Tool implementations ------------------------------------------------

    def _tool_search(self, query: str, limit: int = 5, wing: str = None, room: str = None) -> str:
        # Use L3 deep search via MemoryStack if available
        if self._memory_stack and not wing and not room:
            try:
                result = self._memory_stack.search(query, n_results=min(limit, 20))
                if result:
                    return json.dumps({"query": query, "result": result}, indent=2, ensure_ascii=False)
            except Exception:
                pass

        results = self._searcher.search_memories(
            query=query, palace_path=self._palace_path,
            wing=wing, room=room, n_results=min(limit, 20),
        )
        return json.dumps(results, indent=2, ensure_ascii=False)

    def _tool_add_drawer(self, content: str, wing: str = None, room: str = None) -> str:
        wing = wing or self._default_wing
        room = room or _detect_room(content)
        result = self._add_drawer_sync(wing, room, content, added_by="tool_call")

        # Generate AAAK index
        if self._dialect:
            try:
                aaak = self._dialect.compress(content)
                if aaak:
                    self._add_drawer_sync(
                        wing=wing, room=f"{room}-aaak", content=aaak,
                        added_by="aaak_tool", source_file=f"hermes://aaak/{result.get('drawer_id', '')}",
                    )
            except Exception:
                pass

        return json.dumps(result, indent=2)

    def _tool_status(self) -> str:
        status = self._get_status()
        kg_stats = self._get_kg_stats()

        # Get graph stats
        graph_info = {}
        try:
            if self._palace_graph:
                graph_info = self._palace_graph.graph_stats(config=self._config_mod)
        except Exception:
            pass

        status["knowledge_graph"] = kg_stats
        status["graph"] = graph_info
        status["aaak_dialect"] = "active" if self._dialect else "unavailable"
        status["entity_registry"] = {
            "people": len(self._entity_registry.people) if self._entity_registry else 0,
            "projects": len(self._entity_registry.projects) if self._entity_registry else 0,
        }
        status["memory_stack"] = "active" if self._memory_stack else "unavailable"
        return json.dumps(status, indent=2)

    def _tool_kg_query(self, entity: str, direction: str = "both") -> str:
        kg = self._kg_mod.KnowledgeGraph(
            db_path=os.path.join(self._palace_path, "knowledge_graph.sqlite3")
        )
        try:
            facts = kg.query_entity(entity, direction=direction)

            # Also run fact checker to flag any issues with known facts
            issues = []
            try:
                text = f"{entity}"
                for fact in facts:
                    text += f" {fact.get('predicate', '')} {fact.get('object', '')}"
                issues = self._fact_checker.check_text(text, self._palace_path)
            except Exception:
                pass

            return json.dumps({"entity": entity, "facts": facts, "issues": issues}, indent=2, ensure_ascii=False)
        finally:
            kg.close()

    def _tool_kg_add(self, subject: str, predicate: str, object: str) -> str:
        kg = self._kg_mod.KnowledgeGraph(
            db_path=os.path.join(self._palace_path, "knowledge_graph.sqlite3")
        )
        try:
            # Run fact checker first to detect contradictions
            claim = f"{subject}'s {predicate} is {object}"
            issues = []
            try:
                issues = self._fact_checker.check_text(claim, self._palace_path)
            except Exception:
                pass

            # Invalidate conflicting facts
            kg.invalidate(subject, predicate, object)
            triple_id = kg.add_triple(
                subject, predicate, object,
                valid_from=datetime.now(timezone.utc).isoformat(),
                adapter_name="hermes",
            )

            # Create tunnel between subject and object wings
            try:
                if self._palace_graph:
                    self._palace_graph.create_tunnel(
                        source_wing="hermes",
                        source_room=subject.lower().replace(" ", "_"),
                        target_wing="hermes",
                        target_room=object.lower().replace(" ", "_"),
                        label=f"{predicate}",
                        kind="topic",
                    )
            except Exception:
                pass

            return json.dumps({
                "success": True,
                "triple_id": triple_id,
                "fact": f"{subject} {predicate} {object}",
                "contradiction_warnings": issues,
            }, indent=2)
        finally:
            kg.close()

    def _tool_diary_write(self, entry: str, topic: str = None) -> str:
        content = f"[{self._agent_name}] {entry}" + (f" (topic: {topic})" if topic else "")
        self._add_drawer_sync(wing=self._default_wing, room="diary", content=content, added_by="diary")

        if self._dialect:
            try:
                aaak = self._dialect.compress(entry)
                if aaak:
                    self._add_drawer_sync(
                        wing=self._default_wing, room="diary-aaak",
                        content=aaak, added_by="aaak_diary",
                    )
            except Exception:
                pass

        return json.dumps({
            "success": True, "agent": self._agent_name, "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2)

    def _tool_list_wings(self) -> str:
        return json.dumps({"wings": self._get_status().get("wings", {})}, indent=2)

    def _tool_list_rooms(self, wing: str = None) -> str:
        status = self._get_status()
        rooms = status.get("rooms", {})
        if wing:
            rooms = {k: v for k, v in rooms.items() if k.startswith(wing + "/") or k == wing}
        return json.dumps({"rooms": rooms}, indent=2)

    # -- Internal helpers ----------------------------------------------------

    def _add_drawer_sync(self, wing: str, room: str, content: str,
                         added_by: str = "hermes", source_file: str = None) -> dict:
        """Add a drawer to the palace with closets index."""
        from mempalace.config import sanitize_name, sanitize_content

        wing = sanitize_name(wing)
        room = sanitize_name(room)
        content = sanitize_content(content)

        collection = self._palace.get_collection(self._palace_path)
        closets_col = self._palace.get_closets_collection(self._palace_path)

        drawer_id = f"{wing}__{room}__{hashlib.sha256(content.encode()).hexdigest()[:12]}__{int(time.time())}"

        collection.add(
            documents=[content],
            ids=[drawer_id],
            metadatas=[{
                "wing": wing, "room": room, "added_by": added_by,
                "source_file": source_file or "hermes://agent",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        )

        # Build closets (AAAK-aware index pointers)
        closet_lines = self._palace.build_closet_lines(
            source_file=source_file or "hermes://agent",
            drawer_ids=[drawer_id], content=content, wing=wing, room=room,
        )
        if closet_lines:
            self._palace.upsert_closet_lines(
                closets_col, closet_id_base=drawer_id, lines=closet_lines,
                metadata={"wing": wing, "room": room, "source_file": source_file or "hermes://agent"},
            )

        return {"success": True, "drawer_id": drawer_id, "wing": wing, "room": room}

    def _get_status(self) -> dict:
        try:
            collection = self._palace.get_collection(self._palace_path, create=False)
            total = collection.count()
            wings, rooms = {}, {}
            if total > 0:
                results = collection.get(include=["metadatas"], limit=min(total, 10000))
                for meta in (results.get("metadatas") or []):
                    m = meta or {}
                    w, r = m.get("wing", "unknown"), m.get("room", "unknown")
                    wings[w] = wings.get(w, 0) + 1
                    rooms[f"{w}/{r}"] = rooms.get(f"{w}/{r}", 0) + 1
            return {"total_drawers": total, "wings": wings, "rooms": rooms, "palace_path": self._palace_path}
        except Exception as e:
            return {"total_drawers": 0, "wings": {}, "rooms": {}, "error": str(e)}

    def _get_kg_stats(self) -> dict:
        try:
            kg = self._kg_mod.KnowledgeGraph(
                db_path=os.path.join(self._palace_path, "knowledge_graph.sqlite3")
            )
            stats = kg.stats()
            kg.close()
            return stats
        except Exception:
            return {"entities": 0, "triples": 0, "current_facts": 0}

    def _flush_turns_sync(self):
        with self._sync_lock:
            if not self._pending_turns:
                return
        self._flush_turns()


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register MemPalace as a memory provider plugin for Hermes Agent."""
    ctx.register_memory_provider(MemPalaceMemoryProvider())

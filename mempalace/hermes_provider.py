"""Hermes Agent memory provider plugin for MemPalace.

Integrates MemPalace's local-first verbatim memory into Hermes Agent as a
MemoryProvider plugin. Exposes search, write, knowledge graph, and diary
tools through Hermes's tool-calling interface.

MemPalace stores conversation history as verbatim text in a structured
palace (Wings > Rooms > Drawers) with AAAK compression for fast retrieval.
No API keys required - everything runs locally via ChromaDB + SQLite.

Config:
  Set memory.provider to "mempalace" in config.yaml.
  Optional env vars:
    MEMPALACE_PATH  - path to palace directory (default: ~/.mempalace)
    MEMPALACE_WING   - default wing for Hermes entries (default: "hermes")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "mempalace_search",
    "description": (
        "Search MemPalace for verbatim memories across all wings and rooms. "
        "Returns exact stored text ranked by relevance using hybrid BM25 + vector search. "
        "Use this to recall specific past conversations, facts, or context. "
        "Optional: filter by wing (person/project) or room (topic/day)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in stored memories.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 5, max 20).",
                "default": 5,
            },
            "wing": {
                "type": "string",
                "description": "Filter to a specific wing (e.g. 'people', 'projects', 'hermes').",
            },
            "room": {
                "type": "string",
                "description": "Filter to a specific room within a wing.",
            },
        },
        "required": ["query"],
    },
}

ADD_DRAWER_SCHEMA = {
    "name": "mempalace_add_drawer",
    "description": (
        "Store a verbatim memory in MemPalace. Content is preserved exactly as written. "
        "Specify a wing (broad category like 'people', 'projects') and room (topic or day). "
        "Use this to persist important facts, decisions, or context for future recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wing": {
                "type": "string",
                "description": "Wing to store in (e.g. 'hermes', 'people', 'projects'). Default: 'hermes'.",
                "default": "hermes",
            },
            "room": {
                "type": "string",
                "description": "Room within the wing (e.g. 'decisions', 'preferences', or today's date).",
            },
            "content": {
                "type": "string",
                "description": "The exact text to store verbatim.",
            },
        },
        "required": ["content"],
    },
}

STATUS_SCHEMA = {
    "name": "mempalace_status",
    "description": (
        "Get an overview of the MemPalace memory store: total entries, wings, rooms, "
        "and index health. Use this to understand what's stored and verify the system is working."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

KG_QUERY_SCHEMA = {
    "name": "mempalace_kg_query",
    "description": (
        "Query the MemPalace knowledge graph for structured facts about an entity. "
        "Returns temporal entity-relationship triples (subject-predicate-object) with "
        "validity dates. Use this to look up structured facts like relationships, "
        "roles, preferences, and their history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Entity name to look up (person, project, concept).",
            },
            "direction": {
                "type": "string",
                "description": "Relationship direction: 'outgoing', 'incoming', or 'both' (default).",
                "default": "both",
                "enum": ["outgoing", "incoming", "both"],
            },
        },
        "required": ["entity"],
    },
}

KG_ADD_SCHEMA = {
    "name": "mempalace_kg_add",
    "description": (
        "Add a structured fact to the MemPalace knowledge graph as a temporal triple "
        "(subject-predicate-object). Use for explicit structured knowledge like "
        "'Mane works at Hasbro' or 'Lily prefers concise reports'. "
        "Invalidates any previous conflicting fact automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "The subject entity (e.g. a person or project name).",
            },
            "predicate": {
                "type": "string",
                "description": "The relationship (e.g. 'works_at', 'prefers', 'manages').",
            },
            "object": {
                "type": "string",
                "description": "The object entity or value.",
            },
        },
        "required": ["subject", "predicate", "object"],
    },
}

DIARY_WRITE_SCHEMA = {
    "name": "mempalace_diary_write",
    "description": (
        "Write a diary entry for this agent session. Diary entries are timestamped "
        "and stored per-agent for longitudinal self-reflection. Use to record "
        "session goals, observations, or outcomes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entry": {
                "type": "string",
                "description": "Diary entry text.",
            },
            "topic": {
                "type": "string",
                "description": "Optional topic tag for this entry.",
            },
        },
        "required": ["entry"],
    },
}

LIST_WINGS_SCHEMA = {
    "name": "mempalace_list_wings",
    "description": (
        "List all wings in the MemPalace and their drawer counts. "
        "Use to discover what categories of memory are available."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LIST_ROOMS_SCHEMA = {
    "name": "mempalace_list_rooms",
    "description": (
        "List rooms within a wing. If no wing specified, lists all rooms across all wings."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wing": {
                "type": "string",
                "description": "Wing name to list rooms for. Omit for all rooms.",
            },
        },
        "required": [],
    },
}

ALL_TOOL_SCHEMAS = [
    SEARCH_SCHEMA,
    ADD_DRAWER_SCHEMA,
    STATUS_SCHEMA,
    KG_QUERY_SCHEMA,
    KG_ADD_SCHEMA,
    DIARY_WRITE_SCHEMA,
    LIST_WINGS_SCHEMA,
    LIST_ROOMS_SCHEMA,
]


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class MemPalaceMemoryProvider(MemoryProvider):
    """MemPalace local-first verbatim memory provider for Hermes Agent."""

    def __init__(self):
        self._palace_path: str = ""
        self._default_wing: str = "hermes"
        self._session_id: str = ""
        self._platform: str = ""
        self._agent_context: str = ""
        self._agent_name: str = "hermes"

        # Lazy-loaded modules (avoid import at class definition time)
        self._searcher = None
        self._palace = None
        self._kg = None
        self._diary_mod = None
        self._config_mod = None

        # Background sync
        self._sync_lock = threading.Lock()
        self._sync_thread: Optional[threading.Thread] = None
        self._pending_turns: List[Dict] = []
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

        # Turn counter for cadence
        self._turn_count = 0

    # -- Properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "mempalace"

    # -- Core lifecycle ------------------------------------------------------

    def is_available(self) -> bool:
        """Check if MemPalace is installed and palace exists."""
        try:
            import mempalace
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize MemPalace for this session."""
        import mempalace.config as cfg
        import mempalace.searcher
        import mempalace.palace
        import mempalace.knowledge_graph
        import mempalace.diary_ingest

        self._session_id = session_id
        self._platform = kwargs.get("platform", "cli")
        self._agent_context = kwargs.get("agent_context", "primary")
        self._agent_name = kwargs.get("agent_identity", "hermes")
        hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))

        # Resolve palace path
        self._palace_path = os.environ.get(
            "MEMPALACE_PATH",
            os.path.expanduser("~/.mempalace")
        )
        self._default_wing = os.environ.get("MEMPALACE_WING", "hermes")

        # Store module refs
        self._searcher = mempalace.searcher
        self._palace = mempalace.palace
        self._kg_mod = mempalace.knowledge_graph
        self._diary_mod = mempalace.diary_ingest
        self._config_mod = cfg

        # Ensure palace directory exists
        os.makedirs(self._palace_path, exist_ok=True)

        logger.info(
            "MemPalace initialized: palace=%s wing=%s session=%s",
            self._palace_path, self._default_wing, session_id
        )

    def system_prompt_block(self) -> str:
        """Inject MemPalace context into system prompt."""
        try:
            status = self._get_status()
            total = status.get("total_drawers", 0)
            wings = status.get("wings", {})
            wing_list = ", ".join(f"{w} ({c})" for w, c in wings.items()) if wings else "empty"
            return (
                f"\n\nMemPalace memory is active. {total} verbatim memories stored "
                f"across wings: {wing_list}. Use mempalace_search to recall past context, "
                f"mempalace_add_drawer to persist important information, and mempalace_kg_query "
                f"for structured facts. Default wing: {self._default_wing}."
            )
        except Exception:
            return "\n\nMemPalace memory provider is active."

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SCHEMAS

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached prefetch result."""
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
            return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background search for next turn."""
        if not query or not self._searcher:
            return

        def _do_prefetch():
            try:
                results = self._searcher.search_memories(
                    query=query,
                    palace_path=self._palace_path,
                    n_results=3,
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

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Queue turn for background filing into MemPalace."""
        if self._agent_context != "primary":
            return

        turn_data = {
            "user": user_content,
            "assistant": assistant_content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id or self._session_id,
            "platform": self._platform,
        }

        with self._sync_lock:
            self._pending_turns.append(turn_data)

        # Flush in background if we have accumulated turns
        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._sync_thread = threading.Thread(target=self._flush_turns, daemon=True)
            self._sync_thread.start()

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch tool calls to MemPalace."""
        try:
            if tool_name == "mempalace_search":
                return self._tool_search(**args)
            elif tool_name == "mempalace_add_drawer":
                return self._tool_add_drawer(**args)
            elif tool_name == "mempalace_status":
                return self._tool_status()
            elif tool_name == "mempalace_kg_query":
                return self._tool_kg_query(**args)
            elif tool_name == "mempalace_kg_add":
                return self._tool_kg_add(**args)
            elif tool_name == "mempalace_diary_write":
                return self._tool_diary_write(**args)
            elif tool_name == "mempalace_list_wings":
                return self._tool_list_wings()
            elif tool_name == "mempalace_list_rooms":
                return self._tool_list_rooms(**args)
            else:
                return tool_error(f"Unknown MemPalace tool: {tool_name}")
        except Exception as e:
            logger.error("MemPalace tool %s failed: %s", tool_name, e)
            return tool_error(f"MemPalace {tool_name} failed: {e}")

    def shutdown(self) -> None:
        """Flush pending turns and close resources."""
        self._flush_turns_sync()
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

    # -- Optional hooks ------------------------------------------------------

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract key facts before context compression discards messages."""
        # Save the compressed content as a drawer so nothing is lost
        try:
            if not messages:
                return ""
            combined = []
            for msg in messages[-10:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if content and role in ("user", "assistant"):
                    combined.append(f"[{role}] {content[:300]}")
            if combined:
                text = "\n".join(combined)
                today = datetime.now().strftime("%Y-%m-%d")
                self._add_drawer_sync(
                    wing=self._default_wing,
                    room=f"compressed-{today}",
                    content=text,
                    added_by="pre_compress",
                )
            return ""
        except Exception as e:
            logger.debug("MemPalace on_pre_compress failed: %s", e)
            return ""

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in memory writes to MemPalace."""
        try:
            self._add_drawer_sync(
                wing=self._default_wing,
                room=f"builtin-{target}",
                content=f"[{action}] {content}",
                added_by="builtin_mirror",
            )
        except Exception as e:
            logger.debug("MemPalace mirror write failed: %s", e)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Config fields for 'hermes memory setup'."""
        return [
            {
                "key": "palace_path",
                "description": "Path to MemPalace data directory",
                "default": "~/.mempalace",
                "required": False,
                "secret": False,
            },
            {
                "key": "default_wing",
                "description": "Default wing name for Hermes entries",
                "default": "hermes",
                "required": False,
                "secret": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write MemPalace config."""
        import json as _json
        config_path = os.path.join(hermes_home, "mempalace.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = _json.load(f)
        config.update(values)
        with open(config_path, "w") as f:
            _json.dump(config, f, indent=2)

    # -- Tool implementations ------------------------------------------------

    def _tool_search(self, query: str, limit: int = 5, wing: str = None, room: str = None) -> str:
        results = self._searcher.search_memories(
            query=query,
            palace_path=self._palace_path,
            wing=wing,
            room=room,
            n_results=min(limit, 20),
        )
        return json.dumps(results, indent=2, ensure_ascii=False)

    def _tool_add_drawer(self, content: str, wing: str = None, room: str = None) -> str:
        wing = wing or self._default_wing
        room = room or datetime.now().strftime("%Y-%m-%d")
        result = self._add_drawer_sync(wing, room, content, added_by="tool_call")
        return json.dumps(result, indent=2)

    def _tool_status(self) -> str:
        return json.dumps(self._get_status(), indent=2)

    def _tool_kg_query(self, entity: str, direction: str = "both") -> str:
        kg = self._kg_mod.KnowledgeGraph(
            db_path=os.path.join(self._palace_path, "knowledge_graph.sqlite3")
        )
        try:
            facts = kg.query_entity(entity, direction=direction)
            return json.dumps({"entity": entity, "facts": facts}, indent=2, ensure_ascii=False)
        finally:
            kg.close()

    def _tool_kg_add(self, subject: str, predicate: str, object: str) -> str:
        kg = self._kg_mod.KnowledgeGraph(
            db_path=os.path.join(self._palace_path, "knowledge_graph.sqlite3")
        )
        try:
            # Invalidate any existing conflicting fact
            kg.invalidate(subject, predicate, object)
            triple_id = kg.add_triple(
                subject, predicate, object,
                valid_from=datetime.now(timezone.utc).isoformat(),
                adapter_name="hermes",
            )
            fact = kg.query_entity(subject)
            return json.dumps({
                "success": True,
                "triple_id": triple_id,
                "fact": f"{subject} {predicate} {object}",
            }, indent=2)
        finally:
            kg.close()

    def _tool_diary_write(self, entry: str, topic: str = None) -> str:
        kg = self._kg_mod.KnowledgeGraph(
            db_path=os.path.join(self._palace_path, "knowledge_graph.sqlite3")
        )
        kg.close()  # just needed to verify DB exists
        # Use diary_ingest for structured diary entries
        result = self._add_drawer_sync(
            wing=self._default_wing,
            room="diary",
            content=f"[{self._agent_name}] {entry}" + (f" (topic: {topic})" if topic else ""),
            added_by="diary",
        )
        return json.dumps({
            "success": True,
            "agent": self._agent_name,
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2)

    def _tool_list_wings(self) -> str:
        status = self._get_status()
        return json.dumps({"wings": status.get("wings", {})}, indent=2)

    def _tool_list_rooms(self, wing: str = None) -> str:
        status = self._get_status()
        rooms = status.get("rooms", {})
        if wing:
            rooms = {k: v for k, v in rooms.items() if k.startswith(wing + "/") or k == wing}
        return json.dumps({"rooms": rooms}, indent=2)

    # -- Internal helpers ----------------------------------------------------

    def _add_drawer_sync(self, wing: str, room: str, content: str,
                         added_by: str = "hermes", source_file: str = None) -> dict:
        """Add a drawer to the palace (synchronous)."""
        from mempalace.config import sanitize_name, sanitize_content

        wing = sanitize_name(wing)
        room = sanitize_name(room)
        content = sanitize_content(content)

        collection = self._palace.get_collection(self._palace_path)
        closets_col = self._palace.get_closets_collection(self._palace_path)

        # Generate unique drawer ID
        import hashlib
        drawer_id = f"{wing}__{room}__{hashlib.sha256(content.encode()).hexdigest()[:12]}__{int(time.time())}"

        collection.add(
            documents=[content],
            ids=[drawer_id],
            metadatas=[{
                "wing": wing,
                "room": room,
                "added_by": added_by,
                "source_file": source_file or "hermes://agent",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        )

        # Build closets (index lines) for fast retrieval
        closet_lines = self._palace.build_closet_lines(
            source_file=source_file or "hermes://agent",
            drawer_ids=[drawer_id],
            content=content,
            wing=wing,
            room=room,
        )
        if closet_lines:
            self._palace.upsert_closet_lines(
                closets_col,
                closet_id_base=drawer_id,
                lines=closet_lines,
                metadata={"wing": wing, "room": room, "source_file": source_file or "hermes://agent"},
            )

        return {"success": True, "drawer_id": drawer_id, "wing": wing, "room": room}

    def _get_status(self) -> dict:
        """Get palace status."""
        try:
            collection = self._palace.get_collection(self._palace_path, create=False)
            total = collection.count()

            # Extract wings and rooms from metadata
            wings = {}
            rooms = {}
            if total > 0:
                results = collection.get(include=["metadatas"], limit=min(total, 10000))
                for meta in (results.get("metadatas") or []):
                    w = (meta or {}).get("wing", "unknown")
                    r = (meta or {}).get("room", "unknown")
                    wings[w] = wings.get(w, 0) + 1
                    room_key = f"{w}/{r}"
                    rooms[room_key] = rooms.get(room_key, 0) + 1

            return {
                "total_drawers": total,
                "wings": wings,
                "rooms": rooms,
                "palace_path": self._palace_path,
            }
        except Exception as e:
            return {"total_drawers": 0, "wings": {}, "rooms": {}, "error": str(e)}

    def _flush_turns(self):
        """Background: file pending turns into MemPalace."""
        with self._sync_lock:
            turns = self._pending_turns[:]
            self._pending_turns.clear()

        if not turns:
            return

        for turn in turns:
            try:
                today = turn["timestamp"][:10]  # YYYY-MM-DD
                content = (
                    f"[user] {turn['user'][:500]}\n"
                    f"[assistant] {turn['assistant'][:500]}\n"
                    f"---\n"
                    f"session: {turn['session_id']} | platform: {turn['platform']}"
                )
                self._add_drawer_sync(
                    wing=self._default_wing,
                    room=today,
                    content=content,
                    added_by="sync_turn",
                )
            except Exception as e:
                logger.debug("MemPalace turn filing failed: %s", e)

    def _flush_turns_sync(self):
        """Synchronous flush for shutdown."""
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

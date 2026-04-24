"""Tests for explicit tunnel helpers in mempalace.palace_graph."""

from unittest.mock import MagicMock, patch

import pytest

with patch.dict("sys.modules", {"chromadb": MagicMock()}):
    import mempalace.palace_graph as palace_graph


def _use_tmp_tunnel_file(monkeypatch, tmp_path):
    tunnel_file = tmp_path / "tunnels.json"
    monkeypatch.setattr(palace_graph, "_TUNNEL_FILE", str(tunnel_file))
    return tunnel_file


class TestTunnelStorage:
    def test_load_tunnels_missing_file_returns_empty_list(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        assert palace_graph._load_tunnels() == []

    def test_load_tunnels_corrupt_file_returns_empty_list(self, tmp_path, monkeypatch):
        tunnel_file = _use_tmp_tunnel_file(monkeypatch, tmp_path)
        tunnel_file.write_text("{not valid json", encoding="utf-8")
        assert palace_graph._load_tunnels() == []

    def test_save_and_load_round_trip(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        tunnels = [
            {
                "id": "abc123",
                "source": {"wing": "wing_code", "room": "auth"},
                "target": {"wing": "wing_people", "room": "users"},
                "label": "same concept",
            }
        ]
        palace_graph._save_tunnels(tunnels)
        assert palace_graph._load_tunnels() == tunnels


class TestExplicitTunnels:
    def test_create_tunnel_deduplicates_reverse_order_and_updates_label(
        self, tmp_path, monkeypatch
    ):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        first = palace_graph.create_tunnel(
            "wing_code", "auth", "wing_people", "users", label="same concept"
        )
        second = palace_graph.create_tunnel(
            "wing_people", "users", "wing_code", "auth", label="updated label"
        )

        assert first["id"] == second["id"]
        assert len(palace_graph.list_tunnels()) == 1
        assert second["label"] == "updated label"
        assert second["created_at"] == first["created_at"]
        assert "updated_at" in second

    def test_create_tunnel_rejects_empty_names(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        with pytest.raises(ValueError):
            palace_graph.create_tunnel("", "auth", "wing_people", "users")

    def test_list_tunnels_filters_by_either_side(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        palace_graph.create_tunnel("wing_code", "auth", "wing_people", "users", label="A")
        palace_graph.create_tunnel("wing_ops", "deploy", "wing_people", "users", label="B")

        assert len(palace_graph.list_tunnels()) == 2
        assert len(palace_graph.list_tunnels("wing_people")) == 2
        assert len(palace_graph.list_tunnels("wing_code")) == 1

    def test_delete_tunnel_removes_saved_tunnel(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        tunnel = palace_graph.create_tunnel(
            "wing_code", "auth", "wing_people", "users", label="same concept"
        )

        assert palace_graph.delete_tunnel(tunnel["id"]) == {"deleted": tunnel["id"]}
        assert palace_graph.list_tunnels() == []

    def test_follow_tunnels_returns_direction_and_preview(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        palace_graph.create_tunnel(
            "wing_code",
            "auth",
            "wing_people",
            "users",
            label="same concept",
            target_drawer_id="drawer_users_1",
        )

        col = MagicMock()
        col.get.return_value = {
            "ids": ["drawer_users_1"],
            "documents": ["A" * 400],
            "metadatas": [{}],
        }

        outgoing = palace_graph.follow_tunnels("wing_code", "auth", col=col)
        assert len(outgoing) == 1
        assert outgoing[0]["direction"] == "outgoing"
        assert outgoing[0]["connected_wing"] == "wing_people"
        assert outgoing[0]["connected_room"] == "users"
        assert outgoing[0]["drawer_id"] == "drawer_users_1"
        assert len(outgoing[0]["drawer_preview"]) == 300

        incoming = palace_graph.follow_tunnels("wing_people", "users", col=col)
        assert len(incoming) == 1
        assert incoming[0]["direction"] == "incoming"
        assert incoming[0]["connected_wing"] == "wing_code"

    def test_follow_tunnels_returns_connections_even_if_collection_lookup_fails(
        self, tmp_path, monkeypatch
    ):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        palace_graph.create_tunnel(
            "wing_code",
            "auth",
            "wing_people",
            "users",
            label="same concept",
            target_drawer_id="drawer_users_1",
        )

        col = MagicMock()
        col.get.side_effect = RuntimeError("boom")

        connections = palace_graph.follow_tunnels("wing_code", "auth", col=col)
        assert len(connections) == 1
        assert "drawer_preview" not in connections[0]


class TestTopicTunnels:
    """Cross-wing topic tunnels (issue #1180).

    When two wings share confirmed TOPIC labels above a configurable
    threshold, a symmetric tunnel is created between them. Tunnels are
    routed through the existing ``create_tunnel`` storage so they share
    dedup and persistence with explicit tunnels.
    """

    def test_compute_topic_tunnels_creates_link_for_shared_topic(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_alpha": ["Angular", "OpenAPI"],
            "wing_beta": ["OpenAPI", "Kubernetes"],
        }
        created = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=1)
        assert len(created) == 1
        assert created[0]["source"]["wing"] in {"wing_alpha", "wing_beta"}
        assert created[0]["target"]["wing"] in {"wing_alpha", "wing_beta"}
        # Room is the topic itself (case preserved from the first wing).
        assert created[0]["source"]["room"] == "OpenAPI"
        assert "OpenAPI" in created[0]["label"]

        # Tunnel is retrievable via the standard list_tunnels API.
        listed = palace_graph.list_tunnels()
        assert len(listed) == 1
        assert listed[0]["id"] == created[0]["id"]

    def test_compute_topic_tunnels_no_link_below_threshold(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_alpha": ["Angular", "OpenAPI"],
            "wing_beta": ["OpenAPI", "Kubernetes"],
        }
        # min_count=2 requires two overlapping topics — only one shared.
        created = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=2)
        assert created == []
        assert palace_graph.list_tunnels() == []

    def test_compute_topic_tunnels_above_threshold_creates_per_topic_links(
        self, tmp_path, monkeypatch
    ):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_alpha": ["Angular", "OpenAPI", "Postgres"],
            "wing_beta": ["Angular", "OpenAPI", "Redis"],
        }
        created = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=2)
        # Two shared topics × one wing pair = two tunnels.
        rooms = sorted(t["source"]["room"] for t in created)
        assert rooms == ["Angular", "OpenAPI"]

    def test_compute_topic_tunnels_case_insensitive_overlap(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_alpha": ["openapi"],
            "wing_beta": ["OpenAPI"],
        }
        created = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=1)
        assert len(created) == 1

    def test_compute_topic_tunnels_empty_input_is_noop(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        assert palace_graph.compute_topic_tunnels({}) == []
        assert palace_graph.compute_topic_tunnels({"wing_a": []}) == []
        assert palace_graph.list_tunnels() == []

    def test_compute_topic_tunnels_three_wings_pairwise(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_a": ["foo"],
            "wing_b": ["foo"],
            "wing_c": ["foo"],
        }
        created = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=1)
        # 3 wings sharing the same topic → C(3,2) = 3 pairs → 3 tunnels.
        assert len(created) == 3
        endpoint_pairs = {
            tuple(sorted([t["source"]["wing"], t["target"]["wing"]])) for t in created
        }
        assert endpoint_pairs == {
            ("wing_a", "wing_b"),
            ("wing_a", "wing_c"),
            ("wing_b", "wing_c"),
        }

    def test_topic_tunnels_for_wing_only_links_that_wing(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_a": ["foo", "bar"],
            "wing_b": ["foo"],
            "wing_c": ["bar"],
        }
        # wing_a should link to both b (via foo) and c (via bar).
        created = palace_graph.topic_tunnels_for_wing("wing_a", topics_by_wing)
        endpoint_pairs = {
            tuple(sorted([t["source"]["wing"], t["target"]["wing"]])) for t in created
        }
        assert endpoint_pairs == {("wing_a", "wing_b"), ("wing_a", "wing_c")}
        # The b-c pair is NOT created because wing_a's incremental pass
        # only computes pairs that include wing_a.
        assert len(palace_graph.list_tunnels()) == 2

    def test_topic_tunnels_for_wing_unknown_wing_is_noop(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {"wing_a": ["foo"], "wing_b": ["foo"]}
        assert palace_graph.topic_tunnels_for_wing("wing_missing", topics_by_wing) == []
        assert palace_graph.list_tunnels() == []

    def test_compute_topic_tunnels_dedupe_on_recompute(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_alpha": ["OpenAPI"],
            "wing_beta": ["OpenAPI"],
        }
        first = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=1)
        second = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=1)
        # create_tunnel is symmetric/dedupe — repeated computation should
        # not multiply the stored tunnels.
        assert first[0]["id"] == second[0]["id"]
        assert len(palace_graph.list_tunnels()) == 1

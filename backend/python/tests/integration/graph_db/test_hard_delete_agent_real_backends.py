"""``hard_delete_agent`` removes the agent's whole subgraph on a real Neo4j and ArangoDB, and nothing another agent still uses.

The agent holds knowledge, a toolset with a tool, and an MCP server with two tools; a second agent shares a second MCP server
with it, and a server no agent is attached to holds one of the first server's tools too. After the first agent is deleted, its
knowledge, toolset, MCP server and the tools only they hold are gone; the shared MCP server, its tool, the unattached server, the
tool it holds and the second agent stay. Neo4j left the MCP servers and their tools behind.

Runs in backend-matrix on both graph jobs. Environment: NEO4J_IT_URI, NEO4J_IT_PASSWORD, ARANGO_IT_URL, ARANGO_IT_PASSWORD.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from typing import TYPE_CHECKING

import pytest

from app.config.constants.arangodb import CollectionNames
from app.utils.time_conversion import get_epoch_timestamp_in_ms
from tests.integration.real_graph import (
    backend_unavailable,
    connect_arango,
    connect_neo4j,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.services.graph_db.interface.graph_db_provider import IGraphDBProvider

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

ARANGO_DB = "hard_delete_agent_it"
C = CollectionNames
NODES = (C.AGENT_INSTANCES.value, C.AGENT_KNOWLEDGE.value, C.AGENT_TOOLSETS.value, C.AGENT_TOOLS.value, C.AGENT_MCP_SERVERS.value)

logger = logging.getLogger("hard-delete-agent-it")
NAMES = ("agent", "other", "knowledge", "toolset", "tool", "mcp", "mcp_tool", "shared_mcp", "shared_tool", "loose_mcp", "kept_tool")


async def _present(graph: IGraphDBProvider, ids: dict[str, str]) -> set[str]:
    return {name for name in NAMES if await graph.get_document(ids[name], ids[f"{name}__coll"]) is not None}


def _edge(fc: str, fi: str, tc: str, ti: str) -> dict:
    return {"from_id": fi, "from_collection": fc, "to_id": ti, "to_collection": tc, "createdAtTimestamp": get_epoch_timestamp_in_ms()}


@pytest.fixture(params=["neo4j", "arango"])
async def world(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[IGraphDBProvider, dict[str, str]]]:
    async with contextlib.AsyncExitStack() as cleanup:
        try:
            graph = await (connect_neo4j(logger, monkeypatch) if request.param == "neo4j" else connect_arango(logger, ARANGO_DB))
        except Exception as exc:
            backend_unavailable(request.param, exc)
        disconnect = getattr(graph, "disconnect", None)
        if disconnect is not None:
            cleanup.push_async_callback(disconnect)
        run = uuid.uuid4().hex[:10]
        ids = {k: f"{k}-hda-{run}" for k in NAMES}
        coll = {"agent": C.AGENT_INSTANCES.value, "other": C.AGENT_INSTANCES.value, "knowledge": C.AGENT_KNOWLEDGE.value,
                "toolset": C.AGENT_TOOLSETS.value, "tool": C.AGENT_TOOLS.value, "mcp": C.AGENT_MCP_SERVERS.value,
                "mcp_tool": C.AGENT_TOOLS.value, "shared_mcp": C.AGENT_MCP_SERVERS.value, "shared_tool": C.AGENT_TOOLS.value,
                "loose_mcp": C.AGENT_MCP_SERVERS.value, "kept_tool": C.AGENT_TOOLS.value}

        async def remove() -> None:
            for name, key in ids.items():
                with contextlib.suppress(Exception):
                    await graph.delete_nodes_and_edges([key], coll[name])
        cleanup.push_async_callback(remove)

        now = get_epoch_timestamp_in_ms()
        agent = {"description": "d", "startMessage": "s", "systemPrompt": "p", "models": [], "createdBy": "u", "createdAtTimestamp": now,
                 "updatedAtTimestamp": now}
        assert await graph.batch_upsert_nodes([{"_key": ids["agent"], "name": "a", **agent}, {"_key": ids["other"], "name": "b", **agent}],
                                              C.AGENT_INSTANCES.value)
        assert await graph.batch_upsert_nodes([{"_key": ids["knowledge"], "connectorId": "c", "filters": "{}", "createdBy": "u", "createdAtTimestamp": now}],
                                              C.AGENT_KNOWLEDGE.value)
        assert await graph.batch_upsert_nodes([{"_key": ids["toolset"], "name": "slack", "displayName": "Slack", "type": "app", "userId": "u",
                                                "createdBy": "u", "createdAtTimestamp": now}], C.AGENT_TOOLSETS.value)
        assert await graph.batch_upsert_nodes([{"_key": ids[k], "name": k, "fullName": f"x.{k}", "toolsetName": "x", "createdBy": "u", "createdAtTimestamp": now}
                                               for k in ("tool", "mcp_tool", "shared_tool", "kept_tool")], C.AGENT_TOOLS.value)
        assert await graph.batch_upsert_nodes([{"_key": ids[k], "name": k, "displayName": k, "instanceId": f"i-{k}", "userId": "u", "createdBy": "u",
                                                "createdAtTimestamp": now} for k in ("mcp", "shared_mcp", "loose_mcp")], C.AGENT_MCP_SERVERS.value)
        for edges, collection in (
            ([_edge(C.AGENT_INSTANCES.value, ids["agent"], C.AGENT_KNOWLEDGE.value, ids["knowledge"])], C.AGENT_HAS_KNOWLEDGE.value),
            ([_edge(C.AGENT_INSTANCES.value, ids["agent"], C.AGENT_TOOLSETS.value, ids["toolset"])], C.AGENT_HAS_TOOLSET.value),
            ([_edge(C.AGENT_TOOLSETS.value, ids["toolset"], C.AGENT_TOOLS.value, ids["tool"])], C.TOOLSET_HAS_TOOL.value),
            ([_edge(C.AGENT_INSTANCES.value, a, C.AGENT_MCP_SERVERS.value, m) for a, m in
              ((ids["agent"], ids["mcp"]), (ids["agent"], ids["shared_mcp"]), (ids["other"], ids["shared_mcp"]))], C.AGENT_HAS_MCP_SERVER.value),
            ([_edge(C.AGENT_MCP_SERVERS.value, m, C.AGENT_TOOLS.value, t) for m, t in
              ((ids["mcp"], ids["mcp_tool"]), (ids["mcp"], ids["kept_tool"]), (ids["shared_mcp"], ids["shared_tool"]),
               (ids["loose_mcp"], ids["kept_tool"]))], C.MCP_SERVER_HAS_TOOL.value),
        ):
            assert await graph.batch_create_edges(edges, collection)
        yield graph, {**ids, **{f"{k}__coll": v for k, v in coll.items()}}


async def test_the_agent_subgraph_goes_and_what_another_agent_uses_stays(world: tuple[IGraphDBProvider, dict[str, str]]) -> None:
    graph, ids = world
    result = await graph.hard_delete_agent(ids["agent"])
    assert result["agents_deleted"] == 1 and result["mcp_servers_deleted"] == 1
    # knowledge, agent->toolset, toolset->tool, the two agent->server links, and the server's two links to tools
    assert result["edges_deleted"] == 7
    assert await _present(graph, ids) == {"other", "shared_mcp", "shared_tool", "loose_mcp", "kept_tool"}


async def test_two_agents_sharing_a_server_deleted_at_once_take_it(world: tuple[IGraphDBProvider, dict[str, str]]) -> None:
    """Each delete re-checks its servers after its own links are gone, so whichever removes the last link takes the server."""
    graph, ids = world
    await asyncio.gather(graph.hard_delete_agent(ids["agent"]), graph.hard_delete_agent(ids["other"]))
    assert await _present(graph, ids) == {"loose_mcp", "kept_tool"}


async def test_two_agents_sharing_a_toolset_deleted_at_once_take_it(store: IGraphDBProvider) -> None:
    """The last two agents of a toolset deleted at once: whichever removes the last link takes the toolset, its tools and the
    links between them. Neither delete may fail, and nothing may stay."""
    seeded = Seeded(store)
    failures, left = [], []
    try:
        for round_ in range(ROUNDS):
            run = uuid.uuid4().hex[:10]
            a, b = seeded.key(f"a-tc-{run}", C.AGENT_INSTANCES.value), seeded.key(f"b-tc-{run}", C.AGENT_INSTANCES.value)
            toolset = seeded.key(f"ts-tc-{run}", C.AGENT_TOOLSETS.value)
            tools = [seeded.key(f"t{i}-tc-{run}", C.AGENT_TOOLS.value) for i in range(3)]
            await seeded.agents(a, b)
            await seeded.toolsets(toolset)
            await seeded.tools(*tools)
            await seeded.link_toolsets([(a, toolset), (b, toolset)], [(toolset, t) for t in tools])
            results = await asyncio.gather(store.hard_delete_agent(a), store.hard_delete_agent(b))
            failures += [round_ for r in results if r["agents_deleted"] != 1]
            left += await seeded.present()
            left += [f"{toolset}->{t}" for t in tools
                     if await store.get_edge(toolset, C.AGENT_TOOLSETS.value, t, C.AGENT_TOOLS.value, C.TOOLSET_HAS_TOOL.value) is not None]
        assert failures == [], f"a delete failed in rounds {failures}"
        assert left == [], f"left behind: {left}"
    finally:
        await seeded.remove()


async def test_deleting_every_agent_leaves_no_agent_subgraph(world: tuple[IGraphDBProvider, dict[str, str]]) -> None:
    """``hard_delete_all_agents`` (migrations, cleanup) takes every agent's knowledge, toolsets, MCP servers and the tools only they hold.
    It deletes every agent of the store, so it runs only when this run is the store's sole user (no pytest-xdist workers)."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("deletes every agent of the shared store: run without pytest-xdist (CI runs these files serially)")
    graph, ids = world
    result = await graph.hard_delete_all_agents()
    assert result["agents_deleted"] >= 2 and result["mcp_servers_deleted"] >= 2
    assert await _present(graph, ids) == {"loose_mcp", "kept_tool"}      # a server no agent uses keeps its tool


@pytest.fixture(params=["neo4j", "arango"])
async def store(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[IGraphDBProvider]:
    try:
        graph = await (connect_neo4j(logger, monkeypatch) if request.param == "neo4j" else connect_arango(logger, ARANGO_DB))
    except Exception as exc:
        backend_unavailable(request.param, exc)
    yield graph
    disconnect = getattr(graph, "disconnect", None)
    if disconnect is not None:
        await disconnect()


ROUNDS = 20
AGENT_DOC = {"description": "d", "startMessage": "s", "systemPrompt": "p", "models": [], "createdBy": "u"}


class Seeded:
    """Every key a test writes, registered before its first write and removed in the test's ``finally``."""

    def __init__(self, graph: IGraphDBProvider) -> None:
        self.graph, self.keys = graph, []

    def key(self, name: str, coll: str) -> str:
        self.keys.append((name, coll))
        return name

    async def remove(self) -> None:
        for name, coll in self.keys:
            with contextlib.suppress(Exception):
                await self.graph.delete_nodes_and_edges([name], coll)

    async def agents(self, *keys: str) -> None:
        now = get_epoch_timestamp_in_ms()
        assert await self.graph.batch_upsert_nodes([{"_key": k, "name": k, **AGENT_DOC, "createdAtTimestamp": now, "updatedAtTimestamp": now}
                                                    for k in keys], C.AGENT_INSTANCES.value)

    async def servers(self, *keys: str) -> None:
        now = get_epoch_timestamp_in_ms()
        assert await self.graph.batch_upsert_nodes([{"_key": k, "name": k, "displayName": k, "instanceId": f"i-{k}", "userId": "u", "createdBy": "u",
                                                     "createdAtTimestamp": now} for k in keys], C.AGENT_MCP_SERVERS.value)

    async def tools(self, *keys: str) -> None:
        now = get_epoch_timestamp_in_ms()
        assert await self.graph.batch_upsert_nodes([{"_key": k, "name": k, "fullName": f"x.{k}", "toolsetName": "x", "createdBy": "u",
                                                     "createdAtTimestamp": now} for k in keys], C.AGENT_TOOLS.value)

    async def link(self, agent_servers: list[tuple[str, str]], server_tools: list[tuple[str, str]]) -> None:
        if agent_servers:
            assert await self.graph.batch_create_edges([_edge(C.AGENT_INSTANCES.value, a, C.AGENT_MCP_SERVERS.value, s) for a, s in agent_servers],
                                                       C.AGENT_HAS_MCP_SERVER.value)
        if server_tools:
            assert await self.graph.batch_create_edges([_edge(C.AGENT_MCP_SERVERS.value, s, C.AGENT_TOOLS.value, t) for s, t in server_tools],
                                                       C.MCP_SERVER_HAS_TOOL.value)

    async def toolsets(self, *keys: str) -> None:
        now = get_epoch_timestamp_in_ms()
        assert await self.graph.batch_upsert_nodes([{"_key": k, "name": k, "displayName": k, "type": "app", "userId": "u", "createdBy": "u",
                                                     "createdAtTimestamp": now} for k in keys], C.AGENT_TOOLSETS.value)

    async def link_toolsets(self, agent_toolsets: list[tuple[str, str]], toolset_tools: list[tuple[str, str]]) -> None:
        assert await self.graph.batch_create_edges([_edge(C.AGENT_INSTANCES.value, a, C.AGENT_TOOLSETS.value, t) for a, t in agent_toolsets],
                                                   C.AGENT_HAS_TOOLSET.value)
        assert await self.graph.batch_create_edges([_edge(C.AGENT_TOOLSETS.value, t, C.AGENT_TOOLS.value, tool) for t, tool in toolset_tools],
                                                   C.TOOLSET_HAS_TOOL.value)

    async def present(self) -> list[str]:
        return [name for name, coll in self.keys if await self.graph.get_document(name, coll) is not None]


@pytest.mark.parametrize("layout", ["one-shared-server", "two-servers-one-tool"])
async def test_concurrent_deletes_of_two_agents_never_fail_and_leave_nothing(store: IGraphDBProvider, layout: str) -> None:
    """Two agents deleted at once, sharing a server (each may find it orphaned and clean it up) or holding two servers that share
    a tool (each may read the tool as still held by the other's server). Neither delete may fail, and nothing may stay."""
    seeded = Seeded(store)
    failures, left = [], []
    try:
        for round_ in range(ROUNDS):
            run = uuid.uuid4().hex[:10]
            a, b = seeded.key(f"a-cc-{run}", C.AGENT_INSTANCES.value), seeded.key(f"b-cc-{run}", C.AGENT_INSTANCES.value)
            if layout == "one-shared-server":
                servers = [seeded.key(f"s-cc-{run}", C.AGENT_MCP_SERVERS.value)]
                tools = [seeded.key(f"t{i}-cc-{run}", C.AGENT_TOOLS.value) for i in range(5)]
                agent_servers, server_tools = [(a, servers[0]), (b, servers[0])], [(servers[0], t) for t in tools]
            else:
                servers = [seeded.key(f"s{i}-cc-{run}", C.AGENT_MCP_SERVERS.value) for i in (1, 2)]
                tools = [seeded.key(f"t-cc-{run}", C.AGENT_TOOLS.value)]
                agent_servers, server_tools = [(a, servers[0]), (b, servers[1])], [(s, tools[0]) for s in servers]
            await seeded.agents(a, b)
            await seeded.servers(*servers)
            await seeded.tools(*tools)
            await seeded.link(agent_servers, server_tools)
            results = await asyncio.gather(store.hard_delete_agent(a), store.hard_delete_agent(b))
            # a failed delete answers zero counts (it never raises): each must have deleted its own agent
            failures += [round_ for r in results if r["agents_deleted"] != 1]
            left += await seeded.present()
        assert failures == [], f"a delete failed in rounds {failures}"
        assert left == [], f"left behind: {left}"
    finally:
        await seeded.remove()


@pytest.mark.parametrize("other_keeps_one", [False, True], ids=["both-servers-go", "one-server-kept"])
async def test_a_tool_two_servers_of_one_agent_share(store: IGraphDBProvider, other_keeps_one: bool) -> None:
    """Two MCP servers of one agent hold the same tool. Deleting the agent takes both servers and the tool; when another agent
    still links one of the servers, that server keeps the tool."""
    seeded = Seeded(store)
    run = uuid.uuid4().hex[:10]
    agent_key, other_key = (seeded.key(f"{k}-two-{run}", C.AGENT_INSTANCES.value) for k in ("agent", "other"))
    s1, s2 = (seeded.key(f"{k}-two-{run}", C.AGENT_MCP_SERVERS.value) for k in ("s1", "s2"))
    tool = seeded.key(f"tool-two-{run}", C.AGENT_TOOLS.value)
    try:
        await seeded.agents(agent_key, other_key)
        await seeded.servers(s1, s2)
        await seeded.tools(tool)
        await seeded.link([(agent_key, s1), (agent_key, s2), *([(other_key, s2)] if other_keeps_one else [])], [(s1, tool), (s2, tool)])
        result = await store.hard_delete_agent(agent_key)
        assert result["agents_deleted"] == 1
        assert set(await seeded.present()) == ({other_key, s2, tool} if other_keeps_one else {other_key})
    finally:
        await seeded.remove()


async def test_a_toolset_another_agent_links_keeps_its_tools(store: IGraphDBProvider) -> None:
    """Two agents link one toolset with a tool; the first also has a toolset of its own. Deleting the first takes its own toolset
    and tool; the shared toolset, its tool and the link between them stay for the second agent."""
    seeded = Seeded(store)
    run = uuid.uuid4().hex[:10]
    agent_key, other_key = (seeded.key(f"{k}-ts-{run}", C.AGENT_INSTANCES.value) for k in ("agent", "other"))
    shared, own = (seeded.key(f"{k}-ts-{run}", C.AGENT_TOOLSETS.value) for k in ("shared", "own"))
    shared_tool, own_tool = (seeded.key(f"{k}-tool-ts-{run}", C.AGENT_TOOLS.value) for k in ("shared", "own"))
    try:
        await seeded.agents(agent_key, other_key)
        await seeded.toolsets(shared, own)
        await seeded.tools(shared_tool, own_tool)
        await seeded.link_toolsets([(agent_key, shared), (other_key, shared), (agent_key, own)], [(shared, shared_tool), (own, own_tool)])
        result = await store.hard_delete_agent(agent_key)
        assert (result["agents_deleted"], result["toolsets_deleted"], result["tools_deleted"]) == (1, 1, 1)
        assert set(await seeded.present()) == {other_key, shared, shared_tool}
        assert await store.get_edge(shared, C.AGENT_TOOLSETS.value, shared_tool, C.AGENT_TOOLS.value, C.TOOLSET_HAS_TOOL.value) is not None
        assert await store.get_edge(other_key, C.AGENT_INSTANCES.value, shared, C.AGENT_TOOLSETS.value, C.AGENT_HAS_TOOLSET.value) is not None
    finally:
        await seeded.remove()


@pytest.mark.parametrize("deleted_holder", ["toolset", "mcp_server"])
async def test_a_tool_another_kind_of_holder_links_stays(store: IGraphDBProvider, deleted_holder: str) -> None:
    """A tool held by a toolset and by an MCP server, each used by a different agent. Deleting the agent of one holder takes
    that holder; the tool and the other holder's link to it stay."""
    seeded = Seeded(store)
    run = uuid.uuid4().hex[:10]
    agent_key, other_key = (seeded.key(f"{k}-kind-{run}", C.AGENT_INSTANCES.value) for k in ("agent", "other"))
    toolset = seeded.key(f"ts-kind-{run}", C.AGENT_TOOLSETS.value)
    server = seeded.key(f"s-kind-{run}", C.AGENT_MCP_SERVERS.value)
    tool = seeded.key(f"tool-kind-{run}", C.AGENT_TOOLS.value)
    toolset_agent, server_agent = (agent_key, other_key) if deleted_holder == "toolset" else (other_key, agent_key)
    try:
        await seeded.agents(agent_key, other_key)
        await seeded.toolsets(toolset)
        await seeded.servers(server)
        await seeded.tools(tool)
        await seeded.link_toolsets([(toolset_agent, toolset)], [(toolset, tool)])
        await seeded.link([(server_agent, server)], [(server, tool)])
        result = await store.hard_delete_agent(agent_key)
        assert (result["agents_deleted"], result["tools_deleted"]) == (1, 0)
        kept = {other_key, tool, server if deleted_holder == "toolset" else toolset}
        assert set(await seeded.present()) == kept
        link = ((server, C.AGENT_MCP_SERVERS.value, C.MCP_SERVER_HAS_TOOL.value) if deleted_holder == "toolset"
                else (toolset, C.AGENT_TOOLSETS.value, C.TOOLSET_HAS_TOOL.value))
        assert await store.get_edge(link[0], link[1], tool, C.AGENT_TOOLS.value, link[2]) is not None
    finally:
        await seeded.remove()


@pytest.mark.arango_only
@pytest.mark.parametrize("missing", ["toolset_tool", "server_tool", "toolset", "server"])
async def test_a_link_to_a_missing_node_does_not_block_the_delete(store: IGraphDBProvider, missing: str) -> None:
    """ArangoDB keeps an edge whose node is gone (a Neo4j relationship cannot outlive its node, so the case is ArangoDB only). The delete removes the dangling link and still deletes the
    agent, instead of failing on the missing node and rolling back the link it would need to retry."""
    seeded = Seeded(store)
    run = uuid.uuid4().hex[:10]
    agent_key = seeded.key(f"agent-gone-{run}", C.AGENT_INSTANCES.value)
    toolset, server = f"ts-gone-{run}", f"s-gone-{run}"
    gone = f"gone-{run}"
    try:
        await seeded.agents(agent_key)
        if missing == "toolset":
            links = [(C.AGENT_INSTANCES.value, agent_key, C.AGENT_TOOLSETS.value, gone, C.AGENT_HAS_TOOLSET.value)]
        elif missing == "server":
            links = [(C.AGENT_INSTANCES.value, agent_key, C.AGENT_MCP_SERVERS.value, gone, C.AGENT_HAS_MCP_SERVER.value)]
        elif missing == "toolset_tool":
            await seeded.toolsets(seeded.key(toolset, C.AGENT_TOOLSETS.value))
            links = [(C.AGENT_INSTANCES.value, agent_key, C.AGENT_TOOLSETS.value, toolset, C.AGENT_HAS_TOOLSET.value),
                     (C.AGENT_TOOLSETS.value, toolset, C.AGENT_TOOLS.value, gone, C.TOOLSET_HAS_TOOL.value)]
        else:
            await seeded.servers(seeded.key(server, C.AGENT_MCP_SERVERS.value))
            links = [(C.AGENT_INSTANCES.value, agent_key, C.AGENT_MCP_SERVERS.value, server, C.AGENT_HAS_MCP_SERVER.value),
                     (C.AGENT_MCP_SERVERS.value, server, C.AGENT_TOOLS.value, gone, C.MCP_SERVER_HAS_TOOL.value)]
        for fc, fi, tc, ti, coll in links:
            assert await store.batch_create_edges([_edge(fc, fi, tc, ti)], coll)
        result = await store.hard_delete_agent(agent_key)
        assert result["agents_deleted"] == 1
        assert await seeded.present() == []
        assert [link for link in links if await store.get_edge(link[1], link[0], link[3], link[2], link[4]) is not None] == []
    finally:
        await seeded.remove()

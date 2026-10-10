"""Keep iterative research from spending its budget on identical searches."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from deepsearcher.agent.deep_search import DeepSearch


@pytest.mark.parametrize(
    "initial,gaps,expected_queries,expected_tokens",
    [
        (["first", "first"], [[]], ["first"], 13),
        (["first"], [["first"]], ["first"], 13),
        (["first"], [["first", "second", "second"], []], ["first", "second"], 25),
    ],
)
def test_searches_each_query_once(initial, gaps, expected_queries, expected_tokens):
    agent = DeepSearch.__new__(DeepSearch)
    agent.max_iter = 4
    agent._generate_sub_queries = Mock(return_value=(initial, 1))
    agent._generate_gap_queries = Mock(
        side_effect=[(queries, 2) for queries in gaps] + [(gaps[-1], 2)] * 4
    )
    agent._search_chunks_from_vectordb = AsyncMock(return_value=([], 10))

    results, tokens, metadata = asyncio.run(agent.async_retrieve("question"))

    assert results == []
    assert tokens == expected_tokens
    assert metadata["all_sub_queries"] == expected_queries
    assert [
        call.args[0] for call in agent._search_chunks_from_vectordb.await_args_list
    ] == expected_queries

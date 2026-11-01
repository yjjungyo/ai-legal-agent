"""Langfuse configuration and callback propagation, without external services."""

from collections import Counter
from contextlib import ExitStack
from contextvars import ContextVar
import logging
from threading import Lock
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool

from src.agents import workflow
from src.utils import tracing


class LangfuseConfigTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(tracing.logger, "level", logging.CRITICAL))
        self.stack.enter_context(patch.multiple(
            tracing,
            LANGFUSE_ENABLED=True,
            LANGFUSE_CAPTURE_CONTENT=True,
            LANGFUSE_HOST="https://langfuse.example.test",
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
            LANGFUSE_SYSTEM_NAME="legal-test",
        ))
        self.constructor = self.stack.enter_context(patch("langfuse.Langfuse"))
        self.client = self.constructor.return_value
        self.callback = self.stack.enter_context(patch("langfuse.callback.CallbackHandler"))
        tracing._get_langfuse_client.cache_clear()
        self.addCleanup(tracing._get_langfuse_client.cache_clear)

    def test_disabled_preserves_existing_graph_config(self):
        with patch.object(tracing, "LANGFUSE_ENABLED", False):
            with tracing.langfuse_config("chat-1") as config:
                self.assertEqual(config, {"configurable": {"thread_id": "chat-1"}})
        self.constructor.assert_not_called()
        self.callback.assert_not_called()

    def test_missing_credentials_preserve_existing_graph_config(self):
        for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            with self.subTest(key=key), patch.object(tracing, key, ""):
                tracing._get_langfuse_client.cache_clear()
                with tracing.langfuse_config("chat-1") as config:
                    self.assertNotIn("callbacks", config)
        self.constructor.assert_not_called()

    def test_client_reused_but_handlers_and_sessions_are_per_request(self):
        handlers = [MagicMock(), MagicMock()]
        self.callback.side_effect = handlers
        for thread_id, handler in zip(("chat-1", "chat-2"), handlers):
            with tracing.langfuse_config(thread_id) as config:
                self.assertEqual(config["configurable"]["thread_id"], thread_id)
                self.assertEqual(config["callbacks"], [handler])
                self.assertEqual(config["run_name"], "legal-test")
                self.assertEqual(config["tags"], ["legal-test"])
        self.constructor.assert_called_once()
        self.assertEqual(self.client.flush.call_count, 2)
        self.assertEqual(self.callback.call_count, 2)
        self.assertTrue(all(
            call.kwargs["tags"] == ["legal-test"]
            for call in self.client.trace.call_args_list
        ))
        self.assertEqual(
            [call.kwargs["session_id"] for call in self.client.trace.call_args_list],
            ["chat-1", "chat-2"],
        )

    def test_content_capture_controls_sdk_mask(self):
        for capture in (False, True):
            with self.subTest(capture=capture), patch.object(tracing, "LANGFUSE_CAPTURE_CONTENT", capture):
                tracing._get_langfuse_client.cache_clear()
                with tracing.langfuse_config("chat-1"):
                    pass
                mask = self.constructor.call_args.kwargs.get("mask")
                if capture:
                    self.assertIsNone(mask)
                else:
                    self.assertEqual(mask(data={"question": "private"}), "[REDACTED]")

    def test_sdk_initialization_failure_does_not_stop_the_request(self):
        self.constructor.side_effect = RuntimeError("initialization failed")
        with tracing.langfuse_config("chat-1") as config:
            self.assertEqual(config, {"configurable": {"thread_id": "chat-1"}})

    def test_callback_initialization_failure_does_not_stop_the_request(self):
        self.callback.side_effect = RuntimeError("callback failed")
        with tracing.langfuse_config("chat-1") as config:
            self.assertNotIn("callbacks", config)

    def test_flush_failure_preserves_successful_result(self):
        self.client.flush.side_effect = RuntimeError("export failed")
        with tracing.langfuse_config("chat-1"):
            result = "original answer"
        self.assertEqual(result, "original answer")
        self.client.flush.assert_called_once()

    def test_original_request_error_survives_flush_failure(self):
        original_error = ValueError("agent failed")
        self.client.flush.side_effect = RuntimeError("export failed")
        with self.assertRaises(ValueError) as caught:
            with tracing.langfuse_config("chat-1"):
                raise original_error
        self.assertIs(caught.exception, original_error)
        self.client.flush.assert_called_once()


class CallbackRecorder(BaseCallbackHandler):
    def __init__(self):
        self.events = []
        self.lock = Lock()

    def _record(self, kind, run_id, parent_run_id, name):
        with self.lock:
            self.events.append((kind, run_id, parent_run_id, name))

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self._record("chain", run_id, parent_run_id, kwargs.get("name"))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self._record("tool", run_id, parent_run_id, serialized["name"])

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self._record("llm", run_id, parent_run_id, kwargs.get("name"))


class ParallelWorkflowTracingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(workflow.logger, "level", logging.CRITICAL))
        self.marker = ContextVar("trace_test_marker", default=None)
        self.seen_markers = []
        self.marker_lock = Lock()

        def route(question):
            FakeListChatModel(responses=["route"]).invoke(question)
            return ["civil", "labor"]

        def make_agent(category):
            def retrieve(question):
                with self.marker_lock:
                    self.seen_markers.append(self.marker.get())
                FakeListChatModel(responses=["rewritten query"]).invoke(question)
                return category + " evidence"

            tool = Tool(name=category + "_retrieval", description="test retrieval", func=retrieve)
            return RunnableLambda(
                lambda inputs: {"output": tool.invoke(inputs["input"])},
                name=category + "_agent",
            )

        agents = {category: make_agent(category) for category in ("civil", "labor")}
        self.stack.enter_context(patch.object(workflow, "route_categories", route))
        self.stack.enter_context(patch.object(workflow, "plan_tools", lambda categories: categories))
        self.stack.enter_context(patch.object(workflow, "_GET_AGENT", {
            category: (lambda agent=agent: agent) for category, agent in agents.items()
        }))

        def synthesize(messages, results):
            answer = FakeListChatModel(responses=["final answer"]).invoke(str(results))
            return messages + [AIMessage(content=answer.content)]

        self.stack.enter_context(patch.object(workflow, "synthesize", synthesize))
        self.graph = workflow.build_legal_graph()

    def test_agent_tool_and_llm_callbacks_keep_graph_parent_in_worker_threads(self):
        recorder = CallbackRecorder()
        marker_token = self.marker.set("request-1")
        try:
            result = self.graph.invoke(
                {"messages": [("user", "test question")]},
                config={"configurable": {"thread_id": "trace-test"}, "callbacks": [recorder]},
            )
        finally:
            self.marker.reset(marker_token)

        self.assertEqual(result["messages"][-1].content, "final answer")
        self.assertEqual(result["results"], {"civil": "civil evidence", "labor": "labor evidence"})
        self.assertEqual(self.seen_markers, ["request-1", "request-1"])

        chains = {name: (run_id, parent_id) for kind, run_id, parent_id, name in recorder.events if kind == "chain"}
        agents = [event for event in recorder.events if event[0] == "chain" and event[3] in {"civil_agent", "labor_agent"}]
        self.assertEqual(Counter(event[3] for event in agents), {"civil_agent": 1, "labor_agent": 1})
        for _, _, parent_id, _ in agents:
            self.assertEqual(parent_id, chains["run_agents"][0])
        tools = [event for event in recorder.events if event[0] == "tool"]
        self.assertEqual(len(tools), 2)
        for _, _, parent_id, name in tools:
            self.assertEqual(parent_id, chains[name.replace("_retrieval", "_agent")][0])
        llms = [event for event in recorder.events if event[0] == "llm"]
        self.assertEqual(len(llms), 4)
        self.assertEqual(sum(event[2] is None for event in recorder.events), 1)

    def test_agent_failure_is_preserved_by_parallel_workflow(self):
        error = ValueError("original agent error")

        def fail(inputs):
            raise error

        failing_agent = RunnableLambda(fail, name="failing_agent")
        with patch.object(workflow, "_GET_AGENT", {"civil": lambda: failing_agent, "labor": lambda: failing_agent}):
            with self.assertRaises(ValueError) as caught:
                self.graph.invoke(
                    {"messages": [("user", "test question")]},
                    config={"configurable": {"thread_id": "failure-test"}, "callbacks": [CallbackRecorder()]},
                )
        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()

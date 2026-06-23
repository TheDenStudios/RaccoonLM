import json
import unittest
from unittest.mock import patch

from raccoonlm.core.models import _lmstudio_chat_response_has_output, _llamacpp_chat_response_has_output
from raccoonlm.core import streaming


class LMStudioLoadVerificationTests(unittest.TestCase):
    def test_rejects_empty_chat_completion(self):
        data = {
            "choices": [{"message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 0},
        }
        self.assertFalse(_lmstudio_chat_response_has_output(data))

    def test_accepts_visible_content(self):
        data = {"choices": [{"message": {"content": "ok"}}]}
        self.assertTrue(_lmstudio_chat_response_has_output(data))

    def test_accepts_reasoning_or_positive_completion_tokens(self):
        reasoning = {"choices": [{"message": {"reasoning_content": "thinking"}}]}
        tokens = {
            "choices": [{"message": {"content": ""}}],
            "usage": {"completion_tokens": 1},
        }
        self.assertTrue(_lmstudio_chat_response_has_output(reasoning))
        self.assertTrue(_lmstudio_chat_response_has_output(tokens))


class LlamaCppLoadVerificationTests(unittest.TestCase):
    def test_accepts_llamacpp_openai_content(self):
        data = {"choices": [{"message": {"content": "ok"}}]}
        self.assertTrue(_llamacpp_chat_response_has_output(data))

    def test_rejects_empty_llamacpp_response(self):
        data = {"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 0}}
        self.assertFalse(_llamacpp_chat_response_has_output(data))


class LMStudioStreamingRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_chat_uses_lmstudio_provider_and_emits_tokens(self):
        calls = []

        async def fake_lmstudio_stream(model, messages, options=None):
            calls.append((model, messages, options))
            yield {"message": {"content": "allo", "thinking": ""}, "done": False}
            yield {"done": True, "prompt_eval_count": 2, "eval_count": 1}

        with patch.object(streaming, "_lmstudio_raw_stream", fake_lmstudio_stream):
            events = []
            async for event in streaming.stream_chat(
                "lm-model",
                [{"role": "user", "content": "test"}],
                tools=[{"type": "function"}],
                provider="lmstudio",
            ):
                events.append(event)

        self.assertEqual(calls[0][0], "lm-model")
        self.assertIn('event: token', ''.join(events))
        self.assertIn('"token": "allo"', ''.join(events))
        self.assertIn('event: done', ''.join(events))


class LlamaCppStreamingRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_chat_uses_llamacpp_provider_and_emits_tokens(self):
        calls = []

        async def fake_llamacpp_stream(model, messages, options=None):
            calls.append((model, messages, options))
            yield {"message": {"content": "rapide", "thinking": ""}, "done": False}
            yield {"done": True, "prompt_eval_count": 2, "eval_count": 1}

        with patch.object(streaming, "_llamacpp_raw_stream", fake_llamacpp_stream):
            events = []
            async for event in streaming.stream_chat(
                "gguf-model",
                [{"role": "user", "content": "test"}],
                tools=[{"type": "function"}],
                provider="llamacpp",
            ):
                events.append(event)

        self.assertEqual(calls[0][0], "gguf-model")
        self.assertIn('event: token', ''.join(events))
        self.assertIn('"token": "rapide"', ''.join(events))
        self.assertIn('event: done', ''.join(events))


if __name__ == "__main__":
    unittest.main()

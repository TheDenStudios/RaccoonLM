import json
import unittest
from unittest.mock import patch

from raccoonlm.core.models import _llamacpp_chat_response_has_output
from raccoonlm.core import streaming


class LlamaCppLoadVerificationTests(unittest.TestCase):
    def test_accepts_llamacpp_openai_content(self):
        data = {"choices": [{"message": {"content": "ok"}}]}
        self.assertTrue(_llamacpp_chat_response_has_output(data))

    def test_rejects_empty_llamacpp_response(self):
        data = {"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 0}}
        self.assertFalse(_llamacpp_chat_response_has_output(data))


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

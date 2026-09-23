"""Check the mock service against the SDK and EvalScope's streaming collector."""

import threading
from http.server import ThreadingHTTPServer

from evalscope.models.utils.openai import collect_stream_response
from openai import OpenAI

from scripts.verify_mock import Handler


def test_mock_chat_completions_contract():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="EMPTY")
        request = {"model": "mock-model", "messages": [{"role": "user", "content": "Answer."}]}
        normal = client.chat.completions.create(**request)
        assert normal.id.startswith("chatcmpl-mock-")
        assert normal.model == "mock-model"
        assert normal.choices[0].message.content == r"\boxed{42}"

        stream = client.chat.completions.create(**request, stream=True, stream_options={"include_usage": True})
        collected, ttft = collect_stream_response(stream)
        assert collected.id.startswith("chatcmpl-mock-")
        assert collected.model == "mock-model"
        assert collected.choices[0].message.content == r"\boxed{42}"
        assert collected.usage.completion_tokens == 3
        assert ttft is not None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

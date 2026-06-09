"""RaccoonLM v2 — OpenAI-compatible API server (port 5556)"""
import json, threading, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_running = False
_server = None
_port = 5556
_current_model = "qwen3:4b"

def start(port=5556, model="qwen3:4b"):
    global _running, _server, _port, _current_model
    if _running:
        return {"error": "Already running", "status": "error"}
    _port, _current_model = port, model
    _running = True

    from flask import Flask, request, jsonify, Response
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    @app.route('/v1/models')
    def list_models():
        return jsonify({
            "object": "list",
            "data": [{"id": _current_model, "object": "model", "created": 0, "owned_by": "raccoonlm"}]
        })

    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        data = request.json or {}
        msgs = data.get('messages', [])
        stream = data.get('stream', False)
        temp = data.get('temperature')

        # Import from our async modules
        from raccoonlm.core.llm import chat_sync
        import asyncio

        if stream:
            from raccoonlm.core.streaming import stream_chat
            def generate():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                async def _run():
                    async for event in stream_chat(_current_model, msgs, [], "", None, None, None):
                        yield f"data: {json.dumps({'choices':[{'delta':{'content':''},'index':0}]})}\n\n"
                loop.run_until_complete(_run())
            return Response(generate(), mimetype='text/event-stream')
        else:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resp = loop.run_until_complete(chat_sync(_current_model, msgs, [], temp))
                loop.close()
                content = resp.message.content or ""
                return jsonify({
                    "id": "chatcmpl-raccoonlm",
                    "object": "chat.completion",
                    "created": 0,
                    "model": _current_model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": getattr(resp, "prompt_eval_count", 0),
                              "completion_tokens": getattr(resp, "eval_count", 0),
                              "total_tokens": (getattr(resp, "prompt_eval_count", 0)or 0) + (getattr(resp, "eval_count", 0)or 0)},
                })
            except Exception as e:
                return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500

    from werkzeug.serving import make_server
    _server = make_server('0.0.0.0', _port, app, threaded=True)
    t = threading.Thread(target=_server.serve_forever, daemon=True)
    t.start()
    return {"status": "started", "port": _port, "url": f"http://localhost:{_port}/v1", "model": _current_model}

def stop():
    global _running, _server
    if not _running:
        return {"error": "Not running", "status": "error"}
    try:
        if _server: _server.shutdown()
    except: pass
    _running = False
    _server = None
    return {"status": "stopped"}

def is_running():
    return _running

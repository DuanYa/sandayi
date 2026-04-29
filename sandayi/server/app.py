from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_sock import Sock

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

from sandayi.ai.client import AIPlayer
from sandayi.model.game import RuleError
from sandayi.server.room_manager import RoomManager

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "static"

manager = RoomManager()


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
    sock = Sock(app)

    def spawn_ai(room_id: str, model_name: str | None = None) -> None:
        ai = AIPlayer(server_url="ws://127.0.0.1:5000", room_id=room_id, model_name=model_name)
        ai.start()

    manager.set_ai_factory(spawn_ai)

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.post("/api/rooms")
    def create_room():
        room = manager.create_room()
        return jsonify({"room_id": room.id})

    @app.get("/api/rooms/<room_id>")
    def room_state(room_id: str):
        try:
            room = manager.get_room(room_id)
            return jsonify(room.game.public_view())
        except RuleError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/rooms/<room_id>/join")
    def join_room(room_id: str):
        data = request.get_json(force=True) or {}
        name = data.get("name") or "玩家"
        is_ai = bool(data.get("is_ai", False))
        player_id = data.get("player_id")
        try:
            return jsonify(manager.join_room(room_id, name, is_ai=is_ai, player_id=player_id))
        except RuleError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/checkpoints")
    def checkpoints():
        model_dir = ROOT / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(path.name for path in model_dir.iterdir() if path.suffix in {".pt", ".pth"})
        return jsonify({"models": ["rule", "transformer"] + files})

    @app.post("/api/rooms/<room_id>/ai")
    def add_ai(room_id: str):
        data = request.get_json(silent=True) or {}
        model_name = data.get("model") or "transformer"
        try:
            count = manager.fill_ai(room_id, model_name=model_name)
            return jsonify({"added": count, "model": model_name})
        except RuleError as exc:
            return jsonify({"error": str(exc)}), 400

    @sock.route("/ws/<room_id>/<player_id>")
    def websocket(ws, room_id: str, player_id: str):
        manager.attach_socket(room_id, player_id, ws)
        while True:
            raw = ws.receive()
            if raw is None:
                break
            try:
                payload: dict[str, Any] = json.loads(raw)
                manager.handle_action(room_id, player_id, payload)
            except RuleError as exc:
                ws.send(json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False))
                try:
                    room = manager.get_room(room_id)
                    ws.send(json.dumps({"type": "state", "state": room.game.public_view(player_id)}, ensure_ascii=False))
                except RuleError:
                    pass
            except Exception as exc:
                ws.send(json.dumps({"type": "error", "message": f"服务器错误: {exc}"}, ensure_ascii=False))
        manager.detach_socket(room_id, player_id)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

from __future__ import annotations

import json
import logging
import random
import threading
import time
import uuid
from typing import Any

import requests
import websocket

from sandayi.ai.strategy import create_ai_strategy

logger = logging.getLogger(__name__)


class AIPlayer:
    def __init__(self, server_url: str, room_id: str, model_name: str | None = None) -> None:
        self.server_url = server_url.rstrip("/")
        self.http_url = self.server_url.replace("ws://", "http://").replace("wss://", "https://")
        self.room_id = room_id
        self.model_name = model_name or "transformer"
        self.player_id = f"ai_{uuid.uuid4().hex[:8]}"
        self.name = f"AI-{self.model_name}-{self.player_id[-4:]}"[:16]
        self.thread = threading.Thread(target=self.run, name=f"AIPlayer-{self.player_id}", daemon=True)
        self.ws: websocket.WebSocketApp | None = None
        self.pending_keys: set[tuple[int, str]] = set()
        self.latest_version = -1
        self.lock = threading.Lock()
        self.strategy = create_ai_strategy(self.model_name)

    def start(self) -> None:
        self.thread.start()

    def run(self) -> None:
        try:
            if not self._join_room():
                return
            ws_url = f"{self.server_url}/ws/{self.room_id}/{self.player_id}"
            self.ws = websocket.WebSocketApp(ws_url, on_message=self._on_message, on_error=self._on_error, on_close=self._on_close)
            self.ws.run_forever()
        except Exception:
            logger.exception("AI 运行失败 player=%s room=%s", self.player_id, self.room_id)

    def _join_room(self) -> bool:
        url = f"{self.http_url}/api/rooms/{self.room_id}/join"
        response = requests.post(url, json={"name": self.name, "is_ai": True, "player_id": self.player_id}, timeout=5)
        logger.info("AI 加入房间 player=%s room=%s status=%s body=%s", self.player_id, self.room_id, response.status_code, response.text)
        return response.status_code == 200

    def _on_message(self, ws: websocket.WebSocketApp, raw: str) -> None:
        payload = json.loads(raw)
        if payload.get("type") != "state":
            if payload.get("type") == "error":
                logger.warning("AI 收到错误 player=%s payload=%s", self.player_id, payload)
            return
        state = payload["state"]
        version = int(state.get("version", 0))
        legal = state.get("legal_actions", {})
        action_type = legal.get("type", "wait")
        key = (version, action_type)
        with self.lock:
            self.latest_version = max(self.latest_version, version)
            if action_type == "wait":
                self.pending_keys.clear()
                return
            if key in self.pending_keys:
                return
            self.pending_keys.add(key)
        logger.info("AI 收到可行动状态 player=%s version=%s action_type=%s legal=%s", self.player_id, version, action_type, legal)
        threading.Thread(target=self._delayed_act, args=(ws, state, key), name=f"AIAction-{self.player_id}-{version}-{action_type}", daemon=True).start()

    def _delayed_act(self, ws: websocket.WebSocketApp, state: dict[str, Any], key: tuple[int, str]) -> None:
        time.sleep(random.uniform(0.25, 0.8))
        version, _ = key
        with self.lock:
            if version < self.latest_version:
                self.pending_keys.discard(key)
                return
        action = self._decide(state)
        with self.lock:
            self.pending_keys.discard(key)
            if version < self.latest_version:
                return
        if not action:
            logger.warning("AI 无可执行动作 player=%s version=%s legal=%s hand=%s", self.player_id, version, state.get("legal_actions"), state.get("hand"))
            return
        action["version"] = version
        try:
            logger.info("AI 发送动作 player=%s room=%s version=%s action=%s", self.player_id, self.room_id, version, action)
            ws.send(json.dumps(action, ensure_ascii=False))
        except Exception:
            logger.exception("AI 发送动作失败 player=%s version=%s", self.player_id, version)

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        logger.warning("AI WebSocket 错误 player=%s error=%s", self.player_id, error)

    def _on_close(self, ws: websocket.WebSocketApp, code: int, reason: str) -> None:
        logger.warning("AI WebSocket 关闭 player=%s code=%s reason=%s", self.player_id, code, reason)

    def _decide(self, state: dict[str, Any]) -> dict[str, Any] | None:
        return self.strategy.decide(state)
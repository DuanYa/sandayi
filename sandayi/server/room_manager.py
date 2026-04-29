from __future__ import annotations

import json
import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from flask_sock import Server

from sandayi.model.game import RuleError, SandayiGame

logger = logging.getLogger(__name__)


@dataclass
class Room:
    id: str
    game: SandayiGame = field(default_factory=SandayiGame)
    sockets: dict[str, Server] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    last_action_at: float = field(default_factory=time.monotonic)
    watchdog_started: bool = False


class RoomManager:
    def __init__(self) -> None:
        self.rooms: dict[str, Room] = {}
        # 全局锁只保护房间字典，单个房间状态由房间自己的锁保护。
        self.lock = threading.RLock()
        self.ai_factory = None

    def set_ai_factory(self, factory: Any) -> None:
        self.ai_factory = factory

    def create_room(self) -> Room:
        room_id = uuid.uuid4().hex[:6]
        with self.lock:
            room = Room(room_id)
            self.rooms[room_id] = room
            logger.info("创建房间 room=%s", room_id)
            return room

    def get_room(self, room_id: str) -> Room:
        with self.lock:
            if room_id not in self.rooms:
                raise RuleError("房间不存在")
            return self.rooms[room_id]

    def join_room(self, room_id: str, name: str, is_ai: bool = False, player_id: str | None = None) -> dict[str, str]:
        room = self.get_room(room_id)
        pid = player_id or uuid.uuid4().hex[:8]
        with room.lock:
            room.game.add_player(pid, name, is_ai)
            logger.info("玩家加入 room=%s player=%s name=%s is_ai=%s count=%s", room_id, pid, name, is_ai, len(room.game.state.players))
            room.last_action_at = time.monotonic()
            self.broadcast(room)
        return {"room_id": room_id, "player_id": pid, "name": name}

    def fill_ai(self, room_id: str, model_name: str | None = None) -> int:
        room = self.get_room(room_id)
        with room.lock:
            if not any(not player.is_ai for player in room.game.state.players):
                raise RuleError("房间内至少需要一名真实玩家才能补齐 AI")
            need = 4 - len(room.game.state.players)
        if need <= 0:
            return 0
        if self.ai_factory is None:
            raise RuleError("AI 工厂未初始化")
        logger.info("补齐 AI room=%s need=%s model=%s", room_id, need, model_name)
        for _ in range(need):
            self.ai_factory(room_id, model_name)
        self._ensure_watchdog(room)
        return need

    def handle_action(self, room_id: str, player_id: str, action: dict[str, Any]) -> None:
        room = self.get_room(room_id)
        with room.lock:
            if not any(player.id == player_id for player in room.game.state.players):
                raise RuleError("玩家不在房间中")
            action_type = action.get("type")
            client_version = action.get("version")
            current_version = room.game.state.version
            if client_version is not None and client_version != current_version:
                logger.warning("拒绝过期动作 room=%s player=%s action=%s client_version=%s current_version=%s", room_id, player_id, action, client_version, current_version)
                if player_id in room.sockets:
                    self._send(room.sockets[player_id], {"type": "state", "state": room.game.public_view(player_id)})
                return
            before = room.game.public_view().get("turn_player_id")
            logger.info("收到动作 room=%s player=%s action=%s phase=%s turn=%s version=%s", room_id, player_id, action, room.game.state.phase.value, before, current_version)
            if action_type == "start":
                room.game.start()
            elif action_type == "restart":
                room.game.start()
            elif action_type == "bid":
                room.game.place_bid(player_id, action.get("bid"))
            elif action_type == "bury":
                room.game.bury_kitty(player_id, action.get("cards", []))
            elif action_type == "trump":
                room.game.choose_trump(player_id, action.get("suit"))
            elif action_type == "play":
                room.game.play_card(player_id, action.get("card"))
            else:
                raise RuleError("未知操作")
            after = room.game.public_view().get("turn_player_id")
            room.last_action_at = time.monotonic()
            logger.info("动作完成 room=%s player=%s action_type=%s phase=%s turn=%s version=%s", room_id, player_id, action_type, room.game.state.phase.value, after, room.game.state.version)
            self.broadcast(room)
    def attach_socket(self, room_id: str, player_id: str, ws: Server) -> None:
        room = self.get_room(room_id)
        with room.lock:
            room.sockets[player_id] = ws
            logger.info("WebSocket 接入 room=%s player=%s sockets=%s", room_id, player_id, len(room.sockets))
            self._send(ws, {"type": "state", "state": room.game.public_view(player_id)})

    def detach_socket(self, room_id: str, player_id: str) -> None:
        try:
            room = self.get_room(room_id)
        except RuleError:
            return
        with room.lock:
            room.sockets.pop(player_id, None)
            logger.info("WebSocket 断开 room=%s player=%s sockets=%s", room_id, player_id, len(room.sockets))

    def broadcast(self, room: Room) -> None:
        stale: list[str] = []
        logger.info("广播状态 room=%s sockets=%s phase=%s turn=%s", room.id, len(room.sockets), room.game.state.phase.value, room.game.public_view().get("turn_player_id"))
        for pid, ws in list(room.sockets.items()):
            ok = self._send(ws, {"type": "state", "state": room.game.public_view(pid)})
            if not ok:
                stale.append(pid)
        for pid in stale:
            room.sockets.pop(pid, None)


    def _ensure_watchdog(self, room: Room) -> None:
        if room.watchdog_started:
            return
        room.watchdog_started = True
        thread = threading.Thread(target=self._watch_room, args=(room.id,), name=f"RoomWatchdog-{room.id}", daemon=True)
        thread.start()

    def _watch_room(self, room_id: str) -> None:
        while True:
            time.sleep(1.5)
            try:
                room = self.get_room(room_id)
            except RuleError:
                return
            with room.lock:
                phase = room.game.state.phase.value
                if phase in ("waiting", "finished"):
                    if phase == "finished":
                        room.watchdog_started = False
                        return
                    continue
                current_id = room.game.public_view().get("turn_player_id")
                if not current_id:
                    continue
                try:
                    current_player = room.game._player(current_id)
                except RuleError:
                    continue
                idle = time.monotonic() - room.last_action_at
                if current_player.is_ai and idle >= 2.5:
                    logger.warning("AI 回合超时重推状态 room=%s player=%s phase=%s version=%s idle=%.2f", room_id, current_id, phase, room.game.state.version, idle)
                    ws = room.sockets.get(current_id)
                    if ws:
                        self._send(ws, {"type": "state", "state": room.game.public_view(current_id)})
                    room.last_action_at = time.monotonic()

    def _send(self, ws: Server, payload: dict[str, Any]) -> bool:
        try:
            ws.send(json.dumps(payload, ensure_ascii=False))
            return True
        except Exception:
            logger.exception("WebSocket 发送失败")
            return False

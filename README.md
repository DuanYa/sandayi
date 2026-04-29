# 三打一扑克游戏

## 启动

```bash
conda activate sandayi
pip install -r requirements.txt
python -m sandayi.server.app
```

打开 `http://127.0.0.1:5000`。

## 说明

- 规则模型在 `sandayi/model`，不依赖 Flask。
- 服务层在 `sandayi/server`，负责 HTTP、WebSocket 和房间管理。
- AI 玩家在 `sandayi/ai`，以 WebSocket 客户端方式加入房间。
- 详细方案见 `设计文档.md`。

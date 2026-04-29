let roomId = localStorage.getItem('roomId') || '';
let playerId = localStorage.getItem('playerId') || '';
let ws = null;
let state = null;
let selected = new Set();
let lastRenderedVersion = -1;

const $ = (id) => document.getElementById(id);
$('roomInput').value = roomId;

$('createBtn').onclick = async () => {
  const res = await fetch('/api/rooms', { method: 'POST' });
  const data = await res.json();
  roomId = data.room_id;
  $('roomInput').value = roomId;
  localStorage.setItem('roomId', roomId);
  log(`已创建房间 ${roomId}，正在自动加入`);
  await joinCurrentRoom();
};

$('joinBtn').onclick = async () => {
  roomId = $('roomInput').value.trim();
  if (!roomId) return showMsg('lobbyMsg', '请先输入或创建房间号');
  await joinCurrentRoom();
};

async function joinCurrentRoom() {
  const name = $('nameInput').value.trim() || '玩家';
  const res = await fetch(`/api/rooms/${roomId}/join`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
  const data = await res.json();
  if (data.error) return showMsg('lobbyMsg', data.error);
  playerId = data.player_id;
  localStorage.setItem('roomId', roomId);
  localStorage.setItem('playerId', playerId);
  connect();
}
$('aiBtn').onclick = async () => {
  if (!roomId) return showMsg('waitingMsg', '请先创建或加入房间');
  const model = $('aiModelSelect')?.value || 'transformer';
  const res = await fetch(`/api/rooms/${roomId}/ai`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }) });
  const data = await res.json();
  showMsg('waitingMsg', data.error || `已按 ${data.model} 添加 ${data.added} 个 AI`);
};

async function loadAiModels() {
  const select = $('aiModelSelect');
  if (!select) return;
  try {
    const res = await fetch('/api/checkpoints');
    const data = await res.json();
    const models = data.models || ['rule', 'transformer'];
    const labels = { rule: '规则 AI', transformer: 'Transformer 强化学习 AI' };
    select.innerHTML = models.map(model => `<option value="${model}">${labels[model] || model}</option>`).join('');
  } catch (err) {
    log(`读取 AI 模型列表失败：${err}`, true);
  }
}
$('startBtn').onclick = () => send({ type: 'start' });
$('restartBtn').onclick = () => send({ type: 'restart' });
$('backLobbyBtn').onclick = () => showScreen('lobby');
$('closeResultBtn').onclick = () => $('resultDialog').classList.add('hidden');

function connect() {
  if (ws) ws.close();
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${scheme}://${location.host}/ws/${roomId}/${playerId}`;
  log(`连接 WebSocket：${url}`);
  ws = new WebSocket(url);
  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === 'state') {
      const incoming = payload.state;
      if (state && incoming.version < state.version) {
        log(`忽略旧状态 version=${incoming.version} current=${state.version}`);
        return;
      }
      state = incoming;
      log(`收到状态 version=${state.version} phase=${state.phase} turn=${nameOf(state.turn_player_id) || '-'}`);
      render();
    } else if (payload.type === 'error') {
      log(`服务端错误：${payload.message}`, true);
    }
  };
  ws.onerror = () => log('WebSocket 发生错误，请查看浏览器控制台和后端日志', true);
  ws.onopen = () => log('WebSocket 已连接');
  ws.onclose = () => log('WebSocket 已断开', true);
}

if (roomId && playerId) connect();

function send(payload) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return log('尚未连接房间', true);
  if (state && typeof state.version === 'number') payload.version = state.version;
  log(`发送动作：${JSON.stringify(payload)}`);
  ws.send(JSON.stringify(payload));
}

function render() {
  if (!state) return;
  if (state.phase === 'waiting') {
    showScreen('waiting');
    loadAiModels();
  } else {
    showScreen('gameScreen');
  }
  renderWaiting();
  renderGameInfo();
  renderSeats();
  renderHand();
  renderActions();
  renderPlayedCards();
  renderResult();
  lastRenderedVersion = state.version;
}

function renderWaiting() {
  $('roomIdDisplay').textContent = roomId || '-';
  const players = state?.players || [];
  $('waitingSeats').innerHTML = [0, 1, 2, 3].map(i => {
    const p = players[i];
    return `<div class="seat-slot ${p ? 'occupied' : ''} ${p?.id === playerId ? 'is-me' : ''}">
      <div class="seat-label">座位 ${i + 1}</div>
      <div class="seat-name">${p ? p.name + (p.is_ai ? ' · AI' : '') : '等待中...'}</div>
    </div>`;
  }).join('');
}

function renderGameInfo() {
  if (!state) return;
  $('gRoom').textContent = state.id;
  $('gPhase').textContent = phaseName(state.phase);
  $('gTurn').textContent = nameOf(state.turn_player_id) || '-';
  $('gBanker').textContent = nameOf(state.banker_id) || '-';
  $('gTrump').textContent = suitName(state.trump) || '-';
  $('gScore').textContent = `${state.banker_points}/${state.farmers_points}`;
  $('gBid').textContent = state.highest_bid || '-';
}

function renderSeats() {
  const seats = getSeatPlayers();
  const plays = visibleTrickPlays();
  renderOtherSeat('seatTop', seats.top, false, plays);
  renderOtherSeat('seatLeft', seats.left, true, plays);
  renderOtherSeat('seatRight', seats.right, true, plays);
  renderBottomSeat(seats.bottom, plays);
}

function getSeatPlayers() {
  const players = state.players || [];
  if (!players.length) return { top: null, left: null, right: null, bottom: null };
  const meIndex = Math.max(0, players.findIndex(p => p.id === playerId));
  const ordered = [0, 1, 2, 3].map(offset => players[(meIndex + offset) % players.length]).filter(Boolean);
  return { bottom: ordered[0] || null, right: ordered[1] || null, top: ordered[2] || null, left: ordered[3] || null };
}

function visibleTrickPlays() {
  const current = state.current_trick;
  if (current && current.plays && current.plays.length) return current.plays;
  const last = state.last_trick;
  return last && last.plays ? last.plays : [];
}

function playedCardFor(playerIdToFind, plays) {
  return plays.find(p => p.player_id === playerIdToFind);
}

function playedCardHtml(player, plays, position) {
  if (!player) return '';
  const play = playedCardFor(player.id, plays);
  if (!play) return `<div class="seat-played ${position} empty"></div>`;
  return `<div class="seat-played ${position}">${cardHtml(play.card)}<span>${player.name}</span></div>`;
}

function renderOtherSeat(id, player, vertical, plays) {
  const el = $(id);
  if (!player) { el.innerHTML = ''; return; }
  el.innerHTML = `<div class="player-label ${player.id === state.turn_player_id ? 'active' : ''}">
    <span class="pname">${player.name}</span>${player.is_ai ? '<span class="tag">AI</span>' : ''}${player.id === state.banker_id ? '<span class="tag banker-tag">庄家</span>' : ''}<span class="pcount">${player.hand_count}张</span>
  </div><div class="seat-body ${vertical ? 'seat-body-side' : ''}"><div class="back-cards ${vertical ? 'vertical' : ''}">${renderBackCards(player.hand_count)}</div>${playedCardHtml(player, plays, id)}</div>`;
}

function renderBottomSeat(player, plays) {
  if (!player) return;
  $('myLabel').className = `player-label me ${player.id === state.turn_player_id ? 'active' : ''}`;
  $('myLabel').innerHTML = `<span class="pname">${player.name}</span>${player.id === state.banker_id ? '<span class="tag banker-tag">庄家</span>' : ''}<span class="pcount">${player.hand_count}张</span>${playedCardHtml(player, plays, 'bottom')}`;
}
function renderBackCards(count) {
  return Array.from({ length: Math.min(count, 16) }, () => '<span class="card-back"></span>').join('');
}

function renderHand() {
  const legal = state.legal_actions || { type: 'wait' };
  selected = new Set([...selected].filter(card => (state.hand || []).includes(card)));
  if (legal.type === 'play' && selected.size > 1) selected = new Set([[...selected][0]]);
  $('hand').innerHTML = (state.hand || []).map(c => cardHtml(c, selected.has(c))).join('');
  document.querySelectorAll('.card[data-card]').forEach(el => {
    el.onclick = () => {
      const card = el.dataset.card;
      if (selected.has(card)) {
        selected.delete(card);
      } else if (legal.type === 'play') {
        selected.clear();
        selected.add(card);
      } else {
        selected.add(card);
      }
      renderHand();
      log(`选择手牌：${[...selected].join(',') || '无'}`);
    };
  });
}
function renderActions() {
  const legal = state.legal_actions || { type: 'wait' };
  const box = $('actions');
  if (legal.type === 'bid') {
    $('actionLabel').textContent = '轮到你叫牌';
    box.innerHTML = legal.bids.map(b => `<button class="btn btn-primary" data-bid="${b}">叫 ${b}</button>`).join('') + '<button class="btn btn-pass" data-pass="1">放弃</button>';
    box.querySelectorAll('[data-bid]').forEach(btn => btn.onclick = () => send({ type: 'bid', bid: Number(btn.dataset.bid) }));
    box.querySelector('[data-pass]').onclick = () => send({ type: 'bid', bid: null });
  } else if (legal.type === 'bury') {
    $('actionLabel').textContent = '选择 6 张手牌扣底';
    box.innerHTML = '<button id="buryBtn" class="btn btn-primary">扣底</button>';
    $('buryBtn').onclick = () => send({ type: 'bury', cards: [...selected].slice(0, 6) });
  } else if (legal.type === 'trump') {
    $('actionLabel').textContent = '请选择主牌花色';
    box.innerHTML = '<select id="trumpSelect"><option value="S">黑桃</option><option value="H">红桃</option><option value="C">梅花</option><option value="D">方块</option></select><button id="trumpBtn" class="btn btn-primary">定主</button>';
    $('trumpBtn').onclick = () => send({ type: 'trump', suit: $('trumpSelect').value });
  } else if (legal.type === 'play') {
    $('actionLabel').textContent = '轮到你出牌';
    box.innerHTML = '<button id="playBtn" class="btn btn-play">出牌</button>';
    $('playBtn').onclick = () => {
      const card = [...selected].find(c => legal.cards.includes(c));
      if (!card) return log(`请选择合法牌，可出：${legal.cards.join(',')}`, true);
      send({ type: 'play', card });
    };
  } else {
    $('actionLabel').textContent = state.phase === 'finished' ? '本局结束' : `等待 ${nameOf(state.turn_player_id) || '其他玩家'} 操作`;
    box.innerHTML = '';
  }
}

function renderPlayedCards() {
  const plays = visibleTrickPlays();
  const box = $('playedCards');
  if (!plays.length) {
    box.innerHTML = '<span class="muted">暂无出牌</span>';
    return;
  }
  const source = state.current_trick?.plays?.length ? '当前轮' : '上一轮';
  box.innerHTML = `<span class="muted">${source}出牌已显示在各玩家旁边</span>`;
}
function renderResult() {
  if (!state.result || !Object.keys(state.result).length) return;
  $('resultBody').textContent = JSON.stringify(state.result, null, 2);
  $('resultDialog').classList.remove('hidden');
}

function cardHtml(code, isSelected = false) {
  const info = parseCard(code);
  return `<div class="card ${info.red ? 'red' : 'black'} ${info.jokerClass} ${isSelected ? 'selected' : ''}" data-card="${code}" title="${info.label}"><span class="card-suit">${info.suit}</span><span class="card-rank">${info.rank}</span></div>`;
}

function parseCard(code) {
  if (code === 'BJ') return { rank: '大王', suit: '★', label: '大王', red: true, jokerClass: 'joker-r' };
  if (code === 'XJ' || code === 'SJOKER') return { rank: '小王', suit: '☆', label: '小王', red: false, jokerClass: 'joker-b' };
  const suitCode = code[0];
  const rank = code.slice(1);
  const suitMap = { S: '♠', H: '♥', C: '♣', D: '♦' };
  const nameMap = { S: '黑桃', H: '红桃', C: '梅花', D: '方块' };
  return { rank, suit: suitMap[suitCode], label: `${nameMap[suitCode]}${rank}`, red: suitCode === 'H' || suitCode === 'D', jokerClass: '' };
}

function showScreen(id) {
  ['lobby', 'waiting', 'gameScreen'].forEach(x => $(x).classList.toggle('active', x === id));
}
function showMsg(id, msg) { $(id).textContent = msg; }
function nameOf(id) { const p = state && state.players.find(x => x.id === id); return p ? p.name : ''; }
function phaseName(p) { return ({ waiting: '等待', bidding: '叫牌', kitty: '扣底', trump: '定主', playing: '出牌', finished: '结束' })[p] || p; }
function suitName(s) { return ({ S: '黑桃', H: '红桃', C: '梅花', D: '方块' })[s] || ''; }
function log(msg, danger = false) {
  const line = document.createElement('div');
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  if (danger) line.className = 'err';
  $('log').prepend(line);
  console[danger ? 'error' : 'log'](msg);
}

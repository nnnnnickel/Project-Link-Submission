const form = document.querySelector("#reading-form");
const questionInput = document.querySelector("#question");
const cardsInput = document.querySelector("#cards");
const submitButton = document.querySelector("#submit-button");
const exitButton = document.querySelector("#exit-button");
const newReadingButton = document.querySelector("#new-reading");
const followupButton = document.querySelector("#followup");
const chatLog = document.querySelector("#chat-log");
const statusText = document.querySelector("#status-text");

const SESSION_ID = "default";

const drawSourceButton = document.querySelector("#draw-source");
const manualSourceButton = document.querySelector("#manual-source");
const drawPanel = document.querySelector("#draw-panel");
const drawButton = document.querySelector("#draw-button");
const drawnCardsEl = document.querySelector("#drawn-cards");
const cardsSummaryEl = document.querySelector("#cards-summary");

let followupMode = false;
let closed = false;
let cardSource = "draw";
let currentDrawn = [];
const DRAW_HINT = '<p class="draw-hint">点击「抽一张牌」，可连续抽取，抽几张解读几张</p>';

/* ---------------- Tarot deck + draw feature ---------------- */

const MAJOR_ARCANA = [
  ["The Fool", "愚者"],
  ["The Magician", "魔术师"],
  ["The High Priestess", "女祭司"],
  ["The Empress", "女皇"],
  ["The Emperor", "皇帝"],
  ["The Hierophant", "教皇"],
  ["The Lovers", "恋人"],
  ["The Chariot", "战车"],
  ["Strength", "力量"],
  ["The Hermit", "隐士"],
  ["Wheel of Fortune", "命运之轮"],
  ["Justice", "正义"],
  ["The Hanged Man", "倒吊人"],
  ["Death", "死神"],
  ["Temperance", "节制"],
  ["The Devil", "恶魔"],
  ["The Tower", "塔"],
  ["The Star", "星星"],
  ["The Moon", "月亮"],
  ["The Sun", "太阳"],
  ["Judgement", "审判"],
  ["The World", "世界"],
];

const SUITS = [
  ["Wands", "权杖"],
  ["Cups", "圣杯"],
  ["Swords", "宝剑"],
  ["Pentacles", "星币"],
];

const RANKS = [
  ["Ace", "一"],
  ["Two", "二"],
  ["Three", "三"],
  ["Four", "四"],
  ["Five", "五"],
  ["Six", "六"],
  ["Seven", "七"],
  ["Eight", "八"],
  ["Nine", "九"],
  ["Ten", "十"],
  ["Page", "侍从"],
  ["Knight", "骑士"],
  ["Queen", "皇后"],
  ["King", "国王"],
];

function buildDeck() {
  const deck = MAJOR_ARCANA.map(([en, cn]) => ({ en, cn }));
  for (const [suitEn, suitCn] of SUITS) {
    for (const [rankEn, rankCn] of RANKS) {
      deck.push({ en: `${rankEn} of ${suitEn}`, cn: `${suitCn}${rankCn}` });
    }
  }
  return deck;
}

const TAROT_DECK = buildDeck();

const CARD_CN_LOOKUP = new Map(TAROT_DECK.map((card) => [card.en.toLowerCase(), card.cn]));

function createCardElement(card, revealDelay) {
  const wrap = document.createElement("div");
  wrap.className = "tarot-card" + (card.reversed ? " reversed" : "");
  wrap.innerHTML = `
    <div class="tarot-card-inner">
      <div class="tarot-card-face tarot-card-back"></div>
      <div class="tarot-card-face tarot-card-front">
        <span>${card.cn}<span class="orientation">${card.reversed ? "逆位" : "正位"}</span></span>
      </div>
    </div>
  `;
  window.setTimeout(() => wrap.classList.add("revealed"), revealDelay);
  return wrap;
}

function appendDrawnCard(card, index) {
  const hint = drawnCardsEl.querySelector(".draw-hint");
  if (hint) {
    hint.remove();
  }
  drawnCardsEl.appendChild(createCardElement(card, 60 + index * 40));
}

function parseCardsInput(input) {
  if (!input) {
    return [];
  }
  const parts = Array.isArray(input) ? input : String(input).split(",");
  return parts
    .map((part) => String(part).trim())
    .filter(Boolean)
    .map((part) => {
      const reversedMatch = /\s*(\(reversed\)|reversed|\(逆位\)|逆位)\s*$/i.exec(part);
      const reversed = Boolean(reversedMatch);
      const base = reversed ? part.slice(0, reversedMatch.index).trim() : part;
      const cn = CARD_CN_LOOKUP.get(base.toLowerCase()) || base;
      return { en: base, cn, reversed };
    });
}

function renderTurnCards(container, cardsData) {
  const cards = parseCardsInput(cardsData);
  if (!cards.length) {
    return false;
  }
  cards.forEach((card, index) => {
    const el = createCardElement(card, 60 + index * 90);
    el.classList.add("tarot-card-sm");
    container.appendChild(el);
  });
  return true;
}

function updateCardsSummary(drawn) {
  if (!drawn.length) {
    cardsSummaryEl.hidden = true;
    cardsSummaryEl.textContent = "";
    return;
  }
  cardsSummaryEl.hidden = false;
  cardsSummaryEl.textContent = drawn
    .map((card) => `${card.en}${card.reversed ? " (逆位)" : ""}`)
    .join(" · ");
}

function drawOneCard() {
  const remaining = TAROT_DECK.filter((deckCard) => !currentDrawn.some((drawn) => drawn.en === deckCard.en));
  if (!remaining.length) {
    cardsSummaryEl.hidden = false;
    cardsSummaryEl.textContent = "78 张牌已全部抽出，点击「新解读」开始新的一组。";
    return;
  }
  const idx = Math.floor(Math.random() * remaining.length);
  const card = { ...remaining[idx], reversed: Math.random() < 0.3 };
  currentDrawn.push(card);
  appendDrawnCard(card, currentDrawn.length - 1);
  updateCardsSummary(currentDrawn);
  cardsInput.value = currentDrawn.map((c) => `${c.en}${c.reversed ? " reversed" : ""}`).join(", ");
}

function resetDraw() {
  currentDrawn = [];
  drawnCardsEl.innerHTML = DRAW_HINT;
  cardsSummaryEl.hidden = true;
  cardsSummaryEl.textContent = "";
  if (cardSource === "draw") {
    cardsInput.value = "";
  }
}

function setCardSource(nextSource) {
  cardSource = nextSource;
  const isDraw = cardSource === "draw";
  drawSourceButton.classList.toggle("active", isDraw);
  manualSourceButton.classList.toggle("active", !isDraw);
  drawPanel.classList.toggle("hidden", !isDraw);
  cardsInput.classList.toggle("hidden", isDraw);
  if (isDraw) {
    cardsInput.value = cardsSummaryEl.hidden ? "" : cardsInput.value;
  }
}

drawButton.addEventListener("click", drawOneCard);
drawSourceButton.addEventListener("click", () => setCardSource("draw"));
manualSourceButton.addEventListener("click", () => setCardSource("manual"));

setCardSource("draw");

function setMode(nextFollowupMode) {
  followupMode = nextFollowupMode;
  newReadingButton.classList.toggle("active", !followupMode);
  followupButton.classList.toggle("active", followupMode);
  cardsInput.placeholder = followupMode
    ? "追问可留空；如有补牌，可输入补充牌面"
    : "例如：The Moon, The Empress reversed, Wheel of Fortune";
  statusText.textContent = followupMode ? "追问模式" : "等待提问";
}

function ensureChatStarted() {
  const empty = chatLog.querySelector(".empty-state");
  if (empty) {
    empty.remove();
  }
}

function appendTurn(question, answer, meta, cardsData) {
  ensureChatStarted();
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="bubble question"></div>
    <div class="turn-cards"></div>
    <div class="bubble answer"></div>
    <div class="meta"></div>
  `;
  turn.querySelector(".question").textContent = question;
  const turnCardsEl = turn.querySelector(".turn-cards");
  const hasCards = renderTurnCards(turnCardsEl, cardsData);
  if (!hasCards) {
    turnCardsEl.remove();
  }
  turn.querySelector(".answer").textContent = answer || "没有返回解牌结果。";
  turn.querySelector(".meta").textContent = meta;
  chatLog.appendChild(turn);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendNotice(message) {
  ensureChatStarted();
  const notice = document.createElement("div");
  notice.className = "notice";
  notice.textContent = message;
  chatLog.appendChild(notice);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendLoadingIndicator() {
  ensureChatStarted();
  const loading = document.createElement("div");
  loading.className = "turn loading-turn";
  loading.id = "loading-indicator";
  loading.innerHTML = `
    <div class="loading-bubble">
      <span class="loading-dot"></span><span class="loading-dot"></span><span class="loading-dot"></span>
      <span class="loading-text">正在解读中…</span>
    </div>
  `;
  chatLog.appendChild(loading);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function removeLoadingIndicator() {
  const loading = document.querySelector("#loading-indicator");
  if (loading) {
    loading.remove();
  }
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy || closed;
  exitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "解读中..." : "发送";
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (closed) {
    appendNotice("当前会话已退出，请刷新页面开始新的会话。");
    return;
  }

  const question = questionInput.value.trim();
  const cards = cardsInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }

  setBusy(true);
  statusText.textContent = "正在调用 Tarot Agent";
  appendLoadingIndicator();
  try {
    const data = await postJson("/api/reading", {
      question,
      cards,
      session_id: SESSION_ID,
      followup: followupMode,
      reset_session: !followupMode,
      use_skills: true,
    });
    removeLoadingIndicator();
    const cardText = Array.isArray(data.cards) && data.cards.length ? `牌：${data.cards.join(" / ")}` : "沿用上一轮牌面";
    appendTurn(data.question, data.answer, `${cardText} · ${data.topic || "general"} · ${data.turn_mode || "reading"}`, data.cards);
    questionInput.value = "";
    cardsInput.value = "";
    resetDraw();
    setMode(true);
    statusText.textContent = "可继续追问";
  } catch (error) {
    removeLoadingIndicator();
    appendNotice(error.message);
    statusText.textContent = "请求未完成";
  } finally {
    setBusy(false);
  }
});

exitButton.addEventListener("click", async () => {
  setBusy(true);
  try {
    await postJson("/api/exit", { session_id: SESSION_ID });
    closed = true;
    appendNotice("会话已退出，记忆已清空。页面将尝试自动关闭；如果没有自动关闭，请手动关闭此页面。");
    statusText.textContent = "已退出";
    questionInput.disabled = true;
    cardsInput.disabled = true;
    submitButton.disabled = true;
    drawButton.disabled = true;
    drawSourceButton.disabled = true;
    manualSourceButton.disabled = true;
    window.setTimeout(() => {
      window.close();
      window.setTimeout(() => {
        statusText.textContent = "已退出（请手动关闭此页面）";
      }, 400);
    }, 700);
  } catch (error) {
    appendNotice(error.message);
  } finally {
    exitButton.disabled = true;
  }
});

newReadingButton.addEventListener("click", () => {
  resetDraw();
  setMode(false);
});
followupButton.addEventListener("click", () => setMode(true));

setMode(false);

/* ---------------- Intro overlay ---------------- */

const INTRO_STORAGE_KEY = "agentic-tarot-intro-seen";
const introOverlay = document.querySelector("#intro-overlay");
const helpButton = document.querySelector("#help-button");

function openIntro() {
  introOverlay.classList.remove("hidden");
}

function closeIntro() {
  introOverlay.classList.add("hidden");
  try {
    window.localStorage.setItem(INTRO_STORAGE_KEY, "1");
  } catch (error) {
    /* localStorage unavailable, ignore */
  }
}

introOverlay.querySelectorAll("[data-close-intro]").forEach((el) => {
  el.addEventListener("click", closeIntro);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !introOverlay.classList.contains("hidden")) {
    closeIntro();
  }
});

helpButton.addEventListener("click", openIntro);

let hasSeenIntro = false;
try {
  hasSeenIntro = window.localStorage.getItem(INTRO_STORAGE_KEY) === "1";
} catch (error) {
  hasSeenIntro = false;
}

if (hasSeenIntro) {
  introOverlay.classList.add("hidden");
} else {
  openIntro();
}

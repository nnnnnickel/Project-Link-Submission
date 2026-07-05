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
const DRAW_HINT = '<p class="draw-hint">Click "Draw a Card" as many times as you like — read as many as you draw</p>';

/* ---------------- Tarot deck + draw feature ---------------- */

const MAJOR_ARCANA = [
  "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
  "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
  "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
  "The Devil", "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World",
];

const SUITS = ["Wands", "Cups", "Swords", "Pentacles"];

const RANKS = [
  "Ace", "Two", "Three", "Four", "Five", "Six", "Seven",
  "Eight", "Nine", "Ten", "Page", "Knight", "Queen", "King",
];

function buildDeck() {
  const deck = MAJOR_ARCANA.map((name) => ({ en: name }));
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      deck.push({ en: `${rank} of ${suit}` });
    }
  }
  return deck;
}

const TAROT_DECK = buildDeck();

function createCardElement(card, revealDelay) {
  const wrap = document.createElement("div");
  wrap.className = "tarot-card" + (card.reversed ? " reversed" : "");
  wrap.innerHTML = `
    <div class="tarot-card-inner">
      <div class="tarot-card-face tarot-card-back"></div>
      <div class="tarot-card-face tarot-card-front">
        <span>${card.en}<span class="orientation">${card.reversed ? "Reversed" : "Upright"}</span></span>
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
      const reversedMatch = /\s*(\(reversed\)|reversed)\s*$/i.exec(part);
      const reversed = Boolean(reversedMatch);
      const base = reversed ? part.slice(0, reversedMatch.index).trim() : part;
      return { en: base, reversed };
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
    .map((card) => `${card.en}${card.reversed ? " (reversed)" : ""}`)
    .join(" · ");
}

function drawOneCard() {
  const remaining = TAROT_DECK.filter((deckCard) => !currentDrawn.some((drawn) => drawn.en === deckCard.en));
  if (!remaining.length) {
    cardsSummaryEl.hidden = false;
    cardsSummaryEl.textContent = "All 78 cards have been drawn. Click \"New Reading\" to start a new spread.";
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
    ? "Optional for follow-ups. Add extra cards only if you drew more."
    : "Example: The Moon, The Empress reversed, Wheel of Fortune";
  statusText.textContent = followupMode ? "Follow-up mode" : "Ready for a question";
}

function renderEmptyState(message = "Enter a question and cards for the first reading. Follow-ups can use only a question.") {
  chatLog.innerHTML = `
    <article class="empty-state">
      <h3>Begin a reading</h3>
      <p></p>
    </article>
  `;
  chatLog.querySelector("p").textContent = message;
}

function ensureChatStarted() {
  const empty = chatLog.querySelector(".empty-state");
  if (empty) {
    empty.remove();
  }
}

function scrollToLatest() {
  requestAnimationFrame(() => {
    chatLog.scrollTop = chatLog.scrollHeight;
  });
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
  turn.querySelector(".answer").textContent = answer || "No reading result was returned.";
  turn.querySelector(".meta").textContent = meta;
  chatLog.appendChild(turn);
  scrollToLatest();
}

function appendNotice(message) {
  ensureChatStarted();
  const notice = document.createElement("div");
  notice.className = "notice";
  notice.textContent = message;
  chatLog.appendChild(notice);
  scrollToLatest();
}

function appendLoadingIndicator() {
  ensureChatStarted();
  const loading = document.createElement("div");
  loading.className = "turn loading-turn";
  loading.id = "loading-indicator";
  loading.innerHTML = `
    <div class="loading-bubble">
      <span class="loading-dot"></span><span class="loading-dot"></span><span class="loading-dot"></span>
      <span class="loading-text">Reading the cards…</span>
    </div>
  `;
  chatLog.appendChild(loading);
  scrollToLatest();
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
  submitButton.textContent = isBusy ? "Reading..." : "Send";
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (closed) {
    appendNotice("This session is closed. Start a new reading to continue.");
    return;
  }

  const question = questionInput.value.trim();
  const cards = cardsInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }

  setBusy(true);
  statusText.textContent = "Contacting Tarot Agent";
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
    const cardText = Array.isArray(data.cards) && data.cards.length ? `Cards: ${data.cards.join(" / ")}` : "Using the previous spread";
    appendTurn(data.question, data.answer, `${cardText} - ${data.topic || "general"} - ${data.turn_mode || "reading"}`, data.cards);
    questionInput.value = "";
    cardsInput.value = "";
    resetDraw();
    setMode(true);
    statusText.textContent = "Ready for a follow-up";
  } catch (error) {
    removeLoadingIndicator();
    appendNotice(error.message);
    statusText.textContent = "Request incomplete";
  } finally {
    setBusy(false);
  }
});

exitButton.addEventListener("click", async () => {
  setBusy(true);
  try {
    await postJson("/api/exit", { session_id: SESSION_ID });
    closed = true;
    renderEmptyState("Session closed. The current question-and-answer thread has been cleared. The page will try to close automatically; if it doesn't, please close this page manually.");
    statusText.textContent = "Session closed";
    questionInput.value = "";
    cardsInput.value = "";
    questionInput.disabled = true;
    cardsInput.disabled = true;
    submitButton.disabled = true;
    drawButton.disabled = true;
    drawSourceButton.disabled = true;
    manualSourceButton.disabled = true;
    window.setTimeout(() => {
      window.close();
      window.setTimeout(() => {
        statusText.textContent = "Session closed (please close this page manually)";
      }, 400);
    }, 700);
  } catch (error) {
    appendNotice(error.message);
  } finally {
    exitButton.disabled = true;
  }
});

newReadingButton.addEventListener("click", () => {
  if (closed) {
    closed = false;
    questionInput.disabled = false;
    cardsInput.disabled = false;
    exitButton.disabled = false;
    submitButton.disabled = false;
    drawButton.disabled = false;
    drawSourceButton.disabled = false;
    manualSourceButton.disabled = false;
    renderEmptyState();
  }
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

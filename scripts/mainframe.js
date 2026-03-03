/**
 * The Mainframe - Frontend MVP
 * - Dynamic block grid of cyber intel items
 * - Client-side filters (source, severity, search)
 * - New item indicator with subtle glitch for high-priority entries
 * - Opens clean reader page in a new tab with full content
 *
 * Notes:
 * - This MVP uses mocked items. Backend connectors can replace getItems()
 *   to fetch live feeds (RSS/REST), normalize, enrich, and cache.
 */

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

/* Data model */
async function getItems() {
  // Prefer backend API; fallback to a tiny local dataset if not available.
  try {
    const res = await fetch("/api/items?limit=120", { headers: { "accept": "application/json" } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return Array.isArray(data.items) ? data.items : [];
  } catch {
    const now = Date.now();
    return [
      {
        id: "demo:cisa",
        source: "CISA",
        title: "Backend offline — showing demo content",
        blurb: "Start the server (npm install && npm run dev) to pull live CISA/NVD/THN/arXiv feeds.",
        url: "reader.html?id=demo:cisa",
        published_at: new Date(now - 2 * 60 * 1000).toISOString(),
        severity: "info",
        badges: ["Demo"],
      },
    ];
  }
}

/* State */
const State = {
  items: [],
  filters: {
    sources: new Set(["all"]),
    severity: "any",
    search: "",
    chip: null, // kev, cve-today, ransomware, cloud
    sort: "latest"
  },
  lastRenderAt: Date.now(),
  unseen: 0
};

/* Utils */
function timeSince(iso) {
  const d = new Date(iso);
  const ms = Date.now() - d.getTime();
  const min = Math.round(ms / 60000);
  if (min < 1) return "just now";
  if (min < 60) return `${min}m ago`;
  const h = Math.round(min / 60);
  return `${h}h ago`;
}
function matchesSearch(item, q) {
  if (!q) return true;
  const s = q.toLowerCase();
  return (
    item.title.toLowerCase().includes(s) ||
    (item.blurb || "").toLowerCase().includes(s) ||
    (item.source || "").toLowerCase().includes(s)
  );
}
function chipPredicate(item, chip) {
  if (!chip) return true;
  switch (chip) {
    case "kev":
      return (item.badges || []).some(b => b.toLowerCase().includes("kev"));
    case "cve-today": {
      const isToday = new Date(item.published_at).toDateString() === new Date().toDateString();
      const hasCve = (item.meta?.cves || []).length > 0 || (item.badges || []).some(b => b.toLowerCase() === "cve");
      return isToday && hasCve;
    }
    case "ransomware":
      return (item.badges || []).some(b => b.toLowerCase().includes("ransom"));
    case "cloud":
      return (item.badges || []).some(b => b.toLowerCase().includes("cloud")) ||
             item.title.toLowerCase().includes("azure") ||
             item.title.toLowerCase().includes("aws") ||
             item.title.toLowerCase().includes("gcp");
    default:
      return true;
  }
}
function sortItems(items, mode) {
  if (mode === "priority") {
    const weight = { critical: 4, high: 3, elevated: 2, info: 1 };
    return items.slice().sort((a, b) => {
      const pa = weight[a.severity] || 0;
      const pb = weight[b.severity] || 0;
      if (pb !== pa) return pb - pa;
      return new Date(b.published_at) - new Date(a.published_at);
    });
  }
  // latest
  return items.slice().sort((a, b) => new Date(b.published_at) - new Date(a.published_at));
}

/* Rendering */
function render() {
  const grid = $("#grid");
  const { sources, severity, search, chip, sort } = State.filters;

  let items = State.items.filter(it => {
    const sourceOk = sources.has("all") || sources.has(it.source);
    const severityOk = severity === "any" || it.severity === severity;
    return sourceOk && severityOk && matchesSearch(it, search) && chipPredicate(it, chip);
  });

  items = sortItems(items, sort);

  $("#stat-total").textContent = String(items.length);
  $("#stat-new").textContent = String(State.unseen);

  grid.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "card";
    card.dataset.id = item.id;
    card.innerHTML = `
      <div class="source">via ${item.source}</div>
      <div class="headline">${item.title}</div>
      <div class="blurb">${item.blurb || ""}</div>
      <div class="meta">
        <span>${timeSince(item.published_at)}</span>
        ${item.severity ? `<span class="badge ${item.severity}">${item.severity.toUpperCase()}</span>` : ""}
        ${(item.badges || []).slice(0,2).map(b => `<span class="badge">${b}</span>`).join("")}
      </div>
    `;
    if (item.severity === "critical" || item.severity === "high") {
      const g = document.createElement("div");
      g.className = "badge glitch";
      g.textContent = item.severity === "critical" ? "NEW • PRIORITY" : "NEW";
      card.appendChild(g);
      setTimeout(() => g.remove(), 1200);
    }
    card.addEventListener("click", () => openReader(item));
    grid.appendChild(card);
  }
}

function openReader(item) {
  // Build a robust absolute URL for reader.html relative to the current page
  const readerUrl = new URL("reader.html", window.location.href);
  readerUrl.searchParams.set("id", item.id);
  const url = item.url && item.url.startsWith("http")
    ? item.url
    : readerUrl.toString();
  window.open(url, "_blank", "noopener,noreferrer");
}

/* Controls */
function initControls() {
  const sourceSel = $("#filter-source");
  const severitySel = $("#filter-severity");
  const sortSel = $("#sort-order");
  const search = $("#search");
  const chips = $$(".chip");

  sourceSel.addEventListener("change", () => {
    const selected = new Set(Array.from(sourceSel.selectedOptions).map(o => o.value));
    State.filters.sources = selected.size ? selected : new Set(["all"]);
    render();
  });
  severitySel.addEventListener("change", () => {
    State.filters.severity = severitySel.value;
    render();
  });
  sortSel.addEventListener("change", () => {
    State.filters.sort = sortSel.value;
    render();
  });
  search.addEventListener("input", () => {
    State.filters.search = search.value.trim();
    render();
  });
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const key = chip.dataset.chip;
      State.filters.chip = State.filters.chip === key ? null : key;
      chips.forEach(c => c.classList.toggle("active", c.dataset.chip === State.filters.chip));
      render();
    });
  });
}

/* Polling simulation for new items */
function initPolling() {
  // Simulate new incoming high-priority item every ~40s
  setInterval(() => {
    const t = Date.now();
    const newItem = {
      id: `auto-${t}`,
      source: Math.random() > 0.5 ? "NVD" : "CISA",
      title: Math.random() > 0.5 ? "New KEV entry for actively exploited CVE" : "NVD publishes critical CVE update",
      blurb: "Auto-ingested mock item. Replace with live feed event in production.",
      url: `reader.html?id=auto-${t}`,
      published_at: new Date().toISOString(),
      severity: Math.random() > 0.5 ? "critical" : "high",
      badges: Math.random() > 0.5 ? ["KEV"] : ["CVE"]
    };
    State.items.unshift(newItem);
    State.unseen += 1;
    $("#new-indicator").classList.remove("hidden");
  }, 40000);

  $("#new-indicator").addEventListener("click", () => {
    State.unseen = 0;
    $("#new-indicator").classList.add("hidden");
    render();
  });
}

/* Boot */
document.addEventListener("DOMContentLoaded", async () => {
  State.items = await getItems();
  initControls();
  initPolling();
  render();
});
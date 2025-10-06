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
function getItems() {
  const now = Date.now();
  // Mocked items for demo; replace with backend data
  return [
    {
      id: "cisa-kev-2025-0001",
      source: "CISA",
      title: "CISA adds CVE-2025-0001 to Known Exploited Vulnerabilities",
      blurb: "Active exploitation observed in the wild. Federal agencies must remediate by the specified due date.",
      url: "reader.html?id=cisa-kev-2025-0001",
      published_at: new Date(now - 5 * 60 * 1000).toISOString(),
      severity: "critical",
      badges: ["KEV", "Exploit In The Wild"],
      meta: { cves: ["CVE-2025-0001"], epss: 0.92 }
    },
    {
      id: "nvd-ms-azure-cves",
      source: "NVD",
      title: "Multiple Azure services impacted by elevation of privilege flaws",
      blurb: "NVD published CVEs affecting Azure compute with patches available. Exploitation less likely per MSRC.",
      url: "reader.html?id=nvd-ms-azure-cves",
      published_at: new Date(now - 25 * 60 * 1000).toISOString(),
      severity: "high",
      badges: ["Cloud", "CVE"],
      meta: { cves: ["CVE-2025-1002", "CVE-2025-1003"] }
    },
    {
      id: "arxiv-crypto-scheme",
      source: "arXiv",
      title: "On the Security of Post-Quantum Lattice Schemes under Chosen-Ciphertext",
      blurb: "New preprint analyzes side-channel resilience and proposes mitigations for key encapsulation mechanisms.",
      url: "reader.html?id=arxiv-crypto-scheme",
      published_at: new Date(now - 65 * 60 * 1000).toISOString(),
      severity: "info",
      badges: ["Research"]
    },
    {
      id: "thn-ransomware-campaign",
      source: "The Hacker News",
      title: "New Ransomware Campaign Targets ESXi via Compromised IT MSP Accounts",
      blurb: "Attackers pivot through MSP RMM tools; indicators include IPs and hashes now circulating on OSINT lists.",
      url: "reader.html?id=thn-ransomware-campaign",
      published_at: new Date(now - 8 * 60 * 1000).toISOString(),
      severity: "elevated",
      badges: ["Ransomware"]
    },
    {
      id: "microsoft-threat-intel",
      source: "Microsoft Security",
      title: "Storm-1234 weaponizes open-source module for kernel-level persistence",
      blurb: "Observed in targeted attacks against telecom; mitigations and hunting queries provided.",
      url: "reader.html?id=microsoft-threat-intel",
      published_at: new Date(now - 120 * 60 * 1000).toISOString(),
      severity: "high",
      badges: ["Threat Intel"]
    },
    {
      id: "mandiant-supply-chain",
      source: "Mandiant",
      title: "Supply-chain compromise delivers trojanized desktop update client",
      blurb: "Signed binary abused to deploy loader; C2 over DNS with domain fronting. IOCs and YARA published.",
      url: "reader.html?id=mandiant-supply-chain",
      published_at: new Date(now - 15 * 60 * 1000).toISOString(),
      severity: "critical",
      badges: ["Supply Chain", "YARA"]
    }
  ];
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
document.addEventListener("DOMContentLoaded", () => {
  State.items = getItems();
  initControls();
  initPolling();
  render();
});
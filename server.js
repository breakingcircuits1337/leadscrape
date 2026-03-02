import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { XMLParser } from "fast-xml-parser";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;

app.disable("x-powered-by");

app.use(express.static(__dirname, {
  extensions: ["html"],
  setHeaders(res) {
    res.setHeader("Cache-Control", "no-store");
  },
}));

const xmlParser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "@_",
  trimValues: true,
  parseTagValue: true,
});

function asArray(v) {
  if (!v) return [];
  return Array.isArray(v) ? v : [v];
}

function stripHtml(html) {
  return String(html || "")
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function pickBlurb(text, maxLen = 180) {
  const t = String(text || "").replace(/\s+/g, " ").trim();
  if (t.length <= maxLen) return t;
  return `${t.slice(0, maxLen - 1)}…`;
}

function severityFromSignals({ kev = false, cvss = null, epss = null } = {}) {
  if (kev) return "critical";
  const c = typeof cvss === "number" ? cvss : null;
  if (c !== null) {
    if (c >= 9.0) return "critical";
    if (c >= 7.0) return "high";
    if (c >= 4.0) return "elevated";
    return "info";
  }
  const e = typeof epss === "number" ? epss : null;
  if (e !== null) {
    if (e >= 0.9) return "high";
    if (e >= 0.5) return "elevated";
  }
  return "info";
}

async function fetchJson(url) {
  const res = await fetch(url, {
    headers: {
      "user-agent": "TheMainframe/0.1 (+https://example.invalid)",
      "accept": "application/json,text/plain,*/*",
    },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

async function fetchText(url) {
  const res = await fetch(url, {
    headers: {
      "user-agent": "TheMainframe/0.1 (+https://example.invalid)",
      "accept": "application/rss+xml,application/atom+xml,application/xml,text/xml,text/html,*/*",
    },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.text();
}

function nowIso() {
  return new Date().toISOString();
}

function normalizeItem(item) {
  return {
    id: String(item.id),
    source: String(item.source || "Unknown"),
    title: String(item.title || ""),
    blurb: String(item.blurb || ""),
    published_at: item.published_at || nowIso(),
    url: item.url || "",
    severity: item.severity || "info",
    badges: Array.isArray(item.badges) ? item.badges : [],
    meta: item.meta || {},
    content: item.content || "",
  };
}

async function ingestCisaKev(limit = 30) {
  // Official feed used widely by tooling
  const url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json";
  const data = await fetchJson(url);
  const vulnerabilities = asArray(data?.vulnerabilities).slice(0, limit);

  return vulnerabilities.map((v) => {
    const cve = v.cveID || v.cveId || "";
    const vendor = v.vendorProject || "";
    const product = v.product || "";
    const name = v.vulnerabilityName || "";
    const due = v.dueDate || "";
    const desc = v.shortDescription || "";

    const titleParts = ["CISA KEV", cve, vendor, product].filter(Boolean);
    const title = `${titleParts.join(" • ")}${name ? ` — ${name}` : ""}`;

    const blurb = pickBlurb(desc || `Known exploited vulnerability. Due: ${due}`);

    const badges = ["KEV"].concat(v.knownRansomwareCampaignUse === "Known" ? ["Ransomware"] : []);

    const content = [
      desc ? `Summary: ${desc}` : "",
      v.requiredAction ? `Required action: ${v.requiredAction}` : "",
      due ? `Due date: ${due}` : "",
      v.notes ? `Notes: ${v.notes}` : "",
      v.cwes ? `CWEs: ${v.cwes}` : "",
    ].filter(Boolean).join("\n\n");

    return normalizeItem({
      id: `cisa-kev:${cve || v.dateAdded || Math.random().toString(36).slice(2)}`,
      source: "CISA",
      title,
      blurb,
      published_at: v.dateAdded ? new Date(v.dateAdded).toISOString() : nowIso(),
      url: "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
      severity: severityFromSignals({ kev: true }),
      badges,
      meta: {
        cves: cve ? [cve] : [],
        dueDate: due,
        vendorProject: vendor,
        product,
      },
      content,
    });
  });
}

async function ingestRss({ source, feedUrl, limit = 25, badge = null }) {
  const xml = await fetchText(feedUrl);
  const parsed = xmlParser.parse(xml);

  // RSS2
  const rssItems = asArray(parsed?.rss?.channel?.item);
  if (rssItems.length) {
    return rssItems.slice(0, limit).map((it) => {
      const link = it.link || it.guid || "";
      const pub = it.pubDate ? new Date(it.pubDate).toISOString() : nowIso();
      const desc = stripHtml(it.description || "");
      return normalizeItem({
        id: `${source}:${String(link || it.title || pub)}`,
        source,
        title: stripHtml(it.title || ""),
        blurb: pickBlurb(desc),
        published_at: pub,
        url: link,
        severity: "info",
        badges: badge ? [badge] : [],
        content: desc,
      });
    });
  }

  // Atom
  const entries = asArray(parsed?.feed?.entry);
  return entries.slice(0, limit).map((e) => {
    const title = stripHtml(e.title?.["#text"] ?? e.title ?? "");
    const link = asArray(e.link).find(l => l?.["@_"]?.rel !== "self")?.["@_"]?.href
      || asArray(e.link)[0]?.["@_"]?.href
      || "";
    const updated = e.updated ? new Date(e.updated).toISOString() : (e.published ? new Date(e.published).toISOString() : nowIso());
    const summary = stripHtml(e.summary?.["#text"] ?? e.summary ?? e.content?.["#text"] ?? e.content ?? "");

    return normalizeItem({
      id: `${source}:${String(link || title || updated)}`,
      source,
      title,
      blurb: pickBlurb(summary),
      published_at: updated,
      url: link,
      severity: "info",
      badges: badge ? [badge] : [],
      content: summary,
    });
  });
}

async function ingestNvdRecent(limit = 30) {
  const end = new Date();
  const start = new Date(end.getTime() - 24 * 60 * 60 * 1000);

  // NVD 2.0 API. Unauthenticated requests are rate-limited; keep this modest.
  const url = new URL("https://services.nvd.nist.gov/rest/json/cves/2.0");
  url.searchParams.set("pubStartDate", start.toISOString());
  url.searchParams.set("pubEndDate", end.toISOString());
  url.searchParams.set("resultsPerPage", String(Math.min(limit, 200)));

  const data = await fetchJson(url.toString());
  const vulns = asArray(data?.vulnerabilities).slice(0, limit);

  return vulns.map((row) => {
    const cve = row?.cve;
    const id = cve?.id || "";
    const desc = asArray(cve?.descriptions).find(d => d?.lang === "en")?.value
      || asArray(cve?.descriptions)[0]?.value
      || "";

    const metrics = cve?.metrics || {};
    const cvss31 = metrics?.cvssMetricV31?.[0]?.cvssData;
    const cvss30 = metrics?.cvssMetricV30?.[0]?.cvssData;
    const cvss = (cvss31?.baseScore ?? cvss30?.baseScore);

    const published = cve?.published ? new Date(cve.published).toISOString() : nowIso();

    return normalizeItem({
      id: `nvd:${id || Math.random().toString(36).slice(2)}`,
      source: "NVD",
      title: id ? `NVD: ${id}` : "NVD: CVE Update",
      blurb: pickBlurb(desc),
      published_at: published,
      url: id ? `https://nvd.nist.gov/vuln/detail/${id}` : "https://nvd.nist.gov/",
      severity: severityFromSignals({ cvss: typeof cvss === "number" ? cvss : null }),
      badges: ["CVE"],
      meta: { cves: id ? [id] : [], cvss },
      content: desc,
    });
  });
}

function dedupe(items) {
  const seen = new Set();
  const out = [];
  for (const it of items) {
    const key = it.url ? `url:${it.url}` : `id:${it.id}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(it);
  }
  return out;
}

const Cache = {
  items: [],
  byId: new Map(),
  lastRefresh: 0,
  refreshing: null,
  ttlMs: 5 * 60 * 1000,
};

async function refreshCache() {
  const tasks = [
    ingestCisaKev(35),
    ingestNvdRecent(35),
    ingestRss({ source: "The Hacker News", feedUrl: "https://thehackernews.com/feeds/posts/default?alt=rss", limit: 20, badge: "News" }),
    ingestRss({ source: "arXiv", feedUrl: "https://export.arxiv.org/api/query?search_query=cat:cs.CR&sortBy=submittedDate&sortOrder=descending&max_results=25", limit: 20, badge: "Research" }),
  ];

  const results = await Promise.allSettled(tasks);
  const collected = [];
  for (const r of results) {
    if (r.status === "fulfilled") collected.push(...r.value);
  }

  const merged = dedupe(collected).sort((a, b) => new Date(b.published_at) - new Date(a.published_at));
  Cache.items = merged;
  Cache.byId = new Map(merged.map((i) => [i.id, i]));
  Cache.lastRefresh = Date.now();
  return merged;
}

async function ensureCache() {
  const stale = Date.now() - Cache.lastRefresh > Cache.ttlMs;
  if (!stale && Cache.items.length) return Cache.items;

  if (!Cache.refreshing) {
    Cache.refreshing = refreshCache().finally(() => {
      Cache.refreshing = null;
    });
  }
  await Cache.refreshing;
  return Cache.items;
}

app.get("/api/items", async (req, res) => {
  const limit = Math.max(1, Math.min(200, Number(req.query.limit || 80)));
  const items = await ensureCache();
  res.json({
    refreshed_at: new Date(Cache.lastRefresh).toISOString(),
    items: items.slice(0, limit).map(({ content, ...rest }) => rest),
  });
});

app.get("/api/item/:id", async (req, res) => {
  await ensureCache();
  const id = String(req.params.id || "");
  const item = Cache.byId.get(id);
  if (!item) {
    res.status(404).json({ error: "Not found" });
    return;
  }
  res.json(item);
});

// Let static hosting keep working (index.html is the entry point)
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.listen(PORT, () => {
  console.log(`The Mainframe server listening on http://localhost:${PORT}`);
});

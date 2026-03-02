/**
 * Reader page script
 * - Fetches item content from the backend when available.
 * - Falls back to a minimal local message if the backend isn't running.
 */

async function getItem(id) {
  try {
    const res = await fetch(`/api/item/${encodeURIComponent(id)}`, { headers: { "accept": "application/json" } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } catch {
    return {
      title: "Backend offline",
      source: "The Mainframe",
      blurb: "Start the server (npm install && npm run dev) to load the live reader content.",
      published_at: new Date().toISOString(),
      severity: "info",
      badges: ["Demo"],
      url: "#",
      content: "This reader page fetches full content from /api/item/:id.\n\nOnce the backend is running, open a card from the dashboard again.",
    };
  }
}

function qs(name) {
  const p = new URLSearchParams(location.search);
  return p.get(name) || "";
}

function timeSince(iso) {
  const d = new Date(iso);
  const ms = Date.now() - d.getTime();
  const min = Math.round(ms / 60000);
  if (min < 1) return "just now";
  if (min < 60) return `${min}m ago`;
  const h = Math.round(min / 60);
  return `${h}h ago`;
}

document.addEventListener("DOMContentLoaded", async () => {
  const id = qs("id");
  const item = await getItem(id);

  document.title = `${item.title} — The Mainframe`;
  document.getElementById("title").textContent = item.title;
  document.getElementById("source").textContent = `via ${item.source}`;
  document.getElementById("blurb").textContent = item.blurb || "";
  document.getElementById("time").textContent = timeSince(item.published_at);

  const sev = document.getElementById("severity");
  if (item.severity) {
    sev.classList.add(item.severity);
    sev.textContent = String(item.severity).toUpperCase();
  } else {
    sev.remove();
  }

  const badges = document.getElementById("badges");
  if (item.badges && item.badges.length) {
    badges.textContent = item.badges.slice(0, 3).join(" • ");
  } else {
    badges.remove();
  }

  const link = document.getElementById("open-original");
  link.href = item.url || "#";

  const content = document.getElementById("content");
  content.textContent = "";
  const blocks = String(item.content || "").split(/\n\n+/).map(s => s.trim()).filter(Boolean);
  if (!blocks.length) {
    const p = document.createElement("p");
    p.textContent = "No content available.";
    content.appendChild(p);
    return;
  }
  for (const b of blocks) {
    const p = document.createElement("p");
    p.textContent = b;
    content.appendChild(p);
  }
});
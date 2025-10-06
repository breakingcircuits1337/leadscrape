/**
 * Reader page script
 * - Pulls the same mocked dataset as mainframe.js for demo purposes.
 * - In production this page would receive an ID, call the backend, and render sanitized content.
 */

function dataset() {
  // Keep in sync with mainframe.js mocked data keys
  return [
    {
      id: "cisa-kev-2025-0001",
      source: "CISA",
      title: "CISA adds CVE-2025-0001 to Known Exploited Vulnerabilities",
      blurb: "Active exploitation observed in the wild. Federal agencies must remediate by the specified due date.",
      url: "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
      published_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
      severity: "critical",
      badges: ["KEV", "Exploit In The Wild"],
      content: `
        CISA has added CVE-2025-0001 to the Known Exploited Vulnerabilities (KEV) catalog.
        Federal agencies must remediate by the due date. Indicators and mitigations are provided
        in the advisory. For more details see the original publication.
      `
    },
    {
      id: "nvd-ms-azure-cves",
      source: "NVD",
      title: "Multiple Azure services impacted by elevation of privilege flaws",
      blurb: "NVD published CVEs affecting Azure compute with patches available. Exploitation less likely per MSRC.",
      url: "https://nvd.nist.gov/",
      published_at: new Date(Date.now() - 25 * 60 * 1000).toISOString(),
      severity: "high",
      badges: ["Cloud", "CVE"],
      content: `
        Microsoft has released patches addressing multiple elevation-of-privilege vulnerabilities in Azure services.
        While exploitation is assessed as less likely, administrators should prioritize updates and review MSRC guidance.
      `
    },
    {
      id: "arxiv-crypto-scheme",
      source: "arXiv",
      title: "On the Security of Post-Quantum Lattice Schemes under Chosen-Ciphertext",
      blurb: "New preprint analyzes side-channel resilience and proposes mitigations for key encapsulation mechanisms.",
      url: "https://arxiv.org/",
      published_at: new Date(Date.now() - 65 * 60 * 1000).toISOString(),
      severity: "info",
      badges: ["Research"],
      content: `
        The authors analyze the security of lattice-based KEMs under CCA and propose countermeasures
        against timing and power-analysis vectors. This work informs implementers of PQC schemes.
      `
    }
  ];
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

document.addEventListener("DOMContentLoaded", () => {
  const id = qs("id");
  const item = dataset().find(i => i.id === id) || { title: "Not found", source: "-", blurb: "", published_at: new Date().toISOString(), severity: "", badges: [], url: "#" };

  document.title = `${item.title} — The Mainframe`;
  document.getElementById("title").textContent = item.title;
  document.getElementById("source").textContent = `via ${item.source}`;
  document.getElementById("blurb").textContent = item.blurb || "";
  document.getElementById("time").textContent = timeSince(item.published_at);
  const sev = document.getElementById("severity");
  if (item.severity) {
    sev.classList.add(item.severity);
    sev.textContent = item.severity.toUpperCase();
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
  content.textContent = ""; // clear
  const p = document.createElement("p");
  p.textContent = (item.content || "").trim();
  content.appendChild(p);
});
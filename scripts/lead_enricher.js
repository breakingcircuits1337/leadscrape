/**
 * Lead Enricher
 * - Reads a CSV of leads
 * - Fetches each website and common contact pages
 * - Extracts any public email addresses
 * - Writes an enriched CSV with discovered emails filled in when missing
 *
 * Usage:
 *   node scripts/lead_enricher.js assets/leads/hairdressers_pulaski_va.csv assets/leads/hairdressers_pulaski_va_enriched.csv
 *
 * Notes:
 * - This uses Node's http/https and zlib; no external dependencies.
 * - Many third-party platforms (e.g., Facebook, Vagaro) do not expose emails and may block scraping.
 * - This script focuses on a business's own website and common contact/about pages.
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const zlib = require('zlib');

const INPUT = process.argv[2] || path.join('assets', 'leads', 'hairdressers_pulaski_va.csv');
const OUTPUT = process.argv[3] || path.join('assets', 'leads', 'hairdressers_pulaski_va_enriched.csv');

const EMAIL_PREF_ORDER = ['info', 'contact', 'hello', 'booking', 'appointments', 'support'];

// Minimal CSV parsing that supports quoted fields and commas inside quotes
function parseCSV(text) {
  const rows = [];
  let i = 0;
  const len = text.length;

  function readCell() {
    let cell = '';
    let quoted = false;

    if (text[i] === '"') {
      quoted = true;
      i++; // consume opening quote
      while (i < len) {
        const ch = text[i];
        if (ch === '"') {
          if (text[i + 1] === '"') {
            cell += '"'; // escaped quote
            i += 2;
            continue;
          } else {
            i++; // closing quote
            break;
          }
        } else {
          cell += ch;
          i++;
        }
      }
    } else {
      while (i < len && text[i] !== ',' && text[i] !== '\n' && text[i] !== '\r') {
        cell += text[i++];
      }
    }
    return cell;
  }

  while (i < len) {
    const row = [];
    // skip line breaks at start
    while (i < len && (text[i] === '\n' || text[i] === '\r')) i++;
    if (i >= len) break;
    // read row
    while (i < len) {
      const cell = readCell();
      row.push(cell);
      if (text[i] === ',') {
        i++; // consume comma and continue
        continue;
      }
      // end of row
      while (i < len && text[i] !== '\n' && text[i] !== '\r') i++;
      // consume newline(s)
      while (i < len && (text[i] === '\n' || text[i] === '\r')) i++;
      break;
    }
    // avoid pushing empty trailing row
    if (!(row.length === 1 && row[0] === '')) {
      rows.push(row);
    }
  }
  return rows;
}

function escapeCSVCell(cell) {
  if (cell == null) return '';
  const str = String(cell);
  const needsQuote = /[,"\n\r]/.test(str);
  if (!needsQuote) return str;
  return `"${str.replace(/"/g, '""')}"`;
}

function stringifyCSV(rows) {
  return rows
    .map((row) => row.map(escapeCSVCell).join(','))
    .join('\n');
}

function normalizeURL(url) {
  if (!url) return null;
  let u = url.trim();
  if (!u) return null;
  // MapQuest/Vagaro pages can be used directly; ensure protocol
  if (!/^https?:\/\//i.test(u)) {
    u = 'https://' + u;
  }
  return u;
}

function httpGet(url, timeoutMs = 12000) {
  return new Promise((resolve) => {
    const u = new URL(url);
    const client = u.protocol === 'http:' ? http : https;
    const req = client.request(
      {
        protocol: u.protocol,
        hostname: u.hostname,
        port: u.port || (u.protocol === 'http:' ? 80 : 443),
        path: u.pathname + (u.search || ''),
        method: 'GET',
        headers: {
          'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          'Accept-Encoding': 'gzip, deflate',
          'Connection': 'close',
        },
      },
      (res) => {
        // Handle redirects
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          const loc = res.headers.location;
          const next =
            loc.startsWith('http://') || loc.startsWith('https://')
              ? loc
              : `${u.protocol}//${u.hostname}${loc.startsWith('/') ? loc : path.posix.join(u.pathname, loc)}`;
          resolve(httpGet(next, timeoutMs));
          return;
        }

        let stream = res;
        const enc = (res.headers['content-encoding'] || '').toLowerCase();
        if (enc.includes('gzip')) {
          const gunzip = zlib.createGunzip();
          stream = res.pipe(gunzip);
        } else if (enc.includes('deflate')) {
          const inflate = zlib.createInflate();
          stream = res.pipe(inflate);
        }

        const chunks = [];
        stream.on('data', (d) => chunks.push(d));
        stream.on('end', () => {
          const buf = Buffer.concat(chunks);
          resolve(buf.toString('utf8'));
        });
      }
    );
    req.on('error', () => resolve(null));
    req.setTimeout(timeoutMs, () => {
      req.destroy();
      resolve(null);
    });
    req.end();
  });
}

function buildCandidateURLs(base) {
  const candidates = [];
  const normalized = normalizeURL(base);
  if (!normalized) return candidates;

  candidates.push(normalized);
  // append common contact/about routes (trailing slash variants included)
  const common = [
    '/contact',
    '/contact-us',
    '/about',
    '/about-us',
    '/team',
    '/our-team',
    '/staff',
    '/booking',
    '/book',
  ];
  // For squarespace-like sites that already have extra path in URL, skip naive concatenation for directories like MapQuest/Vagaro
  try {
    const u = new URL(normalized);
    const host = u.hostname.toLowerCase();
    const isDirectoryHost =
      host.includes('mapquest.com') || host.includes('vagaro.com') || host.includes('facebook.com');
    if (!isDirectoryHost) {
      for (const suffix of common) {
        const joined = normalized.endsWith('/') ? normalized.slice(0, -1) + suffix : normalized + suffix;
        candidates.push(joined);
        candidates.push(joined + '/');
      }
    }
  } catch {
    // ignore
  }
  return candidates;
}

function extractEmails(html) {
  if (!html) return [];
  const set = new Set();
  // mailto capture
  const mailtoRegex = /mailto:([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})/gi;
  let m;
  while ((m = mailtoRegex.exec(html)) !== null) {
    set.add(m[1].trim());
  }
  // raw emails in text
  const emailRegex = /([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})/gi;
  while ((m = emailRegex.exec(html)) !== null) {
    set.add(m[1].trim());
  }
  return Array.from(set);
}

function choosePreferredEmail(emails) {
  if (!emails || emails.length === 0) return null;
  const lower = emails.map((e) => e.toLowerCase());
  for (const pref of EMAIL_PREF_ORDER) {
    const match = lower.find((e) => e.includes(pref + '@') || e.includes('.' + pref + '@') || e.includes(pref));
    if (match) return emails[lower.indexOf(match)];
  }
  // fallback first
  return emails[0];
}

async function discoverEmailForWebsite(website) {
  const candidates = buildCandidateURLs(website);
  for (const url of candidates) {
    try {
      const html = await httpGet(url);
      const emails = extractEmails(html);
      if (emails.length > 0) {
        const preferred = choosePreferredEmail(emails);
        if (preferred) return preferred;
        return emails[0];
      }
    } catch {
      // ignore individual fetch errors
    }
  }
  return null;
}

async function main() {
  const raw = fs.readFileSync(INPUT, 'utf8');
  const rows = parseCSV(raw);
  if (rows.length === 0) {
    console.error('No rows found in input CSV.');
    return;
  }

  // Header mapping
  const header = rows[0];
  const nameIdx = header.indexOf('Name of business');
  const ownerIdx = header.indexOf('Name of business owner');
  const websiteIdx = header.indexOf('Website');
  const phoneIdx = header.indexOf('Phone number');
  const emailIdx = header.indexOf('Email');
  const summaryIdx = header.indexOf('Summary of business');
  const pitchIdx = header.indexOf('Cold outreach email suggestion');
  const nextStepIdx = header.indexOf('Next step to uncover email');

  if (websiteIdx === -1 || emailIdx === -1) {
    console.error('Required headers not found. Ensure CSV includes "Website" and "Email" columns.');
    return;
  }

  const out = [header.slice()];
  // Limit concurrency to avoid hammering
  const concurrency = 3;
  let index = 1;

  async function worker(row) {
    const website = (row[websiteIdx] || '').trim();
    const email = (row[emailIdx] || '').trim();

    if (!email && website) {
      const found = await discoverEmailForWebsite(website);
      if (found) {
        row[emailIdx] = found;
      }
    }
    out.push(row.slice());
  }

  const tasks = [];
  while (index < rows.length) {
    const batch = [];
    for (let c = 0; c < concurrency && index < rows.length; c++, index++) {
      // skip any empty rows
      if (rows[index].length === 1 && rows[index][0] === '') continue;
      batch.push(worker(rows[index]));
    }
    // eslint-disable-next-line no-await-in-loop
    await Promise.all(batch);
  }

  const csv = stringifyCSV(out);
  fs.writeFileSync(OUTPUT, csv, 'utf8');
  console.log(`Enriched CSV written to: ${OUTPUT}`);
}

if (require.main === module) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
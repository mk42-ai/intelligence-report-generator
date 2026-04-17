require("dotenv").config();

// Polyfill global.crypto for Node < 20 (Azure SDK v12 requires it at import time).
// On Node 20+ this is a no-op because global.crypto already exists.
if (typeof globalThis.crypto === "undefined") {
    globalThis.crypto = require("crypto").webcrypto;
}

const express = require("express");
const path = require("path");
const Handlebars = require("handlebars");
const bodyParser = require("body-parser");
const app = express();
const puppeteer = require("puppeteer");
const fs = require("fs");
const {
    BlobServiceClient,
    StorageSharedKeyCredential,
    generateBlobSASQueryParameters,
    BlobSASPermissions,
} = require("@azure/storage-blob");

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json({ limit: "50mb" }));

// ---------------------------------------------------------
// AZURE BLOB STORAGE — permanent PDF persistence
// Set these 3 env vars on the OnDemand endpoint:
//   AZURE_STORAGE_ACCOUNT_NAME
//   AZURE_STORAGE_ACCOUNT_KEY
//   AZURE_STORAGE_CONTAINER_NAME
// If unset, falls back to local disk (dev only — ephemeral on serverless).
// ---------------------------------------------------------
const azAccountKey = process.env.AZURE_STORAGE_ACCOUNT_KEY || "mock-key";
const azAccountName = process.env.AZURE_STORAGE_ACCOUNT_NAME || "mock-name";
const azContainerName = process.env.AZURE_STORAGE_CONTAINER_NAME || "mock-container";

// Local-disk fallback only for dev (container restarts wipe this on serverless)
const PDF_STORAGE_DIR = path.join(__dirname, 'pdf-storage');
if (!fs.existsSync(PDF_STORAGE_DIR)) fs.mkdirSync(PDF_STORAGE_DIR, { recursive: true });

function getRandomString(length = 16) {
    const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let result = "";
    for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return result;
}

// Upload PDF bytes to Azure Blob Storage, return a 15-minute pre-signed SAS URL.
// Falls back to local disk save + local download URL when no Azure creds set.
async function uploadPdfToAzure(pdfBytes) {
    // Dev fallback: no Azure creds -> write to local disk
    if (azAccountName === "mock-name") {
        const pdfId = getRandomString(16);
        const filePath = path.join(PDF_STORAGE_DIR, `${pdfId}.pdf`);
        fs.writeFileSync(filePath, pdfBytes);
        console.log(`[dev] Azure creds not set — saved locally: ${filePath}`);
        return { url: null, pdfId, local: true };
    }

    const sharedKeyCredential = new StorageSharedKeyCredential(azAccountName, azAccountKey);
    const blobServiceClient = new BlobServiceClient(
        `https://${azAccountName}.blob.core.windows.net`,
        sharedKeyCredential
    );
    const containerClient = blobServiceClient.getContainerClient(azContainerName);

    if (!(await containerClient.exists())) {
        throw new Error(`Azure container '${azContainerName}' does not exist`);
    }

    const pdfId = getRandomString(16);
    const blobName = `g42-report-${pdfId}.pdf`;
    const blockBlobClient = containerClient.getBlockBlobClient(blobName);
    await blockBlobClient.uploadData(pdfBytes, {
        blobHTTPHeaders: {
            blobContentType: 'application/pdf',
            // attachment forces browser to download instead of rendering inline.
            // Filename includes the date so downloads are self-describing.
            blobContentDisposition: `attachment; filename="G42-Media-Intelligence-${pdfId}.pdf"`,
        },
    });

    // 6-hour read-only SAS token
    const expiryDate = new Date();
    expiryDate.setHours(expiryDate.getHours() + 6);

    const sasToken = generateBlobSASQueryParameters({
        containerName: azContainerName,
        blobName,
        permissions: BlobSASPermissions.parse("r"),
        startsOn: new Date(),
        expiresOn: expiryDate,
    }, sharedKeyCredential).toString();

    const url = `${blockBlobClient.url}?${sasToken}`;
    return { url, pdfId, blobName, local: false };
}

// Local-disk download fallback for dev mode. In production (Azure mode),
// the /generate response returns a direct Azure SAS URL — this route just serves
// local dev PDFs.
app.get('/g42-report/download/:id', (req, res) => {
    const filePath = path.join(PDF_STORAGE_DIR, `${req.params.id}.pdf`);
    if (!fs.existsSync(filePath)) return res.status(404).json({ error: "PDF not found (Azure mode: use the SAS URL from /generate)" });
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `inline; filename="g42-intel-${req.params.id}.pdf"`);
    res.send(fs.readFileSync(filePath));
});

// List local dev PDFs (Azure-mode storage is listed via Azure portal or Storage API directly)
app.get('/g42-report/list', (req, res) => {
    if (azAccountName !== "mock-name") {
        return res.json({
            message: "Azure mode active — PDFs stored in container '" + azContainerName + "'. Use Azure portal or Storage SDK to list.",
            azureContainer: azContainerName,
        });
    }
    const files = fs.readdirSync(PDF_STORAGE_DIR)
        .filter(f => f.endsWith('.pdf'))
        .map(f => {
            const stat = fs.statSync(path.join(PDF_STORAGE_DIR, f));
            return {
                id: f.replace('.pdf', ''),
                size: stat.size,
                created: stat.birthtime,
            };
        })
        .sort((a, b) => b.created - a.created);
    res.json({ count: files.length, reports: files, mode: "local-dev" });
});

// ---------------------------------------------------------
// CONFIGURATION & HELPERS
// ---------------------------------------------------------
const TEMPLATE_PATH = path.join(__dirname, 'G42 Report/template.html');

Handlebars.registerHelper('safe', text => new Handlebars.SafeString(text));
Handlebars.registerHelper('eq', (a, b) => a === b);
Handlebars.registerHelper('gt', (a, b) => Number(a) > Number(b));
Handlebars.registerHelper('add', (a, b) => Number(a) + Number(b));
Handlebars.registerHelper('json', obj => JSON.stringify(obj));

// ---------------------------------------------------------
// SVG ICON LIBRARY — inline SVG for platform logos + metric icons
// ---------------------------------------------------------
const ICONS = {
    linkedin: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>`,
    x: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>`,
    instagram: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.163c3.204 0 3.584.012 4.849.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.849.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z"/></svg>`,
    youtube: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>`,
    facebook: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>`,
    news: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M2.5 4a.5.5 0 0 0-.5.5v15a.5.5 0 0 0 .5.5h11a.5.5 0 0 0 .5-.5V15h6a.5.5 0 0 0 .5-.5v-10a.5.5 0 0 0-.5-.5H2.5zM4 6h12v8h-3V9H4V6zm0 5h7v6H4v-6zm13.5 2.5V8h1v5.5h-1z"/></svg>`,
    heart: `<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:middle;"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>`,
    repost: `<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:middle;"><path d="M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z"/></svg>`,
    comment: `<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:middle;"><path d="M21.99 4c0-1.1-.89-2-1.99-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h14l4 4-.01-18z"/></svg>`,
    view: `<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:middle;"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>`,
    share: `<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:middle;"><path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92 1.61 0 2.92-1.31 2.92-2.92s-1.31-2.92-2.92-2.92z"/></svg>`,
};

function getSourceIcon(source) {
    const s = (source || "").toLowerCase();
    if (s.includes('linkedin')) return { class: 'linkedin', svg: ICONS.linkedin };
    if (s === 'x' || s.includes('twitter') || s.includes('x (')) return { class: 'x', svg: ICONS.x };
    if (s.includes('instagram')) return { class: 'instagram', svg: ICONS.instagram };
    if (s.includes('youtube')) return { class: 'youtube', svg: ICONS.youtube };
    if (s.includes('facebook')) return { class: 'facebook', svg: ICONS.facebook };
    if (s.includes('reuters')) return { class: 'reuters', svg: ICONS.news };
    if (s.includes('bloomberg')) return { class: 'bloomberg', svg: ICONS.news };
    if (s.includes('financial times') || s === 'ft') return { class: 'ft', svg: ICONS.news };
    if (s.includes('new york times') || s === 'nyt') return { class: 'nyt', svg: ICONS.news };
    if (s.includes('bbc')) return { class: 'bbc', svg: ICONS.news };
    return { class: 'news', svg: ICONS.news };
}

function getPlatformIcon(platform) {
    const p = (platform || "").toLowerCase();
    if (p.includes('linkedin')) return { class: 'linkedin', svg: ICONS.linkedin };
    if (p === 'x' || p.includes('twitter') || p.includes('x (')) return { class: 'x', svg: ICONS.x };
    if (p.includes('instagram')) return { class: 'instagram', svg: ICONS.instagram };
    if (p.includes('youtube')) return { class: 'youtube', svg: ICONS.youtube };
    if (p.includes('facebook')) return { class: 'facebook', svg: ICONS.facebook };
    return { class: 'news', svg: ICONS.news };
}

function sanitizeText(text) {
    if (!text) return "";
    return String(text)
        .replace(/[\u200B-\u200F\u2028-\u202F\uFEFF\u00AD]/g, '')
        .replace(/[\uE000-\uF8FF]/g, '')
        .replace(/[\uD800-\uDFFF]/g, '')
        .replace(/[\u2013\u2014]/g, '-')
        .replace(/[\u2018\u2019]/g, "'")
        .replace(/[\u201C\u201D]/g, '"')
        .replace(/[\u2022\u2023\u25E6\u2043]/g, '•')
        .replace(/\u2026/g, '...')
        .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
}

function getSentimentClass(sentiment) {
    const s = (sentiment || "").toLowerCase();
    if (s.includes("positive") || s.includes("pos")) return "positive";
    if (s.includes("negative") || s.includes("neg")) return "negative";
    return "neutral";
}

function getImpactClass(impact) {
    const i = (impact || "").toLowerCase();
    if (i.includes("tier 1") || i.includes("tier-1") || i.includes("tier1") || i === "high" || i.includes("critical")) return "tier1";
    if (i.includes("tier 2") || i.includes("tier-2") || i.includes("tier2") || i === "medium") return "tier2";
    return "tier3";
}

function getStatClass(type) {
    const t = (type || "").toLowerCase();
    if (t.includes("positive")) return "positive";
    if (t.includes("negative")) return "negative";
    if (t.includes("neutral")) return "neutral";
    if (t.includes("gold") || t.includes("highlight")) return "gold";
    return "";
}

function getInitials(name) {
    if (!name) return "?";
    const cleaned = String(name).replace(/^(H\.?H\.?|Sheikh|Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+/i, '').trim();
    const parts = cleaned.split(/\s+/).filter(Boolean);
    if (parts.length === 0) return "?";
    if (parts.length === 1) return parts[0].substring(0, 2).toUpperCase();
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

// Build a CSS conic-gradient for sentiment donut
function buildDonutGradient(pos, neu, neg) {
    const total = pos + neu + neg || 1;
    const posEnd = (pos / total) * 100;
    const neuEnd = posEnd + (neu / total) * 100;
    return `conic-gradient(
        #22c55e 0% ${posEnd.toFixed(2)}%,
        #64748b ${posEnd.toFixed(2)}% ${neuEnd.toFixed(2)}%,
        #ef4444 ${neuEnd.toFixed(2)}% 100%
    )`;
}

function computeSentimentSummary(positive, neutral, negative) {
    const total = (positive || 0) + (neutral || 0) + (negative || 0);
    if (total === 0) return null;
    const positive_pct = Math.round((positive / total) * 100);
    const neutral_pct = Math.round((neutral / total) * 100);
    const negative_pct = 100 - positive_pct - neutral_pct;
    // Score: +1 for positive, 0 for neutral, -1 for negative, normalized to 0-100
    const raw = (positive - negative) / total;
    const score = ((raw + 1) / 2 * 100).toFixed(0);
    return {
        positive, neutral, negative,
        positive_pct, neutral_pct, negative_pct,
        score,
        donut_gradient: buildDonutGradient(positive, neutral, negative),
    };
}

// Fetch an image URL and return a base64 data URI. Returns null on failure.
// This makes the PDF robust against CORS, hotlink blocks, and rate-limiting at render time —
// once the server has the bytes, Puppeteer renders from inline data with zero network deps.
async function fetchAndInlineImage(url, timeoutMs = 12000) {
    if (!url || typeof url !== 'string') return null;
    try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        const res = await fetch(url, {
            signal: controller.signal,
            headers: {
                // Pose as a regular browser so CDN/hotlink filters don't reject
                'User-Agent': 'Mozilla/5.0 (G42IntelBot/1.0; +https://g42.ai)',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            },
        });
        clearTimeout(timer);
        if (!res.ok) {
            console.warn(`Image fetch ${res.status}: ${url.substring(0, 80)}`);
            return null;
        }
        const ct = res.headers.get('content-type') || 'image/jpeg';
        // Size guard — skip anything > 5 MB (unreasonable for a report image)
        const len = parseInt(res.headers.get('content-length') || '0', 10);
        if (len > 5 * 1024 * 1024) {
            console.warn(`Image too large (${len} bytes): ${url.substring(0, 80)}`);
            return null;
        }
        const buf = Buffer.from(await res.arrayBuffer());
        if (buf.length > 5 * 1024 * 1024) return null;
        return `data:${ct};base64,${buf.toString('base64')}`;
    } catch (e) {
        console.warn(`Image fetch failed (${e.name}): ${url.substring(0, 80)}`);
        return null;
    }
}

// Validate and clean a URL. Returns undefined if the URL looks malformed.
// Catches agent-side bugs: trailing ellipses, truncated URLs, non-http schemes,
// obvious tweet-as-article substitutions (stories mislabeled as Tier-1 but pointing to x.com).
function cleanUrl(rawUrl) {
    if (!rawUrl || typeof rawUrl !== 'string') return undefined;
    let u = rawUrl.trim();
    // Strip trailing ellipsis characters (agent truncation artifact)
    u = u.replace(/[\u2026]+$/g, '').replace(/\.{3,}$/g, '');
    // Strip trailing whitespace / quotes
    u = u.replace(/[\s"'`]+$/g, '');
    // Must be http or https
    if (!/^https?:\/\//i.test(u)) return undefined;
    // Try to parse — reject if URL constructor fails
    try {
        const parsed = new URL(u);
        if (!parsed.hostname || parsed.hostname.length < 3) return undefined;
        return parsed.toString();
    } catch {
        return undefined;
    }
}

function processMention(m) {
    const source = sanitizeText(m.source || m.outlet || "Source");
    const icon = getSourceIcon(source);
    return {
        source,
        source_icon_class: icon.class,
        source_icon_svg: icon.svg,
        title: sanitizeText(m.title || m.headline || ""),
        snippet: sanitizeText(m.snippet || m.summary || ""),
        url: cleanUrl(m.url),
        timestamp: sanitizeText(m.timestamp || m.time || m.date || ""),
        sentiment: sanitizeText(m.sentiment || "Neutral"),
        sentiment_class: getSentimentClass(m.sentiment),
        impact: m.impact ? sanitizeText(m.impact) : (m.tier ? `Tier ${m.tier}` : undefined),
        impact_class: getImpactClass(m.impact || m.tier),
        engagement: m.engagement || undefined,
    };
}

function processPost(p) {
    const metrics = [];
    if (p.likes != null) metrics.push({ icon_svg: ICONS.heart, value: String(p.likes), label: 'likes' });
    if (p.reposts != null || p.retweets != null) metrics.push({ icon_svg: ICONS.repost, value: String(p.reposts ?? p.retweets), label: 'reposts' });
    if (p.comments != null) metrics.push({ icon_svg: ICONS.comment, value: String(p.comments), label: 'comments' });
    if (p.views != null) metrics.push({ icon_svg: ICONS.view, value: String(p.views), label: 'views' });
    if (p.shares != null) metrics.push({ icon_svg: ICONS.share, value: String(p.shares), label: 'shares' });

    const cleanedUrl = cleanUrl(p.url);
    let urlDisplay = "";
    if (cleanedUrl) {
        try { urlDisplay = new URL(cleanedUrl).hostname.replace('www.', ''); }
        catch { urlDisplay = cleanedUrl.substring(0, 30); }
    }

    return {
        author_name: sanitizeText(p.author_name || p.author || "Unknown"),
        author_handle: sanitizeText(p.author_handle || p.handle || ""),
        author_role: p.author_role ? sanitizeText(p.author_role) : undefined,
        author_initials: getInitials(p.author_name || p.author),
        author_is_g42: p.author_is_g42 === true,
        verified: p.verified === true,
        timestamp: sanitizeText(p.timestamp || p.time || ""),
        content_html: sanitizeText(p.content || p.text || ""),
        metrics,
        url: cleanedUrl,
        url_display: urlDisplay,
        sentiment: sanitizeText(p.sentiment || "Neutral"),
        sentiment_class: getSentimentClass(p.sentiment),
    };
}

// ---------------------------------------------------------
// DIGEST PROCESSORS — agency-style text-first items
// ---------------------------------------------------------

// Process a single digest mention: Outlet + Title + Date + (optional) context blurb.
// No sentiment, no engagement metrics — matches the real agency report format.
function processDigestMention(m) {
    const outlet = sanitizeText(m.outlet || m.source || "");
    const icon = getSourceIcon(outlet);
    return {
        outlet,
        outlet_icon_class: icon.class,
        outlet_icon_svg: icon.svg,
        title: sanitizeText(m.title || m.headline || ""),
        url: cleanUrl(m.url),
        date: sanitizeText(m.date || m.publication_date || m.timestamp || ""),
        context: m.context ? sanitizeText(m.context) : (m.note ? sanitizeText(m.note) : undefined),
    };
}

// Build 1-N digest pages with auto-split.
// Every bucket is ALWAYS rendered — empty buckets show the emptyLabel ("No notable mentions." etc.)
// sectionRef → top-bar label; eyebrow/title/desc → heading; buckets → [{ name, items }]
function buildDigestPages(sectionRef, eyebrow, title, desc, buckets, emptyLabel = "No notable mentions.", subnote = null) {
    const processedBuckets = (Array.isArray(buckets) ? buckets : []).map(b => {
        const rawItems = Array.isArray(b.items) ? b.items : [];
        const items = rawItems.map(processDigestMention);
        return {
            name: sanitizeText(b.name || ""),
            items,
            has_items: items.length > 0,
            item_count: items.length > 0 ? String(items.length) : undefined,
            empty_label: sanitizeText(b.empty_label || emptyLabel),
        };
    });

    // Pagination: 1 unit per bucket header + 1 unit per item (minimum 1 for empty bucket).
    // ~22 units per page is comfortable at 10.5px font with the current layout.
    const MAX_UNITS = 22;
    const pages = [];
    let pageBuckets = [];
    let pageUnits = 0;
    let isFirst = true;

    for (const bucket of processedBuckets) {
        const bucketUnits = 1 + Math.max(bucket.items.length, 1);
        const wouldOverflow = pageUnits + bucketUnits > MAX_UNITS;
        const bucketIsOversized = bucketUnits > MAX_UNITS;

        if (wouldOverflow && pageBuckets.length > 0 && !bucketIsOversized) {
            pages.push({
                is_digest_page: true,
                section_ref: sectionRef,
                digest_eyebrow: eyebrow,
                digest_title: isFirst ? title : `${title} (cont.)`,
                digest_desc: isFirst ? desc : undefined,
                buckets: pageBuckets,
            });
            pageBuckets = [];
            pageUnits = 0;
            isFirst = false;
        }

        // If a single bucket is oversized, flush accumulated buckets first, then split.
        if (bucketIsOversized) {
            if (pageBuckets.length > 0) {
                pages.push({
                    is_digest_page: true,
                    section_ref: sectionRef,
                    digest_eyebrow: eyebrow,
                    digest_title: isFirst ? title : `${title} (cont.)`,
                    digest_desc: isFirst ? desc : undefined,
                    buckets: pageBuckets,
                });
                pageBuckets = [];
                pageUnits = 0;
                isFirst = false;
            }
            const PER_PAGE = MAX_UNITS - 1;
            for (let i = 0; i < bucket.items.length; i += PER_PAGE) {
                pages.push({
                    is_digest_page: true,
                    section_ref: sectionRef,
                    digest_eyebrow: eyebrow,
                    digest_title: isFirst ? title : `${title} (cont.)`,
                    digest_desc: isFirst ? desc : undefined,
                    buckets: [{
                        ...bucket,
                        items: bucket.items.slice(i, i + PER_PAGE),
                        item_count: String(bucket.items.length),
                    }],
                });
                isFirst = false;
            }
            continue;
        }

        pageBuckets.push(bucket);
        pageUnits += bucketUnits;
    }

    if (pageBuckets.length > 0) {
        const lastPage = {
            is_digest_page: true,
            section_ref: sectionRef,
            digest_eyebrow: eyebrow,
            digest_title: isFirst ? title : `${title} (cont.)`,
            digest_desc: isFirst ? desc : undefined,
            buckets: pageBuckets,
        };
        if (subnote) lastPage.digest_subnote = subnote;
        pages.push(lastPage);
    } else if (pages.length > 0 && subnote) {
        pages[pages.length - 1].digest_subnote = subnote;
    }

    return pages;
}

// ---------------------------------------------------------
// PUPPETEER RENDER
// ---------------------------------------------------------
const PAGE_WIDTH = 794;
const PAGE_HEIGHT = 1123;

async function renderDynamicPdf(html) {
    const launchOpts = {
        headless: "new",
        args: ["--no-sandbox", "--disable-setuid-sandbox"]
    };
    // Optional override: point to system Chrome/Chromium when Puppeteer's bundled Chromium
    // is unavailable (Docker slim images, CI, dev environments).
    if (process.env.PUPPETEER_EXECUTABLE_PATH) {
        launchOpts.executablePath = process.env.PUPPETEER_EXECUTABLE_PATH;
    }
    const browser = await puppeteer.launch(launchOpts);

    try {
        const page = await browser.newPage();
        await page.setViewport({ width: PAGE_WIDTH, height: PAGE_HEIGHT });
        await page.setContent(html, { waitUntil: "networkidle0", timeout: 60000 });

        await page.addStyleTag({ content: `
            @page { size: ${PAGE_WIDTH}px ${PAGE_HEIGHT}px; margin: 0; }
            html, body { margin: 0; padding: 0; }
            .page-container { max-width: ${PAGE_WIDTH}px; margin: 0; padding: 0; }
            .page-wrapper {
                width: ${PAGE_WIDTH}px;
                height: ${PAGE_HEIGHT}px;
                max-height: ${PAGE_HEIGHT}px;
                overflow: hidden;
                page-break-after: always;
                page-break-inside: avoid;
                box-sizing: border-box;
                position: relative;
                display: flex;
                flex-direction: column;
                padding: 78px 32px 44px;
                margin: 0;
            }
            .page-wrapper:last-child { page-break-after: avoid; }
            .page-brand-bar {
                position: fixed; top: 0; left: 0; right: 0; height: 62px; z-index: 1500;
            }
            .page-footer-bar {
                position: fixed; bottom: 0; left: 0; right: 0; height: 28px; z-index: 2000;
            }
        `});

        const pdfBuffer = await page.pdf({
            printBackground: true,
            width: `${PAGE_WIDTH}px`,
            height: `${PAGE_HEIGHT}px`,
            margin: { top: 0, right: 0, bottom: 0, left: 0 }
        });

        return Buffer.from(pdfBuffer);
    } finally {
        await browser.close();
    }
}

// ---------------------------------------------------------
// SESSION STORE (in-memory; sessions are ephemeral, PDFs are permanent)
// ---------------------------------------------------------
const sessionStore = new Map();

setInterval(() => {
    const now = Date.now();
    for (const [id, session] of sessionStore) {
        if (now - session.timestamp > 2 * 60 * 60 * 1000) {
            sessionStore.delete(id);
        }
    }
}, 10 * 60 * 1000);


// ---------------------------------------------------------
// API ENDPOINTS
// ---------------------------------------------------------

app.post('/g42-report/start', (req, res) => {
    let { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    sessionId = sessionId.toLowerCase().trim();

    if (sessionStore.has(sessionId)) {
        sessionStore.set(sessionId, { timestamp: Date.now(), data: {} });
        return res.json({ message: "Session reset", sessionId });
    }
    sessionStore.set(sessionId, { timestamp: Date.now(), data: {} });
    console.log(`G42 session started: ${sessionId}`);
    res.json({ message: "Session started", sessionId });
});

app.post('/g42-report/delete', (req, res) => {
    let { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    sessionId = sessionId.toLowerCase().trim();
    if (sessionStore.has(sessionId)) {
        sessionStore.delete(sessionId);
        return res.json({ message: "Session deleted", sessionId });
    }
    res.json({ message: "Session not found", sessionId });
});

app.post('/g42-report/update', (req, res) => {
    const body = req.body;
    if (!body.sessionId) return res.status(400).json({ error: "sessionId required" });
    const sessionId = body.sessionId.toLowerCase().trim();
    if (!sessionStore.has(sessionId)) return res.status(404).json({ error: "Session not found. Call /start first." });

    const session = sessionStore.get(sessionId);
    session.timestamp = Date.now();
    const data = session.data;

    for (const [key, value] of Object.entries(body)) {
        if (key === 'sessionId') continue;
        if (Array.isArray(value)) {
            if (Array.isArray(data[key])) data[key] = [...data[key], ...value];
            else data[key] = value;
        } else if (value && typeof value === 'object' && !Array.isArray(value)) {
            if (data[key] && typeof data[key] === 'object' && !Array.isArray(data[key])) {
                data[key] = { ...data[key], ...value };
            } else data[key] = value;
        } else {
            data[key] = value;
        }
    }
    const updatedKeys = Object.keys(body).filter(k => k !== 'sessionId');
    res.json({ message: "Data updated", updatedKeys, totalKeys: Object.keys(data).length });
});

app.get('/g42-report/sessions', (req, res) => {
    const sessions = [];
    for (const [id, session] of sessionStore) {
        sessions.push({
            sessionId: id,
            created: new Date(session.timestamp).toISOString(),
            dataKeys: Object.keys(session.data || {})
        });
    }
    res.json({ sessions });
});

// ---------------------------------------------------------
// GENERATE
// ---------------------------------------------------------
app.post('/g42-report/generate', async (req, res) => {
    let { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    sessionId = sessionId.toLowerCase().trim();
    if (!sessionStore.has(sessionId)) return res.status(404).json({ error: "Session not found" });

    try {
        const session = sessionStore.get(sessionId);
        const input = session.data || {};

        const today = new Date();
        const reportTitle = sanitizeText(input.report_title || "Daily Media Intelligence");
        const reportSubtitle = sanitizeText(input.report_subtitle || "G42 Executive Briefing — Tier-1 editorial, social, and policy signal across UAE, GCC, and global markets");
        const reportDateLong = sanitizeText(input.report_date_long || today.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' }));
        const reportDate = sanitizeText(input.report_date || today.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase());
        const reportId = sanitizeText(input.report_id || `G42-${today.toISOString().slice(0,10).replace(/-/g,'')}-${getRandomString(4).toUpperCase()}`);
        const classification = sanitizeText(input.classification || "CONFIDENTIAL");
        // Match agency phrasing exactly: "Media headlines from [previous day], at 5:30 PM EST"
        // Default uses yesterday's date in the agency's format.
        const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
        const yesterdayLong = yesterday.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
        const coverageWindow = sanitizeText(input.coverage_window || `Media headlines from ${yesterdayLong}, at 5:30 PM EST`);
        const publishedAt = sanitizeText(input.published_at || "05:00 GST Daily");
        const analyst = sanitizeText(input.analyst || "OnDemand OS · Autonomous Agent");
        const orgName = sanitizeText(input.org_name || "G42 INTELLIGENCE DESK");

        let allPages = [];

        // ══════════ PAGE 1 — COVER ══════════
        const coverStats = Array.isArray(input.key_stats) && input.key_stats.length > 0
            ? input.key_stats.map(s => ({
                value: String(s.value || "—"),
                label: sanitizeText(s.label || ""),
                sub: s.sub ? sanitizeText(s.sub) : undefined,
                stat_class: getStatClass(s.type),
            }))
            : undefined;

        let overallSentiment = null;
        if (input.sentiment && (input.sentiment.positive != null || input.sentiment.neutral != null || input.sentiment.negative != null)) {
            overallSentiment = computeSentimentSummary(
                Number(input.sentiment.positive || 0),
                Number(input.sentiment.neutral || 0),
                Number(input.sentiment.negative || 0)
            );
        }

        const execSummary = Array.isArray(input.executive_summary)
            ? input.executive_summary.map(sanitizeText)
            : (input.executive_summary ? [sanitizeText(input.executive_summary)] : []);

        const topTakeaways = Array.isArray(input.top_takeaways)
            ? input.top_takeaways.map(t => ({
                label: sanitizeText(t.label || "Key Signal"),
                text: sanitizeText(t.text || t),
            }))
            : undefined;

        allPages.push({
            is_cover_page: true,
            section_ref: "Cover",
            stats: coverStats,
            executive_summary_paragraphs: execSummary.length ? execSummary : undefined,
            overall_sentiment: overallSentiment,
            top_takeaways: topTakeaways,
        });

        // ══════════ SHEIKH TAHNOON SPOTLIGHT ══════════
        if (input.sheikh_tahnoon) {
            const t = input.sheikh_tahnoon;
            const mentions = Array.isArray(t.mentions) ? t.mentions.map(processMention) : [];

            allPages.push({
                is_tahnoon_page: true,
                section_ref: "Sheikh Tahnoon",
                tahnoon_stats: {
                    total_mentions: String(t.total_mentions ?? mentions.length),
                    reach: String(t.reach || "—"),
                    tier1_mentions: String(t.tier1_mentions ?? mentions.filter(m => m.impact_class === 'tier1').length),
                    sentiment_score: String(t.sentiment_score || "—"),
                },
                tahnoon_mentions: mentions.slice(0, 5),
                analyst_note: t.analyst_note ? sanitizeText(t.analyst_note) : undefined,
            });

            // Overflow continuation page
            if (mentions.length > 5) {
                const remaining = mentions.slice(5);
                for (let i = 0; i < remaining.length; i += 6) {
                    allPages.push({
                        is_tahnoon_page: true,
                        section_ref: "Sheikh Tahnoon",
                        tahnoon_stats: {
                            total_mentions: String(t.total_mentions ?? mentions.length),
                            reach: String(t.reach || "—"),
                            tier1_mentions: String(t.tier1_mentions ?? mentions.filter(m => m.impact_class === 'tier1').length),
                            sentiment_score: String(t.sentiment_score || "—"),
                        },
                        tahnoon_mentions: remaining.slice(i, i + 6),
                    });
                }
            }
        }

        // ══════════ G42 EXECUTIVES ══════════
        if (Array.isArray(input.g42_executives) && input.g42_executives.length > 0) {
            const execs = input.g42_executives.map(e => ({
                name: sanitizeText(e.name || ""),
                title: sanitizeText(e.title || ""),
                initials: getInitials(e.name),
                mention_count: String(e.mention_count ?? (e.mentions || []).length),
                tier1_count: String(e.tier1_count ?? "0"),
                sentiment: sanitizeText(e.sentiment || "—"),
                mentions: Array.isArray(e.mentions) ? e.mentions.slice(0, 3).map(m => ({
                    outlet: sanitizeText(m.outlet || m.source || ""),
                    headline: sanitizeText(m.headline || m.title || ""),
                    url: m.url ? String(m.url) : undefined,
                })) : [],
            }));

            // 6 execs per page (2-col grid, 3 rows)
            for (let i = 0; i < execs.length; i += 6) {
                allPages.push({
                    is_executives_page: true,
                    section_ref: "Executives",
                    section_title: sanitizeText(input.g42_executives_title || "G42 Executive Coverage"),
                    section_desc: sanitizeText(input.g42_executives_desc || "Mentions, sentiment, and Tier-1 pickups for named G42 leadership across the reporting window."),
                    executives: execs.slice(i, i + 6),
                });
            }
        }

        // ══════════ G42 CORPORATE & PORTFOLIO ══════════
        if (input.g42_corporate || input.portfolio_companies) {
            const corp = input.g42_corporate || {};
            const g42Mentions = Array.isArray(corp.mentions) ? corp.mentions.slice(0, 3).map(processMention) : [];
            const portfolio = Array.isArray(input.portfolio_companies) ? input.portfolio_companies.map(p => ({
                name: sanitizeText(p.name || ""),
                sector: sanitizeText(p.sector || ""),
                description: sanitizeText(p.description || ""),
                mention_count: String(p.mention_count ?? "0"),
                sentiment: sanitizeText(p.sentiment || "—"),
                top_story: p.top_story ? sanitizeText(p.top_story) : undefined,
            })) : [];

            allPages.push({
                is_portfolio_page: true,
                section_ref: "G42 & Portfolio",
                section_title: sanitizeText(input.portfolio_title || "G42 Corporate & Portfolio Companies"),
                section_desc: sanitizeText(input.portfolio_desc || "Mentions for G42 itself and portfolio entities: Space42, Presight, Inception, Core42, and other group companies."),
                g42_mentions: g42Mentions.length ? g42Mentions : undefined,
                portfolio: portfolio.length ? portfolio : undefined,
            });
        }

        // ══════════ PLATFORM PAGES ══════════
        const platformSections = [
            { key: 'linkedin_coverage', class: 'linkedin', title: 'LinkedIn Coverage', subtitle: 'Professional network mentions, executive posts, and company activity' },
            { key: 'x_coverage', class: 'x', title: 'X (Twitter) Coverage', subtitle: 'Real-time public conversation, high-signal accounts, and breaking news' },
            { key: 'instagram_coverage', class: 'instagram', title: 'Instagram Coverage', subtitle: 'Visual coverage from G42 channels and affiliated creators' },
            { key: 'youtube_coverage', class: 'youtube', title: 'YouTube Coverage', subtitle: 'Long-form video, interviews, and press conference coverage' },
            { key: 'news_coverage', class: 'news', title: 'Tier-1 Editorial Coverage', subtitle: 'Reuters, Bloomberg, FT, NYT, Washington Post, BBC, and wire pickup' },
        ];

        platformSections.forEach(ps => {
            const sec = input[ps.key];
            if (!sec) return;
            const icon = getPlatformIcon(ps.class);
            const posts = Array.isArray(sec.posts) ? sec.posts.map(processPost) : [];

            const pageBase = {
                is_platform_page: true,
                section_ref: ps.title,
                platform_class: ps.class,
                platform_icon_svg: icon.svg,
                platform_title: sanitizeText(sec.title || ps.title),
                platform_subtitle: sanitizeText(sec.subtitle || ps.subtitle),
                platform_stats: {
                    posts: String(sec.total_posts ?? posts.length),
                    reach: String(sec.reach || "—"),
                    engagement: String(sec.engagement || "—"),
                    sentiment: sanitizeText(sec.sentiment_summary || "—"),
                },
            };

            // Split ~4 posts per page (rich cards with metadata)
            const PER_PAGE = 4;
            if (posts.length === 0) {
                allPages.push({ ...pageBase, posts: [], analyst_note: sec.analyst_note ? sanitizeText(sec.analyst_note) : undefined });
            } else {
                for (let i = 0; i < posts.length; i += PER_PAGE) {
                    const isLast = i + PER_PAGE >= posts.length;
                    allPages.push({
                        ...pageBase,
                        posts: posts.slice(i, i + PER_PAGE),
                        analyst_note: isLast && sec.analyst_note ? sanitizeText(sec.analyst_note) : undefined,
                    });
                }
            }
        });

        // ══════════ FEATURED IMAGERY PAGE (magazine-style 5-image grid) ══════════
        // Agent pulls image URLs + captions + outlet attribution from Perplexity search
        // results. Up to 5 images: 1 featured hero + 2×2 grid.
        // Server pre-fetches each image and inlines as base64 data URI so the PDF is
        // robust against hotlink blocks, CORS, rate-limiting, and broken sources.
        // Shape: { title, description, images: [{ url, caption, outlet, date, source_url }] }
        {
            const data = input.featured_imagery;
            if (data && Array.isArray(data.images) && data.images.length > 0) {
                // Only take first 5 images, pre-fetch in parallel
                const rawImages = data.images.slice(0, 5);
                console.log(`Pre-fetching ${rawImages.length} imagery URLs in parallel...`);
                const fetchedImages = await Promise.all(
                    rawImages.map(async (img) => {
                        const cleanOriginal = cleanUrl(img.url);
                        const inlined = cleanOriginal ? await fetchAndInlineImage(cleanOriginal) : null;
                        return {
                            url: inlined,  // null if fetch failed
                            original_url: cleanOriginal,
                            caption: sanitizeText(img.caption || img.alt || ""),
                            outlet: sanitizeText(img.outlet || img.source || ""),
                            date: sanitizeText(img.date || img.timestamp || ""),
                            source_url: cleanUrl(img.source_url || img.article_url) || cleanOriginal,
                            meta: sanitizeText(img.meta || ""),
                        };
                    })
                );
                // Drop images that couldn't be fetched
                const validImages = fetchedImages.filter(i => i.url);
                console.log(`  ${validImages.length}/${rawImages.length} images successfully inlined`);

                const featured = validImages[0];
                const gridImages = validImages.slice(1, 5); // up to 4 for 2x2 grid

                if (featured || gridImages.length > 0) {
                    allPages.push({
                        is_imagery_page: true,
                        section_ref: "Imagery",
                        imagery_title: sanitizeText(data.title || "Featured Visual Coverage"),
                        imagery_desc: sanitizeText(data.description || "Key images drawn from Perplexity-retrieved Tier-1 editorial sources across the coverage window."),
                        featured_image: featured,
                        grid_images: gridImages.length > 0 ? gridImages : undefined,
                    });
                }
            }
        }

        // ══════════ AI & POLICY LEADERS (agency-style digest) ══════════
        // External AI figures and policy leaders tracked by name: Jensen Huang, Demis Hassabis,
        // Sam Altman, Dario Amodei, etc. Empty entities still render with "No notable mentions."
        {
            const data = input.ai_policy_leaders;
            if (data && Array.isArray(data.entities) && data.entities.length > 0) {
                const buckets = data.entities.map(e => ({
                    name: e.name,
                    items: e.mentions || [],
                }));
                const pages = buildDigestPages(
                    "AI & Policy Leaders",
                    "External AI & Policy Landscape",
                    sanitizeText(data.title || "AI & Policy Leaders"),
                    sanitizeText(data.description || "External AI figures, researchers, and policy leaders whose actions shape G42's competitive and regulatory landscape."),
                    buckets,
                    "No notable mentions.",
                    data.subnote ? sanitizeText(data.subnote) : null
                );
                allPages.push(...pages);
            }
        }

        // ══════════ PARTNERS & PEERS (agency-style digest) ══════════
        // OpenAI, Microsoft, NVIDIA, AMD, Amazon, xAI, Google, Anthropic, Meta,
        // and cross-company stories ("Microsoft and NVIDIA", "OpenAI and AWS").
        {
            const data = input.partners_peers;
            if (data && Array.isArray(data.entities) && data.entities.length > 0) {
                const buckets = data.entities.map(e => ({
                    name: e.name,
                    items: e.mentions || [],
                }));
                const pages = buildDigestPages(
                    "Partners & Peers",
                    "Strategic Partners & Peer Companies",
                    sanitizeText(data.title || "Partners & Peers"),
                    sanitizeText(data.description || "G42's ecosystem partners and peer companies. Cross-company buckets appear when a story involves two or more (e.g. 'Microsoft and NVIDIA')."),
                    buckets,
                    "No notable mentions.",
                    data.subnote ? sanitizeText(data.subnote) : null
                );
                allPages.push(...pages);
            }
        }

        // ══════════ INDUSTRY NEWS (agency-style digest) ══════════
        // Landscape coverage by category: Artificial Intelligence, ESG/Sustainability, AI in Sports, etc.
        {
            const data = input.industry_news;
            if (data && Array.isArray(data.categories) && data.categories.length > 0) {
                const buckets = data.categories.map(c => ({
                    name: c.name,
                    items: c.mentions || [],
                }));
                const pages = buildDigestPages(
                    "Industry News",
                    "Industry Landscape",
                    sanitizeText(data.title || "Industry News"),
                    sanitizeText(data.description || "AI industry landscape, ESG / sustainability signal, and sector-specific AI adoption — the operating environment context."),
                    buckets,
                    "No notable coverage.",
                    data.subnote ? sanitizeText(data.subnote) : null
                );
                allPages.push(...pages);
            }
        }

        // ══════════ MARKETS — REGIONAL (agency-style digest) ══════════
        // USA, UAE, US-China Relations, Europe, India, GCC / Saudi Arabia, Africa.
        {
            const data = input.markets;
            if (data && Array.isArray(data.regions) && data.regions.length > 0) {
                const buckets = data.regions.map(r => ({
                    name: r.name,
                    items: r.mentions || [],
                }));
                const pages = buildDigestPages(
                    "Markets",
                    "Regional & Geopolitical",
                    sanitizeText(data.title || "Markets"),
                    sanitizeText(data.description || "Regional coverage: USA, UAE, US-China Relations, Europe, India, GCC / Saudi Arabia, Africa. Geopolitics and market signals material to G42."),
                    buckets,
                    "No notable coverage.",
                    data.subnote ? sanitizeText(data.subnote) : null
                );
                allPages.push(...pages);
            }
        }

        // ══════════ SENTIMENT DEEP DIVE ══════════
        if (input.sentiment_analysis) {
            const sa = input.sentiment_analysis;
            let overall = null;
            if (sa.positive != null || sa.neutral != null || sa.negative != null) {
                overall = computeSentimentSummary(
                    Number(sa.positive || 0),
                    Number(sa.neutral || 0),
                    Number(sa.negative || 0)
                );
                overall.label = sanitizeText(sa.overall_label || "Overall");
            }

            const byPlatform = Array.isArray(sa.by_platform) ? sa.by_platform.map(p => {
                const s = computeSentimentSummary(
                    Number(p.positive || 0),
                    Number(p.neutral || 0),
                    Number(p.negative || 0)
                ) || { positive_pct: 0, neutral_pct: 0, negative_pct: 0 };
                return {
                    name: sanitizeText(p.name || ""),
                    positive_pct: s.positive_pct,
                    neutral_pct: s.neutral_pct,
                    negative_pct: s.negative_pct,
                    volume: String(p.volume || (Number(p.positive || 0) + Number(p.neutral || 0) + Number(p.negative || 0))),
                };
            }) : undefined;

            const byTopic = Array.isArray(sa.by_topic) ? sa.by_topic.map(p => {
                const s = computeSentimentSummary(
                    Number(p.positive || 0),
                    Number(p.neutral || 0),
                    Number(p.negative || 0)
                ) || { positive_pct: 0, neutral_pct: 0, negative_pct: 0 };
                return {
                    name: sanitizeText(p.name || ""),
                    positive_pct: s.positive_pct,
                    neutral_pct: s.neutral_pct,
                    negative_pct: s.negative_pct,
                    volume: String(p.volume || (Number(p.positive || 0) + Number(p.neutral || 0) + Number(p.negative || 0))),
                };
            }) : undefined;

            const trending = Array.isArray(sa.trending_topics) ? sa.trending_topics.map(t => ({
                topic: sanitizeText(t.topic || ""),
                sentiment: sanitizeText(t.sentiment || "Neutral"),
                sentiment_class: getSentimentClass(t.sentiment),
                volume: String(t.volume || "0"),
            })) : undefined;

            allPages.push({
                is_sentiment_page: true,
                section_ref: "Sentiment",
                section_title: sanitizeText(sa.title || "Sentiment Analysis"),
                section_desc: sanitizeText(sa.description || "Multi-platform sentiment breakdown by source and by topic. Scoring combines outlet tier, reach, and engagement velocity."),
                overall,
                by_platform: byPlatform,
                by_topic: byTopic,
                trending,
                analyst_note: sa.analyst_note ? sanitizeText(sa.analyst_note) : undefined,
            });
        }

        // ══════════ SOURCES ══════════
        if (Array.isArray(input.sources) && input.sources.length > 0) {
            const sources = input.sources.map((s, i) => ({
                num: i + 1,
                title: s.title ? sanitizeText(s.title) : undefined,
                text: sanitizeText(typeof s === 'string' ? s : s.text || ""),
                url: typeof s === 'object' && s.url ? String(s.url) : undefined,
            }));
            const PER_PAGE = 12;
            for (let i = 0; i < sources.length; i += PER_PAGE) {
                allPages.push({
                    is_sources_page: true,
                    section_ref: "References",
                    sources: sources.slice(i, i + PER_PAGE),
                    methodology: i + PER_PAGE >= sources.length && input.methodology ? sanitizeText(input.methodology) : undefined,
                });
            }
        }

        // Page numbers
        allPages.forEach((page, index) => {
            page.page_number = index + 1;
            page.total_pages = allPages.length;
        });

        // Context
        // Load the official G42 logo PNG from the repo, encode as data URI for inline rendering.
        // File lives next to the template at G42 Report/g42-logo.png (18 KB, 382×290, sourced from g42.ai).
        let g42LogoDataUri = "";
        try {
            const logoBytes = fs.readFileSync(path.join(__dirname, 'G42 Report/g42-logo.png'));
            g42LogoDataUri = `data:image/png;base64,${logoBytes.toString('base64')}`;
        } catch (e) {
            console.warn("G42 logo not found at G42 Report/g42-logo.png — falling back to text wordmark");
        }

        const context = {
            REPORT_TITLE: reportTitle,
            REPORT_SUBTITLE: reportSubtitle,
            REPORT_DATE: reportDate,
            REPORT_DATE_LONG: reportDateLong,
            REPORT_ID: reportId,
            CLASSIFICATION: classification,
            COVERAGE_WINDOW: coverageWindow,
            PUBLISHED_AT: publishedAt,
            ANALYST: analyst,
            ORG_NAME: orgName,
            G42_LOGO: g42LogoDataUri,
            PAGES: allPages,
        };

        // G42 logo is permanently embedded in the template as inline SVG.
        // No file dependency, no override path — guaranteed to render on every page.

        // Render
        const compiledTemplate = Handlebars.compile(fs.readFileSync(TEMPLATE_PATH, 'utf8'));
        const html = compiledTemplate(context);
        // Debug: dump rendered HTML alongside PDF when G42_DEBUG_HTML=1
        if (process.env.G42_DEBUG_HTML) {
            fs.writeFileSync(path.join(PDF_STORAGE_DIR, `_last_render.html`), html);
        }
        const pdfBytes = await renderDynamicPdf(html);

        // Upload to Azure Blob Storage (15-minute SAS URL) — permanent persistence
        const azureResult = await uploadPdfToAzure(pdfBytes);

        let url = azureResult.url;
        // Local-dev fallback: build the local download URL
        if (azureResult.local) {
            const protocol = req.headers['x-forwarded-proto'] || req.protocol || 'http';
            const host = req.headers['x-forwarded-host'] || req.headers['host'];
            const basePath = process.env.BASE_PATH || '';
            url = `${protocol}://${host}${basePath}/g42-report/download/${azureResult.pdfId}`;
        }

        sessionStore.delete(sessionId);

        console.log(`G42 report generated: ${azureResult.pdfId} — ${allPages.length} pages — ${azureResult.local ? 'local-dev' : 'azure blob'}`);
        res.json({
            message: "G42 intelligence report generated",
            url,
            pages: allPages.length,
            pdfId: azureResult.pdfId,
            permanent: !azureResult.local,
            urlExpiresInMinutes: azureResult.local ? null : 360,
            storageMode: azureResult.local ? "local-dev" : "azure-blob",
        });

    } catch (e) {
        console.error("G42 PDF Generation Error:", e);
        res.status(500).json({ error: "Failed to generate PDF: " + e.message });
    }
});

// OnDemand serverless sets PORT; fall back to G42_PORT or 3001 for local dev.
const PORT = process.env.PORT || process.env.G42_PORT || 3001;
app.listen(PORT, () => {
    console.log(`G42 Daily Media Intelligence Server running on port ${PORT}`);
    console.log(`PDF storage (permanent): ${PDF_STORAGE_DIR}`);
});

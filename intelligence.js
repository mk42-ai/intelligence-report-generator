require("dotenv").config();
const express = require("express");
const path = require("path");
const Handlebars = require("handlebars");
const bodyParser = require("body-parser");
const app = express();
const puppeteer = require("puppeteer");
const { PDFDocument } = require("pdf-lib");
const fs = require("fs");

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json({ limit: "50mb" }));

// ---------------------------------------------------------
// PDF STORE & HELPERS
// ---------------------------------------------------------
const pdfStore = new Map();
const PDF_TTL = 60 * 60 * 1000;

setInterval(() => {
    const now = Date.now();
    for (const [id, entry] of pdfStore) {
        if (now - entry.createdAt > PDF_TTL) pdfStore.delete(id);
    }
}, 10 * 60 * 1000);

function getRandomString(length = 10) {
    const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let result = "";
    for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return result;
}

app.get('/intelligence-report/download/:id', (req, res) => {
    const entry = pdfStore.get(req.params.id);
    if (!entry) return res.status(404).json({ error: "PDF not found or expired" });
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `inline; filename="intelligence-report-${req.params.id}.pdf"`);
    res.send(entry.buffer);
});

// ---------------------------------------------------------
// CONFIGURATION & HELPERS
// ---------------------------------------------------------
const TEMPLATE_PATH = path.join(__dirname, 'Intelligence Report/template.html');

Handlebars.registerHelper('safe', text => new Handlebars.SafeString(text));
Handlebars.registerHelper('eq', (a, b) => a === b);
Handlebars.registerHelper('gt', (a, b) => Number(a) > Number(b));
Handlebars.registerHelper('add', (a, b) => Number(a) + Number(b));
Handlebars.registerHelper('json', obj => JSON.stringify(obj));

async function getImageDataUrl(filePath) {
    try {
        const data = fs.readFileSync(filePath);
        const ext = path.extname(filePath).toLowerCase().replace('.', '');
        const mimeMap = { svg: 'image/svg+xml', png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp' };
        const mime = mimeMap[ext] || `image/${ext}`;
        return `data:${mime};base64,${data.toString('base64')}`;
    } catch {
        return "";
    }
}

function getThreatBadgeHtml(level) {
    const l = (level || "moderate").toLowerCase();
    let cls = "moderate";
    if (l.includes("critical")) cls = "critical";
    else if (l.includes("high")) cls = "high";
    else if (l.includes("low")) cls = "low";
    return `<span class="threat-badge ${cls}">${level}</span>`;
}

function getConfidenceBadgeHtml(confidence) {
    const c = (confidence || "moderate").toLowerCase();
    let cls = "moderate-conf";
    if (c.includes("high")) cls = "high-conf";
    else if (c.includes("low")) cls = "low-conf";
    return `<span class="conf-badge ${cls}">${confidence || "Moderate"}</span>`;
}

function getEscalationHtml(value) {
    if (!value) return "";
    const v = String(value);
    const num = parseInt(v);
    let color = "var(--moderate)";
    if (num >= 70) color = "var(--critical)";
    else if (num >= 40) color = "var(--high)";
    else if (num < 20) color = "var(--low)";
    return `<span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:${color};">${v}</span>`;
}

// Sanitize text: strip problematic Unicode chars that cause (cid:0) rendering
function sanitizeText(text) {
    if (!text) return "";
    return String(text)
        // Remove zero-width and invisible chars
        .replace(/[\u200B-\u200F\u2028-\u202F\uFEFF\u00AD]/g, '')
        // Remove private use area chars
        .replace(/[\uE000-\uF8FF]/g, '')
        // Remove surrogate pairs that aren't valid
        .replace(/[\uD800-\uDFFF]/g, '')
        // Replace em/en dashes with ASCII
        .replace(/[\u2013\u2014]/g, '-')
        // Replace smart quotes
        .replace(/[\u2018\u2019]/g, "'")
        .replace(/[\u201C\u201D]/g, '"')
        // Replace bullets
        .replace(/[\u2022\u2023\u25E6\u2043]/g, '•')
        // Replace ellipsis
        .replace(/\u2026/g, '...')
        // Remove other control chars (except newlines/tabs)
        .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
}

function getClassificationClass(classification) {
    const c = (classification || "").toLowerCase().replace(/[^a-z]/g, '');
    if (c.includes("topsecret")) return "top-secret";
    if (c.includes("secret")) return "secret";
    if (c.includes("confidential")) return "confidential";
    if (c.includes("fouo") || c.includes("official") || c.includes("sensitive")) return "fouo";
    return "unclassified";
}

function getStatClass(type) {
    const t = (type || "").toLowerCase();
    if (t.includes("critical")) return "critical-stat";
    if (t.includes("high")) return "high-stat";
    if (t.includes("moderate")) return "moderate-stat";
    if (t.includes("low")) return "low-stat";
    return "accent-stat";
}

function getSentimentClass(sentiment) {
    const s = (sentiment || "").toLowerCase();
    if (s.includes("positive") || s.includes("pos")) return "pos";
    if (s.includes("negative") || s.includes("neg")) return "neg";
    return "neu";
}

function getSentimentOverallClass(sentiment) {
    const s = (sentiment || "").toLowerCase();
    if (s.includes("positive")) return "positive";
    if (s.includes("negative")) return "negative";
    return "neutral";
}

function getExplosionStatusClass(status) {
    const s = (status || "").toLowerCase();
    if (s.includes("confirmed") && !s.includes("un")) return "confirmed";
    return "unconfirmed";
}

function getBarColorClass(color) {
    const c = (color || "accent").toLowerCase();
    if (c.includes("critical") || c.includes("red")) return "critical";
    if (c.includes("high") || c.includes("orange")) return "high";
    if (c.includes("moderate") || c.includes("yellow")) return "moderate";
    if (c.includes("low") || c.includes("green")) return "low";
    if (c.includes("olive")) return "olive";
    return "accent";
}

function processCharts(charts) {
    if (!Array.isArray(charts) || charts.length === 0) return undefined;
    return charts.map(chart => {
        const bars = (chart.bars || []);
        // For percentage-based values, use percent directly. Otherwise calculate from value.
        const hasPercent = bars.some(b => b.percent != null);
        const maxVal = hasPercent ? 100 : Math.max(...bars.map(b => parseFloat(String(b.value).replace(/[^0-9.]/g, '')) || 0), 1);
        return {
            title: String(chart.title || ""),
            bars: bars.map(bar => {
                const pct = bar.percent != null ? Number(bar.percent) : Math.round((parseFloat(String(bar.value).replace(/[^0-9.]/g, '')) || 0) / maxVal * 100);
                return {
                    label: String(bar.label || ""),
                    value: String(bar.value || "0"),
                    percent: Math.min(pct, 100),
                    color: getBarColorClass(bar.color)
                };
            })
        };
    });
}

// ---------------------------------------------------------
// PUPPETEER RENDER
// ---------------------------------------------------------
// A4 at 96 DPI: 794 × 1123 px
const PAGE_WIDTH = 794;
const PAGE_HEIGHT = 1123;

async function renderDynamicPdf(html) {
    const browser = await puppeteer.launch({
        headless: "new",
        args: ["--no-sandbox", "--disable-setuid-sandbox"]
    });

    try {
        const page = await browser.newPage();
        await page.setViewport({ width: PAGE_WIDTH, height: PAGE_HEIGHT });
        await page.setContent(html, { waitUntil: "networkidle0", timeout: 60000 });

        // Inject fixed-page CSS: each .page-wrapper = exactly one A4 page,
        // overflow hidden so content clips rather than creating mega-tall pages.
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
                padding: 76px 28px 44px;
                margin: 0;
            }
            .page-wrapper:last-child { page-break-after: avoid; }
            .classification-banner {
                position: fixed; top: 0; left: 0; right: 0; height: 24px; z-index: 2000;
            }
            .classification-banner-bottom {
                position: fixed; bottom: 0; left: 0; right: 0; height: 20px; z-index: 2000;
            }
            .page-header-bar {
                position: fixed; top: 24px; left: 0; right: 0; height: 48px; z-index: 1500;
            }
        `});

        // Use Puppeteer's native page-break-aware PDF rendering
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
// SESSION STORE
// ---------------------------------------------------------
const sessionStore = new Map();

// Auto-cleanup sessions older than 2 hours
setInterval(() => {
    const now = Date.now();
    for (const [id, session] of sessionStore) {
        if (now - session.timestamp > 2 * 60 * 60 * 1000) {
            sessionStore.delete(id);
            console.log(`Session expired and removed: ${id}`);
        }
    }
}, 10 * 60 * 1000);

// ---------------------------------------------------------
// API ENDPOINTS
// ---------------------------------------------------------

/**
 * START SESSION — creates or resets a session
 */
app.post('/intelligence-report/start', (req, res) => {
    let { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });

    sessionId = sessionId.toLowerCase().trim();

    // If session already exists, reset it instead of erroring
    if (sessionStore.has(sessionId)) {
        sessionStore.set(sessionId, { timestamp: Date.now(), data: {} });
        console.log(`Session reset: ${sessionId}`);
        return res.json({ message: "Session reset (previous data cleared)", sessionId });
    }

    sessionStore.set(sessionId, { timestamp: Date.now(), data: {} });
    console.log(`Session started: ${sessionId}`);
    res.json({ message: "Session started", sessionId });
});

/**
 * DELETE SESSION
 */
app.post('/intelligence-report/delete', (req, res) => {
    let { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });

    sessionId = sessionId.toLowerCase().trim();
    if (sessionStore.has(sessionId)) {
        sessionStore.delete(sessionId);
        console.log(`Session deleted: ${sessionId}`);
        return res.json({ message: "Session deleted", sessionId });
    }
    res.json({ message: "Session not found (already clean)", sessionId });
});

/**
 * LIST SESSIONS — debug helper
 */
app.get('/intelligence-report/sessions', (req, res) => {
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

/**
 * UPDATE DATA — deep merges data into session
 */
app.post('/intelligence-report/update', (req, res) => {
    const body = req.body;
    if (!body.sessionId) return res.status(400).json({ error: "sessionId required" });

    const sessionId = body.sessionId.toLowerCase().trim();
    if (!sessionStore.has(sessionId)) return res.status(404).json({ error: "Session not found. Call /start first." });

    const session = sessionStore.get(sessionId);
    session.timestamp = Date.now();

    // Deep merge: arrays get replaced, objects get merged
    const data = session.data;
    for (const [key, value] of Object.entries(body)) {
        if (key === 'sessionId') continue;

        if (Array.isArray(value)) {
            // Arrays: append if same key exists and is array, otherwise replace
            if (Array.isArray(data[key])) {
                data[key] = [...data[key], ...value];
            } else {
                data[key] = value;
            }
        } else if (value && typeof value === 'object' && !Array.isArray(value)) {
            // Objects: deep merge
            if (data[key] && typeof data[key] === 'object' && !Array.isArray(data[key])) {
                data[key] = { ...data[key], ...value };
            } else {
                data[key] = value;
            }
        } else {
            data[key] = value;
        }
    }

    const updatedKeys = Object.keys(body).filter(k => k !== 'sessionId');
    console.log(`Session updated: ${sessionId} — keys: ${updatedKeys.join(', ')}`);
    res.json({ message: "Data updated", updatedKeys, totalKeys: Object.keys(data).length });
});

/**
 * GENERATE PDF
 */
app.post('/intelligence-report/generate', async (req, res) => {
    let { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });

    sessionId = sessionId.toLowerCase().trim();
    if (!sessionStore.has(sessionId)) return res.status(404).json({ error: "Session not found" });

    try {
        const session = sessionStore.get(sessionId);
        const input = session.data || {};

        // ────────────────────────────────────
        // SAFE DEFAULTS
        // ────────────────────────────────────
        const reportTitle = sanitizeText(input.report_title || "Intelligence Assessment");
        const reportSubtitle = sanitizeText(input.report_subtitle || "Predictive Threat Assessment");
        const reportDate = sanitizeText(input.report_date || new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }));
        const reportId = sanitizeText(input.report_id || `IR-${Date.now().toString(36).toUpperCase()}`);
        const classification = sanitizeText(input.classification || "UNCLASSIFIED");
        const classificationClass = getClassificationClass(classification);
        const updateCadence = sanitizeText(input.update_cadence || "As Required");
        const orgName = sanitizeText(input.org_name || "BASEERA INTELLIGENCE");

        // ────────────────────────────────────
        // BUILD PAGES
        // ────────────────────────────────────
        const compiledTemplate = Handlebars.compile(fs.readFileSync(TEMPLATE_PATH, 'utf8'));
        let allPages = [];

        // ── PAGE 1: COVER PAGE ──
        const coverStats = Array.isArray(input.key_stats) ? input.key_stats.map(s => ({
            value: String(s.value || "\u2014"),
            label: String(s.label || ""),
            stat_class: getStatClass(s.type)
        })) : [];

        const coverPage = {
            is_cover_page: true,
            section_ref: "Cover",
            executive_summary: sanitizeText(input.executive_summary || ""),
            stats: coverStats.length > 0 ? coverStats : undefined,
            charts: processCharts(input.cover_charts),
        };

        if (input.cover_image) {
            coverPage.cover_image = {
                url: String(input.cover_image.url || ""),
                caption: String(input.cover_image.caption || ""),
                source: String(input.cover_image.source || "")
            };
        }

        allPages.push(coverPage);

        // ── PAGE 2: THREAT MATRIX ──
        if (Array.isArray(input.threat_matrix) && input.threat_matrix.length > 0) {
            const threatRows = input.threat_matrix.map(t => ({
                domain: sanitizeText(t.domain),
                threat_badge_html: getThreatBadgeHtml(String(t.threat_level || "Moderate")),
                vector: sanitizeText(t.vector),
                status: sanitizeText(t.status),
                concerns: sanitizeText(t.concerns)
            }));

            const keyIndicators = Array.isArray(input.key_indicators) ? input.key_indicators.map(ind => ({
                text: sanitizeText(ind.text || ind),
                indicator_class: (ind.level || "").toLowerCase().includes("critical") ? "critical-ind" :
                                 (ind.level || "").toLowerCase().includes("high") ? "high-ind" : ""
            })) : [];

            allPages.push({
                is_threat_matrix_page: true,
                section_ref: "Threat Assessment",
                section_title: sanitizeText(input.threat_matrix_title || "Multi-Domain Threat Matrix"),
                section_desc: sanitizeText(input.threat_matrix_desc || "Comprehensive threat assessment across all operational domains with current status and key concerns."),
                threat_rows: threatRows,
                key_indicators: keyIndicators.length > 0 ? keyIndicators : undefined,
                indicator_horizon: String(input.indicator_horizon || "72h"),
                charts: processCharts(input.threat_charts),
            });
        }

        // ── PAGES 3-N: ANALYSIS SECTIONS (each gets its own page, split if too much content) ──
        const analysisSections = Array.isArray(input.analysis_sections) ? input.analysis_sections : [];
        analysisSections.forEach(section => {
            const sectionRef = sanitizeText(section.section_ref || section.eyebrow || "Analysis");
            const sectionEyebrow = sanitizeText(section.eyebrow || "Domain Analysis");
            const sectionTitle = sanitizeText(section.title || "Analysis");
            const sectionDesc = sanitizeText(section.description || "");

            const allImages = Array.isArray(section.images) ? section.images.map(img => ({
                url: String(img.url || ""),
                caption: sanitizeText(img.caption),
                meta: sanitizeText(img.meta),
                type: sanitizeText(img.type)
            })) : [];

            const allVulnCards = Array.isArray(section.vulnerability_cards) ? section.vulnerability_cards.map(vc => ({
                title: sanitizeText(vc.title),
                threat_badge_html: getThreatBadgeHtml(String(vc.threat_level || "Moderate")),
                image_url: vc.image_url ? String(vc.image_url) : undefined,
                details: sanitizeText(vc.details)
            })) : [];

            // Determine if we need to split: if lots of content (images + vuln cards + text)
            const hasText = !!section.text;
            const hasTable = !!section.table;
            const needsSplit = (allImages.length > 2) || (allVulnCards.length > 2) ||
                              (allImages.length > 0 && allVulnCards.length > 0 && hasText);

            // PAGE 1: Main analysis content (text, table, first 2 images, first 2 vuln cards)
            const page = {
                is_analysis_page: true,
                section_ref: sectionRef,
                section_eyebrow: sectionEyebrow,
                section_title: sectionTitle,
                section_desc: sectionDesc,
            };

            if (section.assessment_confidence) {
                page.assessment_confidence = true;
                page.assessment_confidence_html = getConfidenceBadgeHtml(String(section.assessment_confidence));
            }

            if (hasTable) {
                page.analysis_table = {
                    headers: Array.isArray(section.table.headers) ? section.table.headers.map(String) : [],
                    rows: Array.isArray(section.table.rows) ? section.table.rows.map(row => ({
                        cells: Array.isArray(row.cells) ? row.cells.map(cell => sanitizeText(cell)) : Array.isArray(row) ? row.map(s => sanitizeText(s)) : []
                    })) : []
                };
            }

            if (hasText) {
                let text = sanitizeText(section.text);
                // If page has images or vuln cards, limit text to prevent overflow
                if (allImages.length > 0 || allVulnCards.length > 0) {
                    text = text.length > 800 ? text.substring(0, 800) + '...' : text;
                }
                page.analysis_text = text;
            }

            // First batch of images (max 2) and vuln cards (max 2)
            if (allImages.length > 0) {
                page.images = needsSplit ? allImages.slice(0, 2) : allImages;
            }
            if (allVulnCards.length > 0) {
                page.vulnerability_cards = needsSplit ? allVulnCards.slice(0, 2) : allVulnCards;
            }

            if (!needsSplit && section.analyst_assessment) {
                page.analyst_assessment = sanitizeText(section.analyst_assessment);
            }
            if (!needsSplit) {
                page.charts = processCharts(section.charts);
            }

            allPages.push(page);

            // PAGE 2 (if split): Remaining images, vuln cards, analyst assessment, charts
            if (needsSplit) {
                const remainingImages = allImages.slice(2);
                const remainingVulnCards = allVulnCards.slice(2);

                if (remainingImages.length > 0 || remainingVulnCards.length > 0 || section.analyst_assessment || section.charts) {
                    const page2 = {
                        is_analysis_continued_page: true,
                        section_ref: sectionRef,
                        section_eyebrow: sectionEyebrow,
                        section_title: sectionTitle,
                    };

                    if (remainingImages.length > 0) {
                        page2.images = remainingImages.slice(0, 4); // max 4 images per continuation
                    }
                    if (remainingVulnCards.length > 0) {
                        page2.vulnerability_cards = remainingVulnCards.slice(0, 4);
                    }
                    if (section.analyst_assessment) {
                        page2.analyst_assessment = sanitizeText(section.analyst_assessment);
                    }
                    page2.charts = processCharts(section.charts);

                    allPages.push(page2);
                }
            }
        });

        // ── X / TWITTER SENTIMENT PAGE (split into 2 pages) ──
        if (input.x_sentiment) {
            const xs = input.x_sentiment;
            const xTitle = sanitizeText(xs.title || "X (Twitter) Sentiment Analysis");

            // PAGE 1: Stats + keyword sentiment + trending hashtags
            const xPage1 = {
                is_x_sentiment_page: true,
                section_ref: "X/Social Sentiment",
                section_title: xTitle,
                section_desc: sanitizeText(xs.description || "Real-time sentiment analysis from X/Twitter posts related to the crisis."),
                overall_sentiment: String(xs.overall_sentiment || "Neutral"),
                overall_sentiment_class: getSentimentOverallClass(xs.overall_sentiment),
                overall_score: String(xs.overall_score || "0.00"),
                total_posts: String(xs.total_posts || "0"),
                time_period: String(xs.time_period || "24h"),
            };

            // Sentiment breakdown by keyword/hashtag (max 5 to fit page)
            if (Array.isArray(xs.keyword_sentiment) && xs.keyword_sentiment.length > 0) {
                xPage1.keyword_sentiment = xs.keyword_sentiment.slice(0, 5).map(kw => ({
                    keyword: sanitizeText(kw.keyword),
                    positive: Number(kw.positive) || 0,
                    neutral: Number(kw.neutral) || 0,
                    negative: Number(kw.negative) || 0,
                    volume: String(kw.volume || "0"),
                }));
            }

            // Trending hashtags (max 6 to fit page)
            if (Array.isArray(xs.trending_hashtags) && xs.trending_hashtags.length > 0) {
                xPage1.trending_hashtags = xs.trending_hashtags.slice(0, 6).map(ht => ({
                    hashtag: sanitizeText(ht.hashtag),
                    volume: String(ht.volume || "0"),
                    sentiment: String(ht.sentiment || "Neutral"),
                    sentiment_class: getSentimentClass(ht.sentiment),
                    change: String(ht.change || ""),
                }));
            }

            allPages.push(xPage1);

            // PAGE 2: Evidence tweets + analyst assessment + charts
            const hasTweets = Array.isArray(xs.evidence_tweets) && xs.evidence_tweets.length > 0;
            const hasAssessment = !!xs.analyst_assessment;
            const hasCharts = Array.isArray(xs.charts) && xs.charts.length > 0;

            if (hasTweets || hasAssessment || hasCharts) {
                const xPage2 = {
                    is_x_evidence_page: true,
                    section_ref: "X/Social Sentiment",
                    section_title: xTitle,
                    charts: processCharts(xs.charts),
                };

                if (hasTweets) {
                    xPage2.evidence_tweets = xs.evidence_tweets.slice(0, 3).map(tw => ({
                        username: sanitizeText(tw.username || "@unknown"),
                        handle: sanitizeText(tw.handle),
                        timestamp: sanitizeText(tw.timestamp),
                        text: sanitizeText(tw.text),
                        sentiment: String(tw.sentiment || "Neutral"),
                        sentiment_class: getSentimentClass(tw.sentiment),
                        retweets: String(tw.retweets || "0"),
                        likes: String(tw.likes || "0"),
                        verified: tw.verified === true,
                    }));
                }

                if (hasAssessment) {
                    xPage2.analyst_assessment = sanitizeText(xs.analyst_assessment);
                }

                allPages.push(xPage2);
            }
        }

        // ── GENERAL SENTIMENT PAGE ──
        if (input.sentiment_analysis) {
            const sa = input.sentiment_analysis;

            const sentimentPage = {
                is_sentiment_page: true,
                section_ref: "Sentiment Analysis",
                section_title: sanitizeText(sa.title || "Multi-Source Sentiment Intelligence"),
                section_desc: sanitizeText(sa.description || "Aggregated sentiment analysis across social media, state media, diplomatic channels, and financial markets."),
                overall_sentiment: String(sa.overall_sentiment || "Neutral"),
                overall_sentiment_class: getSentimentOverallClass(sa.overall_sentiment),
                overall_score: String(sa.overall_score || "0.00"),
                sources_analyzed: String(sa.sources_analyzed || "0"),
                time_period: String(sa.time_period || "N/A"),
                charts: processCharts(sa.charts),
            };

            if (Array.isArray(sa.categories) && sa.categories.length > 0) {
                sentimentPage.categories = sa.categories.slice(0, 6).map(cat => ({
                    name: sanitizeText(cat.name),
                    positive: Number(cat.positive) || 0,
                    neutral: Number(cat.neutral) || 0,
                    negative: Number(cat.negative) || 0,
                    key_narrative: cat.key_narrative ? sanitizeText(cat.key_narrative) : undefined,
                }));
            }

            if (Array.isArray(sa.trending_topics) && sa.trending_topics.length > 0) {
                sentimentPage.trending_topics = sa.trending_topics.slice(0, 6).map(tt => ({
                    topic: sanitizeText(tt.topic),
                    sentiment: String(tt.sentiment || "Neutral"),
                    sentiment_class: getSentimentClass(tt.sentiment),
                    volume: String(tt.volume || "0"),
                }));
            }

            if (sa.analyst_assessment) {
                sentimentPage.analyst_assessment = sanitizeText(sa.analyst_assessment);
            }

            allPages.push(sentimentPage);
        }

        // ── FIRE / THERMAL DETECTION PAGE (NASA FIRMS) — split if many detections ──
        if (input.fire_detection) {
            const fd = input.fire_detection;
            const summary = fd.summary || {};
            const fireTitle = sanitizeText(fd.title || "NASA FIRMS Thermal Anomaly Detection");

            const allDetections = Array.isArray(fd.detections) ? fd.detections.map(det => ({
                lat: String(det.lat || det.latitude || ""),
                lon: String(det.lon || det.longitude || ""),
                location_name: sanitizeText(det.location_name || det.location || "Unknown"),
                bright_ti4: String(det.bright_ti4 || det.brightness || ""),
                frp: String(det.frp || ""),
                confidence: String(det.confidence || ""),
                acq_date: String(det.acq_date || det.date || ""),
                acq_time: String(det.acq_time || det.time || ""),
                classification: String(det.classification || "Standard"),
                classification_class: (det.classification || "").toLowerCase().includes("military") ? "military" :
                                      (det.classification || "").toLowerCase().includes("significant") ? "significant" : "standard",
                details: det.details ? sanitizeText(det.details) : undefined,
            })) : [];

            const hasAreaSummary = Array.isArray(fd.area_summary) && fd.area_summary.length > 0;
            // With area summary table, fit fewer detections on page 1
            const maxDetectionsPage1 = hasAreaSummary ? 5 : 8;
            const needsSplit = allDetections.length > maxDetectionsPage1;

            const firePage = {
                is_fire_page: true,
                section_ref: "Fire Detection",
                section_title: fireTitle,
                section_desc: sanitizeText(fd.description || "VIIRS satellite infrared detections cross-referenced with known target coordinates."),
                summary_total: String(summary.total_detections || "0"),
                summary_significant: String(summary.significant || "0"),
                summary_military: String(summary.military_grade || "0"),
                summary_timespan: String(summary.timespan || "N/A"),
            };

            if (allDetections.length > 0) {
                firePage.detections = needsSplit ? allDetections.slice(0, maxDetectionsPage1) : allDetections;
            }

            if (hasAreaSummary) {
                firePage.area_summary = fd.area_summary.map(a => ({
                    area: sanitizeText(a.area),
                    count: String(a.count || "0"),
                    max_frp: String(a.max_frp || "0"),
                    assessment: sanitizeText(a.assessment),
                }));
            }

            if (!needsSplit) {
                if (fd.analyst_assessment) firePage.analyst_assessment = sanitizeText(fd.analyst_assessment);
                firePage.charts = processCharts(fd.charts);
            }

            allPages.push(firePage);

            // Continuation page(s) for remaining detections
            if (needsSplit) {
                const remaining = allDetections.slice(maxDetectionsPage1);
                const MAX_PER_CONT = 12; // more room on continuation pages (no summary cards)
                for (let i = 0; i < remaining.length; i += MAX_PER_CONT) {
                    const chunk = remaining.slice(i, i + MAX_PER_CONT);
                    const isLastChunk = (i + MAX_PER_CONT >= remaining.length);

                    const contPage = {
                        is_fire_continued_page: true,
                        section_ref: "Fire Detection",
                        section_title: fireTitle,
                        detections: chunk,
                    };

                    if (isLastChunk) {
                        if (fd.analyst_assessment) contPage.analyst_assessment = sanitizeText(fd.analyst_assessment);
                        contPage.charts = processCharts(fd.charts);
                    }

                    allPages.push(contPage);
                }
            }
        }

        // ── EXPLOSION DETECTION PAGE — split if many events ──
        if (input.explosion_detection) {
            const ed = input.explosion_detection;
            const summary = ed.summary || {};
            const explosionTitle = sanitizeText(ed.title || "Explosion / Blast Event Detection");

            const allEvents = Array.isArray(ed.events) ? ed.events.map(evt => {
                const event = {
                    event_id: sanitizeText(evt.event_id || "EXP-???"),
                    timestamp: sanitizeText(evt.timestamp),
                    location: sanitizeText(evt.location || evt.location_name),
                    magnitude: sanitizeText(evt.magnitude || "Unknown"),
                    confidence: String(evt.confidence != null ? (typeof evt.confidence === 'number' && evt.confidence <= 1 ? (evt.confidence * 100).toFixed(0) + '%' : evt.confidence) : "N/A"),
                    blast_radius: evt.blast_radius_m ? String(evt.blast_radius_m) + 'm' : (evt.blast_radius ? String(evt.blast_radius) : undefined),
                    source: evt.source ? sanitizeText(evt.source) : undefined,
                    status: String(evt.status || "Unconfirmed"),
                    status_class: getExplosionStatusClass(evt.status),
                    details: evt.details ? sanitizeText(evt.details) : undefined,
                };

                if (Array.isArray(evt.images) && evt.images.length > 0) {
                    event.images = evt.images.slice(0, 2).map(img => ({
                        url: String(img.url || ""),
                        caption: sanitizeText(img.caption),
                    }));
                }

                return event;
            }) : [];

            // Events with images are tall (~250px each), without images ~120px. Max 2 with images on page 1.
            const hasImgEvents = allEvents.some(e => e.images && e.images.length > 0);
            const maxEventsPage1 = hasImgEvents ? 2 : 3;
            const needsSplit = allEvents.length > maxEventsPage1;

            const explosionPage = {
                is_explosion_page: true,
                section_ref: "Explosion Detection",
                section_title: explosionTitle,
                section_desc: sanitizeText(ed.description || "Monitored blast and explosion events detected via satellite IR, seismic, and acoustic sensors."),
                summary_total: String(summary.total_events || "0"),
                summary_confirmed: String(summary.confirmed || "0"),
                summary_unconfirmed: String(summary.unconfirmed || "0"),
                summary_timespan: String(summary.timespan || summary.time_span || "N/A"),
                events: needsSplit ? allEvents.slice(0, maxEventsPage1) : allEvents,
            };

            if (!needsSplit) {
                if (ed.analyst_assessment) explosionPage.analyst_assessment = sanitizeText(ed.analyst_assessment);
                explosionPage.charts = processCharts(ed.charts);
            }

            allPages.push(explosionPage);

            // Continuation pages for remaining events
            if (needsSplit) {
                const remaining = allEvents.slice(maxEventsPage1);
                const MAX_PER_CONT = hasImgEvents ? 3 : 4;
                for (let i = 0; i < remaining.length; i += MAX_PER_CONT) {
                    const chunk = remaining.slice(i, i + MAX_PER_CONT);
                    const isLastChunk = (i + MAX_PER_CONT >= remaining.length);

                    const contPage = {
                        is_explosion_continued_page: true,
                        section_ref: "Explosion Detection",
                        section_title: explosionTitle,
                        events: chunk,
                    };

                    if (isLastChunk) {
                        if (ed.analyst_assessment) contPage.analyst_assessment = sanitizeText(ed.analyst_assessment);
                        contPage.charts = processCharts(ed.charts);
                    }

                    allPages.push(contPage);
                }
            }
        }

        // ── SCENARIO / OUTLOOK PAGE ──
        if (Array.isArray(input.scenarios) && input.scenarios.length > 0) {
            const scenarioPage = {
                is_scenario_page: true,
                section_ref: "Predictive Outlook",
                section_title: sanitizeText(input.scenario_title || "Predictive Outlook & Scenario Analysis"),
                section_desc: sanitizeText(input.scenario_desc || "Forward-looking assessment of likely developments, decision triggers, and inflection points."),
                scenarios: input.scenarios.map(s => ({
                    horizon: sanitizeText(s.horizon),
                    most_likely: sanitizeText(s.most_likely),
                    escalation_html: getEscalationHtml(s.escalation_risk),
                    triggers: sanitizeText(s.triggers)
                })),
                charts: processCharts(input.scenario_charts),
            };

            if (Array.isArray(input.escalation_indicators)) {
                scenarioPage.escalation_indicators = input.escalation_indicators.map(String);
            }
            if (Array.isArray(input.deescalation_indicators)) {
                scenarioPage.deescalation_indicators = input.deescalation_indicators.map(String);
            }

            allPages.push(scenarioPage);
        }

        // ── PRIORITY ACTIONS PAGE ──
        if (Array.isArray(input.priority_actions) && input.priority_actions.length > 0) {
            const actionsPage = {
                is_actions_page: true,
                section_ref: "Priority Actions",
                section_title: sanitizeText(input.actions_title || "Priority Actions & Recommendations"),
                section_desc: sanitizeText(input.actions_desc || "Immediate and near-term actions recommended based on current intelligence assessment."),
                actions: input.priority_actions.map((a, i) => ({
                    num: i + 1,
                    title: sanitizeText(a.title),
                    description: sanitizeText(a.description),
                    priority: a.priority ? String(a.priority) : undefined,
                    priority_class: (a.priority || "").toLowerCase().includes("critical") ? "critical" :
                                    (a.priority || "").toLowerCase().includes("high") ? "high" : "moderate",
                }))
            };

            if (input.highest_yield_action) {
                actionsPage.highest_yield_action = sanitizeText(input.highest_yield_action);
            }

            if (Array.isArray(input.data_quality) && input.data_quality.length > 0) {
                actionsPage.data_quality = input.data_quality.map(d => ({
                    domain: sanitizeText(d.domain),
                    confidence_html: getConfidenceBadgeHtml(String(d.confidence || "Moderate")),
                    notes: sanitizeText(d.notes)
                }));
            }

            allPages.push(actionsPage);
        }

        // ── SOURCES PAGE ──
        if (Array.isArray(input.sources) && input.sources.length > 0) {
            allPages.push({
                is_sources_page: true,
                section_ref: "References",
                sources: input.sources.map((s, i) => ({
                    num: i + 1,
                    text: sanitizeText(typeof s === 'string' ? s : s.text || ""),
                    url: typeof s === 'object' && s.url ? String(s.url) : undefined,
                })),
                deliverable_note: input.deliverable_note ? sanitizeText(input.deliverable_note) : undefined
            });
        }

        // ── ADDITIONAL / APPENDIX PAGES ──
        if (Array.isArray(input.additional_pages)) {
            input.additional_pages.forEach(pg => {
                const page = {
                    is_content_page: true,
                    section_ref: String(pg.section_ref || "Appendix"),
                    section_eyebrow: String(pg.eyebrow || "Additional Analysis"),
                    section_title: String(pg.title || ""),
                    section_desc: pg.description ? String(pg.description) : undefined,
                    content_html: String(pg.content_html || ""),
                    charts: processCharts(pg.charts),
                };

                if (Array.isArray(pg.images) && pg.images.length > 0) {
                    page.images = pg.images.map(img => ({
                        url: String(img.url || ""),
                        caption: String(img.caption || ""),
                        meta: String(img.meta || ""),
                        type: String(img.type || "")
                    }));
                }

                if (pg.analyst_assessment) {
                    page.analyst_assessment = String(pg.analyst_assessment);
                }

                allPages.push(page);
            });
        }

        // ────────────────────────────────────
        // PAGE NUMBERS
        // ────────────────────────────────────
        allPages.forEach((page, index) => {
            page.page_number = index + 1;
            page.total_pages = allPages.length;
        });

        // ────────────────────────────────────
        // RENDER
        // ────────────────────────────────────
        const context = {
            REPORT_TITLE: reportTitle,
            REPORT_SUBTITLE: reportSubtitle,
            REPORT_DATE: reportDate,
            REPORT_ID: reportId,
            CLASSIFICATION: classification,
            CLASSIFICATION_CLASS: classificationClass,
            UPDATE_CADENCE: updateCadence,
            ORG_NAME: orgName,
            PAGES: allPages
        };

        // Try to load logo
        const logoSearchPaths = [
            path.join(__dirname, "Intelligence Report/logo.png"),
            "/usr/src/app/Intelligence Report/logo.png",
            path.join(__dirname, "Intelligence Report/logo.svg"),
            "/usr/src/app/Intelligence Report/logo.svg",
            path.join(__dirname, "Intelligence Report/logo.jpg"),
            "/usr/src/app/Intelligence Report/logo.jpg",
        ];

        for (const lp of logoSearchPaths) {
            if (fs.existsSync(lp)) {
                context.LOGO = await getImageDataUrl(lp);
                console.log(`Logo loaded from: ${lp}`);
                break;
            }
        }

        // Allow logo override via API input (base64 data URL or https URL)
        if (input.logo_url) {
            context.LOGO = String(input.logo_url);
        }
        if (input.logo_base64) {
            context.LOGO = String(input.logo_base64);
        }

        const html = compiledTemplate(context);
        const pdfBytes = await renderDynamicPdf(html);

        // Store PDF
        const pdfId = getRandomString(16);
        pdfStore.set(pdfId, { buffer: pdfBytes, createdAt: Date.now() });

        const protocol = req.headers['x-forwarded-proto'] || req.protocol || 'https';
        const host = req.headers['x-forwarded-host'] || req.headers['host'];
        const basePath = process.env.BASE_PATH || '/apps/intelligence-report';
        const url = `${protocol}://${host}${basePath}/intelligence-report/download/${pdfId}`;

        // Cleanup session
        sessionStore.delete(sessionId);

        console.log(`Report generated: ${pdfId} — ${allPages.length} pages`);
        res.json({ message: "Intelligence report generated", url, pages: allPages.length });

    } catch (e) {
        console.error("PDF Generation Error:", e);
        res.status(500).json({ error: "Failed to generate PDF: " + e.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Intelligence Report Server running on port ${PORT}`);
});

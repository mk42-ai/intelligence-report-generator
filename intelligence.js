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
const pdfStore = new Map(); // id -> { buffer, createdAt }
const PDF_TTL = 60 * 60 * 1000; // 1 hour

// Cleanup expired PDFs every 10 minutes
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

// Serve PDFs directly from memory
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

async function getImageDataUrl(filePath) {
    try {
        const data = fs.readFileSync(filePath);
        const ext = path.extname(filePath).toLowerCase().replace('.', '');
        const mime = ext === 'svg' ? 'image/svg+xml' : `image/${ext}`;
        return `data:${mime};base64,${data.toString('base64')}`;
    } catch {
        return "";
    }
}

// Threat level badge HTML
function getThreatBadgeHtml(level) {
    const l = (level || "moderate").toLowerCase();
    let cls = "moderate";
    if (l.includes("critical")) cls = "critical";
    else if (l.includes("high")) cls = "high";
    else if (l.includes("low")) cls = "low";
    return `<span class="threat-badge ${cls}">${level}</span>`;
}

// Confidence badge HTML
function getConfidenceBadgeHtml(confidence) {
    const c = (confidence || "moderate").toLowerCase();
    let cls = "moderate-conf";
    if (c.includes("high")) cls = "high-conf";
    else if (c.includes("low")) cls = "low-conf";
    const label = confidence || "Moderate";
    return `<span class="conf-badge ${cls}">${label}</span>`;
}

// Escalation badge HTML
function getEscalationHtml(value) {
    if (!value) return "";
    const v = String(value);
    // If it contains a percentage, color it
    const num = parseInt(v);
    let color = "var(--moderate)";
    if (num >= 70) color = "var(--critical)";
    else if (num >= 40) color = "var(--high)";
    else if (num < 20) color = "var(--low)";
    return `<span style="font-weight:700;color:${color};">${v}</span>`;
}

// Classification class mapping
function getClassificationClass(classification) {
    const c = (classification || "").toLowerCase().replace(/[^a-z]/g, '');
    if (c.includes("topsecret")) return "top-secret";
    if (c.includes("secret")) return "secret";
    if (c.includes("confidential")) return "confidential";
    if (c.includes("fouo") || c.includes("official")) return "fouo";
    return "unclassified";
}

// Stat card class
function getStatClass(type) {
    const t = (type || "").toLowerCase();
    if (t.includes("critical")) return "critical-stat";
    if (t.includes("high")) return "high-stat";
    if (t.includes("moderate")) return "moderate-stat";
    if (t.includes("low")) return "low-stat";
    return "accent-stat";
}

// Assessment box class
function getAssessmentBoxClass(level) {
    const l = (level || "").toLowerCase();
    if (l.includes("critical")) return "critical-box";
    if (l.includes("high")) return "high-box";
    if (l.includes("moderate")) return "moderate-box";
    if (l.includes("low")) return "low-box";
    return "accent-box";
}

// ---------------------------------------------------------
// PUPPETEER RENDER
// ---------------------------------------------------------
async function renderDynamicPdf(html) {
    const browser = await puppeteer.launch({
        headless: "new",
        args: ["--no-sandbox", "--disable-setuid-sandbox"]
    });

    try {
        const page = await browser.newPage();
        await page.setViewport({ width: 960, height: 1200 });
        await page.setContent(html, { waitUntil: "networkidle0" });

        const mergedPdf = await PDFDocument.create();

        const pageCount = await page.evaluate(() => {
            return document.querySelectorAll('.page-wrapper').length;
        });

        for (let i = 0; i < pageCount; i++) {
            const dimensions = await page.evaluate((index) => {
                const wrappers = document.querySelectorAll('.page-wrapper');
                const container = document.querySelector('.page-container');

                // Hide all wrappers
                wrappers.forEach(w => w.style.display = 'none');

                // Show current wrapper
                const current = wrappers[index];
                current.style.display = 'flex';

                // Remove container padding so content starts at top
                if (container) {
                    container.style.paddingTop = '0px';
                    container.style.marginTop = '0px';
                }

                // Force reflow
                current.offsetHeight;

                const box = current.getBoundingClientRect();
                return {
                    width: box.width,
                    height: Math.ceil(box.height) + 76 // Add space for fixed header
                };
            }, i);

            const pageBuffer = await page.pdf({
                printBackground: true,
                width: `${dimensions.width}px`,
                height: `${dimensions.height}px`,
                pageRanges: '1',
                margin: { top: 0, right: 0, bottom: 0, left: 0 }
            });

            const pdfDoc = await PDFDocument.load(pageBuffer);
            const [copiedPage] = await mergedPdf.copyPages(pdfDoc, [0]);
            mergedPdf.addPage(copiedPage);
        }

        const pdfBytes = await mergedPdf.save();
        return Buffer.from(pdfBytes);
    } finally {
        await browser.close();
    }
}

// ---------------------------------------------------------
// SESSION STORE
// ---------------------------------------------------------
const sessionStore = new Map();

// ---------------------------------------------------------
// API ENDPOINTS
// ---------------------------------------------------------

/**
 * START SESSION
 */
app.post('/intelligence-report/start', (req, res) => {
    let { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });

    sessionId = sessionId.toLowerCase();
    if (sessionStore.has(sessionId)) {
        return res.status(400).json({ error: "Session exists" });
    }

    sessionStore.set(sessionId, { timestamp: Date.now(), data: {} });
    console.log(`Session started: ${sessionId}`);
    res.json({ message: "Session started", sessionId });
});

/**
 * UPDATE DATA
 */
app.post('/intelligence-report/update', (req, res) => {
    const body = req.body;
    if (!body.sessionId) return res.status(400).json({ error: "sessionId required" });

    const sessionId = body.sessionId.toLowerCase();
    if (!sessionStore.has(sessionId)) return res.status(404).json({ error: "Session not found" });

    const session = sessionStore.get(sessionId);
    session.timestamp = Date.now();
    session.data = { ...session.data, ...body };

    res.json({ message: "Data updated" });
});

/**
 * GENERATE PDF
 */
app.post('/intelligence-report/generate', async (req, res) => {
    let { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });

    sessionId = sessionId.toLowerCase();
    if (!sessionStore.has(sessionId)) return res.status(404).json({ error: "Session not found" });

    try {
        const session = sessionStore.get(sessionId);
        const input = session.data || {};

        // ────────────────────────────────────
        // SAFE DEFAULTS
        // ────────────────────────────────────
        const reportTitle = String(input.report_title || "Intelligence Assessment");
        const reportSubtitle = String(input.report_subtitle || "Predictive Threat Assessment");
        const reportDate = String(input.report_date || new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }));
        const reportId = String(input.report_id || `IR-${Date.now().toString(36).toUpperCase()}`);
        const classification = String(input.classification || "UNCLASSIFIED");
        const classificationClass = getClassificationClass(classification);
        const updateCadence = String(input.update_cadence || "As Required");

        // ────────────────────────────────────
        // BUILD PAGES
        // ────────────────────────────────────
        const compiledTemplate = Handlebars.compile(fs.readFileSync(TEMPLATE_PATH, 'utf8'));
        let allPages = [];

        // ── COVER PAGE ──
        const coverStats = Array.isArray(input.key_stats) ? input.key_stats.map(s => ({
            value: String(s.value || "—"),
            label: String(s.label || ""),
            stat_class: getStatClass(s.type)
        })) : [];

        const coverPage = {
            is_cover_page: true,
            section_ref: "Cover",
            executive_summary: String(input.executive_summary || ""),
            stats: coverStats.length > 0 ? coverStats : undefined,
        };

        if (input.cover_image) {
            coverPage.cover_image = {
                url: String(input.cover_image.url || ""),
                caption: String(input.cover_image.caption || ""),
                source: String(input.cover_image.source || "")
            };
        }

        allPages.push(coverPage);

        // ── THREAT MATRIX PAGE ──
        if (Array.isArray(input.threat_matrix) && input.threat_matrix.length > 0) {
            const threatRows = input.threat_matrix.map(t => ({
                domain: String(t.domain || ""),
                threat_badge_html: getThreatBadgeHtml(String(t.threat_level || "Moderate")),
                vector: String(t.vector || ""),
                status: String(t.status || ""),
                concerns: String(t.concerns || "")
            }));

            const keyIndicators = Array.isArray(input.key_indicators) ? input.key_indicators.map(ind => ({
                text: String(ind.text || ind),
                indicator_class: (ind.level || "").toLowerCase().includes("critical") ? "critical-ind" :
                                 (ind.level || "").toLowerCase().includes("high") ? "high-ind" : ""
            })) : [];

            allPages.push({
                is_threat_matrix_page: true,
                section_ref: "Threat Assessment",
                section_title: String(input.threat_matrix_title || "Threat Matrix"),
                section_desc: String(input.threat_matrix_desc || "Multi-domain threat assessment with current status and key concerns."),
                threat_rows: threatRows,
                key_indicators: keyIndicators.length > 0 ? keyIndicators : undefined,
                indicator_horizon: String(input.indicator_horizon || "72h")
            });
        }

        // ── ANALYSIS SECTIONS ──
        const analysisSections = Array.isArray(input.analysis_sections) ? input.analysis_sections : [];
        analysisSections.forEach(section => {
            const page = {
                is_analysis_page: true,
                section_ref: String(section.section_ref || section.eyebrow || "Analysis"),
                section_eyebrow: String(section.eyebrow || "Domain Analysis"),
                section_title: String(section.title || "Analysis"),
                section_desc: String(section.description || ""),
            };

            // Assessment confidence
            if (section.assessment_confidence) {
                page.assessment_confidence = true;
                page.assessment_confidence_html = getConfidenceBadgeHtml(String(section.assessment_confidence));
            }

            // Table data
            if (section.table) {
                page.analysis_table = {
                    headers: Array.isArray(section.table.headers) ? section.table.headers.map(String) : [],
                    rows: Array.isArray(section.table.rows) ? section.table.rows.map(row => ({
                        cells: Array.isArray(row.cells) ? row.cells.map(cell => {
                            // Auto-detect threat/confidence badges in cell content
                            return String(cell);
                        }) : Array.isArray(row) ? row.map(String) : []
                    })) : []
                };
            }

            // Free text
            if (section.text) {
                page.analysis_text = String(section.text);
            }

            // Images (satellite, evidence, etc.)
            if (Array.isArray(section.images) && section.images.length > 0) {
                page.images = section.images.map(img => ({
                    url: String(img.url || ""),
                    caption: String(img.caption || ""),
                    meta: String(img.meta || ""),
                    type: String(img.type || "")
                }));
            }

            // Vulnerability cards
            if (Array.isArray(section.vulnerability_cards) && section.vulnerability_cards.length > 0) {
                page.vulnerability_cards = section.vulnerability_cards.map(vc => ({
                    title: String(vc.title || ""),
                    threat_badge_html: getThreatBadgeHtml(String(vc.threat_level || "Moderate")),
                    image_url: vc.image_url ? String(vc.image_url) : undefined,
                    details: String(vc.details || "")
                }));
            }

            // Analyst assessment
            if (section.analyst_assessment) {
                page.analyst_assessment = String(section.analyst_assessment);
            }

            allPages.push(page);
        });

        // ── SCENARIO / OUTLOOK PAGE ──
        if (Array.isArray(input.scenarios) && input.scenarios.length > 0) {
            const scenarioPage = {
                is_scenario_page: true,
                section_ref: "Predictive Outlook",
                section_title: String(input.scenario_title || "Predictive Outlook & Scenario Analysis"),
                section_desc: String(input.scenario_desc || "Forward-looking assessment of likely developments and decision triggers."),
                scenarios: input.scenarios.map(s => ({
                    horizon: String(s.horizon || ""),
                    most_likely: String(s.most_likely || ""),
                    escalation_html: getEscalationHtml(s.escalation_risk),
                    triggers: String(s.triggers || "")
                }))
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
                section_title: String(input.actions_title || "Priority Actions & Recommendations"),
                section_desc: String(input.actions_desc || "Immediate and near-term actions recommended based on current intelligence."),
                actions: input.priority_actions.map((a, i) => ({
                    num: i + 1,
                    title: String(a.title || ""),
                    description: String(a.description || "")
                }))
            };

            if (input.highest_yield_action) {
                actionsPage.highest_yield_action = String(input.highest_yield_action);
            }

            // Data quality table
            if (Array.isArray(input.data_quality) && input.data_quality.length > 0) {
                actionsPage.data_quality = input.data_quality.map(d => ({
                    domain: String(d.domain || ""),
                    confidence_html: getConfidenceBadgeHtml(String(d.confidence || "Moderate")),
                    notes: String(d.notes || "")
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
                    text: String(typeof s === 'string' ? s : s.text || "")
                })),
                deliverable_note: input.deliverable_note ? String(input.deliverable_note) : undefined
            });
        }

        // ── GENERIC CONTENT PAGES ──
        if (Array.isArray(input.additional_pages)) {
            input.additional_pages.forEach(pg => {
                const page = {
                    is_content_page: true,
                    section_ref: String(pg.section_ref || "Appendix"),
                    section_eyebrow: String(pg.eyebrow || "Additional Analysis"),
                    section_title: String(pg.title || ""),
                    section_desc: pg.description ? String(pg.description) : undefined,
                    content_html: String(pg.content_html || ""),
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
            PAGES: allPages
        };

        // Try to load logo
        const logoPath = "/usr/src/app/Intelligence Report/logo.svg";
        const localLogoPath = path.join(__dirname, "Intelligence Report/logo.svg");
        const logoFile = fs.existsSync(logoPath) ? logoPath : (fs.existsSync(localLogoPath) ? localLogoPath : null);
        if (logoFile) {
            context.LOGO = await getImageDataUrl(logoFile);
        }

        const html = compiledTemplate(context);
        const pdfBytes = await renderDynamicPdf(html);

        // Store PDF in memory and return download URL
        const pdfId = getRandomString(16);
        pdfStore.set(pdfId, { buffer: pdfBytes, createdAt: Date.now() });

        // Build download URL using the request's host
        const protocol = req.headers['x-forwarded-proto'] || req.protocol || 'https';
        const host = req.headers['x-forwarded-host'] || req.headers['host'];
        const url = `${protocol}://${host}/intelligence-report/download/${pdfId}`;

        // Cleanup session
        sessionStore.delete(sessionId);

        res.json({ message: "Intelligence report generated", url });

    } catch (e) {
        console.error("PDF Generation Error:", e);
        res.status(500).json({ error: "Failed to generate PDF: " + e.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Intelligence Report Server running on port ${PORT}`);
});

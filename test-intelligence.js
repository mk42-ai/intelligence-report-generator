/**
 * Test script for the Intelligence Report Generator
 * Run: node test-intelligence.js
 * Requires the server to be running on localhost:3000
 */

const BASE_URL = "http://localhost:3000";

async function postJSON(endpoint, body) {
    const res = await fetch(`${BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    });
    const data = await res.json();
    console.log(`${endpoint} =>`, res.status, JSON.stringify(data).slice(0, 200));
    return data;
}

async function runTest() {
    const sessionId = `test-${Date.now()}`;

    console.log("\n=== STARTING INTELLIGENCE REPORT TEST ===\n");

    // 1. Start session
    await postJSON("/intelligence-report/start", { sessionId });

    // 2. Send report data
    await postJSON("/intelligence-report/update", {
        sessionId,

        // Report metadata
        report_title: "UAE-Iran Crisis — Predictive Threat Assessment",
        report_subtitle: "Launch Sites • Maritime Routes • Airspace • Strike Vectors • Economic Impact — Forward-looking operational intelligence",
        report_date: "March 7, 2026",
        report_id: "IR-2026-0307-UAE",
        classification: "SENSITIVE — FOR OFFICIAL USE ONLY",
        update_cadence: "24h Rolling",

        // Executive Summary
        executive_summary: "Iran's military posture has shifted from deterrence signaling to <strong>pre-positioning for potential kinetic action</strong> across multiple domains. Satellite imagery confirms MRBM launcher dispersal from known garrisons, naval fast-attack craft surge in the Strait of Hormuz, and elevated SIGINT activity consistent with operational planning. UAE critical infrastructure faces a coordinated multi-vector threat with <strong>48% probability of limited strike within 7 days</strong>. Immediate defensive measures recommended.",

        // Key Stats
        key_stats: [
            { value: "CRITICAL", label: "Overall Threat Level", type: "critical" },
            { value: "48%", label: "7-Day Strike Probability", type: "high" },
            { value: "5", label: "Active Threat Domains", type: "moderate" },
            { value: "24h", label: "Update Cadence", type: "accent" }
        ],

        // Cover image
        cover_image: {
            url: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=900&q=80",
            caption: "Satellite monitoring coverage of the Persian Gulf strategic corridor",
            source: "Visual context / illustrative"
        },

        // Threat Matrix
        threat_matrix_title: "Multi-Domain Threat Matrix",
        threat_matrix_desc: "Comprehensive assessment across all operational domains with current threat levels, primary vectors, and key concerns for UAE defense planners.",
        threat_matrix: [
            { domain: "Ballistic / Cruise Missiles", threat_level: "CRITICAL", vector: "Shahab-3, Fateh-110, Soumar LACM", status: "Launchers dispersed / elevated", concerns: "MRBM dispersal from known garrisons consistent with pre-launch preparation" },
            { domain: "Maritime / Naval", threat_level: "CRITICAL", vector: "IRGCN fast-attack + mine warfare", status: "Detected surge / interdiction posture", concerns: "Dhow-concealed mines, fast-boat swarms targeting commercial shipping" },
            { domain: "Hezbollah / Proxy", threat_level: "HIGH", vector: "UAV/drone + infiltration teams", status: "Elevated cross-border signals", concerns: "Proxy activation via IRGC-QF coordination channels" },
            { domain: "Cyber / EW", threat_level: "HIGH", vector: "APT33/34 + GPS spoofing", status: "Active probing / increased scanning", concerns: "Pre-positioned access to energy sector SCADA systems" },
            { domain: "Infrastructure / Energy", threat_level: "MODERATE", vector: "Oil exports + desalination targeting", status: "Contingent on escalation", concerns: "Cascading effects on civilian infrastructure if strikes proceed" },
            { domain: "Diplomatic", threat_level: "HIGH", vector: "Backchannel + economic signaling", status: "Fluid / uncertain", concerns: "Talks may be cover for operational preparation" }
        ],

        // Key Indicators
        key_indicators: [
            { text: "<strong>Ballistic geometry change:</strong> 3 new mid-altitude assets identified from launch site near Isfahan. Azimuth 115° suggests Gulf-facing trajectory.", level: "critical" },
            { text: "<strong>Port movement:</strong> Bandar Abbas fast-attack craft sorties increased 3x over baseline. Multiple vessels conducting night exercises.", level: "critical" },
            { text: "<strong>SIGINT intercept:</strong> Encrypted C2 traffic volume between Tehran and southern military districts up 280% vs. 30-day average.", level: "high" },
            { text: "<strong>Diplomatic cover:</strong> Senior Iranian officials have made contradictory public statements — classic operational security pattern.", level: "high" },
            { text: "<strong>Signaling shift:</strong> Iranian state media shifted from diplomatic messaging to military capability showcasing.", level: "moderate" }
        ],
        indicator_horizon: "72h",

        // Analysis Sections
        analysis_sections: [
            {
                eyebrow: "Strike Vectors",
                title: "Strike Vectors & Ground-Level Vulnerability",
                description: "Analysis of Iranian strike capabilities mapped to UAE critical infrastructure targets, with satellite imagery verification.",
                section_ref: "Strike Vectors",
                assessment_confidence: "High",
                table: {
                    headers: ["Site / Target", "Threat Level", "Vulnerability", "Strike Probability (7-day)", "Key Concern"],
                    rows: [
                        { cells: ["<strong>DXB Terminal 2</strong>", getThreatBadgeStr("CRITICAL"), "Concentrated civilian infrastructure", "25%", "Runway damage would cascade to regional logistics"] },
                        { cells: ["<strong>Jebel Ali Port</strong>", getThreatBadgeStr("CRITICAL"), "Largest Middle East port facility", "30%", "Attempted interdiction/blockade approach"] },
                        { cells: ["<strong>Al Dhafra Air Base</strong>", getThreatBadgeStr("HIGH"), "Combined UAE/US military operations", "20%", "Aircraft dispersal recommended immediately"] },
                        { cells: ["<strong>Taweelah Desalination</strong>", getThreatBadgeStr("HIGH"), "Supplies 40% of Abu Dhabi fresh water", "15%", "Cascading civilian impact — highest humanitarian risk"] },
                        { cells: ["<strong>Fujairah Oil Terminal</strong>", getThreatBadgeStr("MODERATE"), "Strategic petroleum reserves", "10%", "Previous sabotage target (2019 precedent)"] }
                    ]
                },
                images: [
                    { url: "https://images.unsplash.com/photo-1569154941061-e231b4725ef1?w=600&q=80", caption: "Jebel Ali Port Complex — Strategic Infrastructure", meta: "Critical logistics hub serving 3.8M+ TEU annually", type: "Satellite" },
                    { url: "https://images.unsplash.com/photo-1540575467063-178a50c2df87?w=600&q=80", caption: "Al Dhafra Air Base Perimeter Assessment", meta: "Combined military operations facility", type: "Overhead" },
                    { url: "https://images.unsplash.com/photo-1513828583688-c52646db42da?w=600&q=80", caption: "Taweelah Water Desalination Complex", meta: "Critical civilian infrastructure — 40% regional water supply", type: "Satellite" },
                    { url: "https://images.unsplash.com/photo-1518709766631-a6a7f45921c3?w=600&q=80", caption: "Fujairah Oil Terminal & Storage Facility", meta: "Strategic petroleum reserve — ADCOP pipeline terminus", type: "Overhead" }
                ],
                analyst_assessment: "The concentration of high-value targets within a 200km coastal corridor creates a <strong>force-multiplier vulnerability</strong>. A coordinated multi-vector strike combining ballistic missiles with maritime disruption could overwhelm THAAD/Patriot coverage windows. <strong>Recommend immediate dispersal of mobile assets</strong> and activation of passive defense measures across all Tier-1 infrastructure."
            },
            {
                eyebrow: "Maritime Analysis",
                title: "Maritime: Strait of Hormuz, Gulf Approaches & Port Activity",
                description: "Live vessel tracking, observable ship densities, anchoring behavior, and chokepoint forward models.",
                section_ref: "Maritime",
                assessment_confidence: "High",
                table: {
                    headers: ["Route / Zone", "Status", "Observed Activity", "Risk Level", "Interpretation"],
                    rows: [
                        { cells: ["Strait of Hormuz TSS", "<strong>Congested</strong>", "N+11 vessels above baseline (+4.1 SD)", getThreatBadgeStr("CRITICAL"), "Port access delays reaching 8h+"] },
                        { cells: ["Unidentified Dhow Cluster", "<strong>Disputed</strong>", "N+200 records (anomalous vs 7-day avg)", getThreatBadgeStr("HIGH"), "Throughout anchorage zones — mining concern"] },
                        { cells: ["Gulf of Oman Approaches", "<strong>Congested</strong>", "Slowdown records; congestion confirmed", getThreatBadgeStr("MODERATE"), "Refueling / rerouting activity increasing"] },
                        { cells: ["Abu Dhabi Anchorage", "<strong>Elevated</strong>", "12 unscheduled anchorages detected", getThreatBadgeStr("HIGH"), "Holding condition; may await military clearance"] }
                    ]
                },
                images: [
                    { url: "https://images.unsplash.com/photo-1589656966895-2f33e7653819?w=600&q=80", caption: "Strait of Hormuz — Maritime Traffic Density", meta: "AIS tracking shows 4.1 SD above baseline vessel count", type: "AIS Data" },
                    { url: "https://images.unsplash.com/photo-1534008897995-27a23e859048?w=600&q=80", caption: "Gulf of Oman Satellite Composite", meta: "Multi-spectral imagery — vessel pattern analysis", type: "Sentinel" }
                ],
                analyst_assessment: "The vessel density anomaly in the Hormuz TSS is <strong>not consistent with normal commercial patterns</strong>. Historical comparison to the 2019 tanker incidents shows similar pre-event clustering. The unidentified dhow activity warrants immediate investigation — this pattern is consistent with <strong>mine-laying preparation</strong> or ISR (intelligence, surveillance, reconnaissance) operations."
            },
            {
                eyebrow: "Airspace & Infrastructure",
                title: "Airspace Disruption, Infrastructure Continuity & Economic Impact",
                description: "Live air traffic analysis, infrastructure status monitoring, and quantified economic pressure indicators.",
                section_ref: "Airspace & Economy",
                assessment_confidence: "Moderate",
                table: {
                    headers: ["Target / System", "Damage Scenario", "Status", "Repair Timeline", "Criticality"],
                    rows: [
                        { cells: ["DXB Terminal 2", "Runway/terminal infrastructure damage", "<strong>Operational</strong>", "6-18 months", getThreatBadgeStr("CRITICAL")] },
                        { cells: ["Jebel Ali Port", "Disruption to container handling", "<strong>Operational</strong>", "12-24 months", getThreatBadgeStr("CRITICAL")] },
                        { cells: ["Taweelah Desalination", "Water supply disruption to Abu Dhabi", "At Risk", "6-12 months", getThreatBadgeStr("HIGH")] },
                        { cells: ["Fujairah Pipeline", "Maritime-route oil export disruption", "<strong>International monitoring</strong>", "Ongoing", getThreatBadgeStr("MODERATE")] }
                    ]
                },
                analyst_assessment: "Aviation disruption becomes a <strong>strategic lever</strong> even without kinetic strikes. Airspace closures or GPS spoofing in UAE FIR could trigger international insurance repricing, airline rerouting, and a confidence crisis affecting $50B+ in annual tourism revenue. The <strong>economic bleed rate</strong> under sustained tensions is estimated at $180-340M per week across affected sectors."
            }
        ],

        // Scenarios
        scenario_title: "Predictive Outlook (72h / 30 / 90 Day)",
        scenario_desc: "Forward-looking assessment based on pattern analysis, historical precedent modeling, and SIGINT correlation.",
        scenarios: [
            { horizon: "7 Days", most_likely: "Intensified cross-domain signaling with graduated escalation. Possible limited kinetic test.", escalation_risk: "48%", triggers: "High-casualty event, missile or airspace degradation, commercial vessel strike" },
            { horizon: "30 Days", most_likely: "Proxy activation via Houthi/Hezbollah vectors if direct escalation is politically constrained.", escalation_risk: "35%", triggers: "Escalation through proxy channels with plausible deniability" },
            { horizon: "90 Days", most_likely: "Either negotiated de-escalation through back-channels or sustained grey-zone pressure campaign.", escalation_risk: "20%", triggers: "Failure of diplomatic channels; escalatory spiral through miscalculation" }
        ],

        // Escalation / De-escalation
        escalation_indicators: [
            "Air defense asset management: IRGC 'whiteout interrogation' noticed on THAAD-adjacent frequencies",
            "Port-based marker sequencing: Additional IRGCN vessel movements from Bandar Abbas",
            "Cyber pre-positioning: Increased scanning of UAE critical infrastructure networks",
            "Diplomatic breakdown: Withdrawal of backchannel communication partners"
        ],
        deescalation_indicators: [
            "Resumption of bilateral communication through Swiss/Omani intermediaries",
            "MRBM launcher return to garrison positions (verifiable via satellite)",
            "Reduction in SIGINT activity volume to baseline levels",
            "Public diplomatic overtures with substantive confidence-building measures"
        ],

        // Priority Actions
        actions_title: "Priority Actions (Next 24-72h)",
        actions_desc: "Immediate and near-term defensive actions recommended for UAE decision-makers.",
        priority_actions: [
            { title: "Air defense asset management", description: "Distribute air-defense interceptors for consequence reduction. <strong>Activate all THAAD and Patriot batteries</strong> to maximum readiness. Coordinate with US CENTCOM for integrated air defense picture." },
            { title: "Maritime domain awareness", description: "Deploy additional ISR assets to Hormuz TSS. <strong>Investigate unidentified dhow cluster</strong> for potential mine-laying activity. Issue NOTAM for commercial shipping." },
            { title: "Cyber defense activation", description: "Elevate cybersecurity posture across all critical infrastructure SCADA systems. <strong>Implement emergency network segmentation</strong> for energy sector assets." },
            { title: "Port/airport reverse sequencing", description: "If Maritime & Airspace threats materialize, execute <strong>staged evacuation planning</strong> for non-essential personnel at Tier-1 facilities." },
            { title: "Diplomatic parallel track", description: "Maintain back-channel communication via Oman/Switzerland while simultaneously preparing defensive posture. <strong>Do not signal weakness but maintain off-ramp availability.</strong>" }
        ],
        highest_yield_action: "Immediate deployment of a <strong>distributed air defense architecture</strong> covering the Dubai-Abu Dhabi corridor, combined with <strong>aggressive maritime ISR</strong> in the Hormuz approaches. This provides maximum deterrent signal while maintaining escalation control.",

        // Data Quality
        data_quality: [
            { domain: "Satellite / IMINT", confidence: "HIGH", notes: "Multiple commercial + classified sources; 24h revisit rate" },
            { domain: "SIGINT / COMINT", confidence: "HIGH", notes: "Strong pattern correlation; confirmed by multiple collection platforms" },
            { domain: "Maritime AIS", confidence: "MODERATE", notes: "AIS gaps in some areas; spoofing suspected for IRGCN vessels" },
            { domain: "Cyber / OSINT", confidence: "MODERATE", notes: "Open-source cross-referenced; some unverified claims filtered" },
            { domain: "Diplomatic / HUMINT", confidence: "LOW", notes: "Limited direct access; relying on partner assessments" },
            { domain: "Proxy / Non-State", confidence: "MODERATE", notes: "Pattern analysis from historical precedent; limited real-time data" }
        ],

        // Sources
        sources: [
            "OSINT — UAE national aviation authority flight data (publicly available schedules and NOTAMs)",
            "AIS Data — MarineTraffic / VesselFinder (commercial maritime tracking platforms)",
            "Satellite Imagery — Copernicus Sentinel-2 open data portal (ESA)",
            "Geopolitical Analysis — International Institute for Strategic Studies (IISS) conflict tracker",
            "Economic Data — IMF regional economic outlook, UAE Ministry of Economy public statistics",
            "Cyber Threat Intelligence — CISA advisories, Mandiant APT tracking (public reports)",
            "Historical Precedent — 2019 Fujairah/Abqaiq attack pattern analysis (public domain)"
        ],

        deliverable_note: "This intelligence product is generated for <strong>authorized decision-makers only</strong>. It is not a full-spectrum intelligence assessment and should be supplemented with classified briefings. Source reliability has been assessed using standard intelligence community confidence scales. All satellite imagery in this report is from <strong>commercially available open-source platforms</strong>."
    });

    // 3. Generate PDF
    console.log("\nGenerating PDF... (this may take 30-60 seconds)\n");
    const result = await postJSON("/intelligence-report/generate", { sessionId });

    console.log("\n=== TEST COMPLETE ===");
    console.log("PDF URL:", result.url);
}

// Helper for test data - returns raw HTML string for table cells
function getThreatBadgeStr(level) {
    const l = level.toLowerCase();
    let cls = "moderate";
    if (l.includes("critical")) cls = "critical";
    else if (l.includes("high")) cls = "high";
    else if (l.includes("low")) cls = "low";
    return `<span class="threat-badge ${cls}">${level}</span>`;
}

runTest().catch(err => {
    console.error("Test failed:", err);
    process.exit(1);
});

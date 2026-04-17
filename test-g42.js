// Test script for G42 Daily Media Intelligence Report
// Populates a session with realistic sample data and generates a PDF.
// Usage: node g42-intelligence.js (in one terminal) && node test-g42.js

const http = require('http');

const HOST = 'localhost';
const PORT = process.env.G42_PORT || 3001;
const SESSION = 'g42-test-' + Date.now();

function req(method, path, body) {
    return new Promise((resolve, reject) => {
        const data = body ? JSON.stringify(body) : null;
        const opts = {
            hostname: HOST, port: PORT, path, method,
            headers: { 'Content-Type': 'application/json', ...(data ? { 'Content-Length': Buffer.byteLength(data) } : {}) },
        };
        const r = http.request(opts, res => {
            let chunks = '';
            res.on('data', c => chunks += c);
            res.on('end', () => {
                try { resolve({ status: res.statusCode, body: JSON.parse(chunks) }); }
                catch { resolve({ status: res.statusCode, body: chunks }); }
            });
        });
        r.on('error', reject);
        if (data) r.write(data);
        r.end();
    });
}

const reportData = {
    sessionId: SESSION,
    report_title: "Daily Media Intelligence",
    report_subtitle: "G42 Executive Briefing — Tier-1 editorial, social, and policy signal across UAE, GCC, and global markets",
    classification: "CONFIDENTIAL",
    coverage_window: "Media headlines from April 15, 2026, at 5:30 PM EST",
    published_at: "05:00 GST Daily",
    analyst: "OnDemand OS · Autonomous Agent v1.2",

    key_stats: [
        { value: "247", label: "Total Mentions", sub: "+18% vs 7-day avg", type: "gold" },
        { value: "63", label: "Tier-1 Outlets", sub: "Reuters, FT, NYT, WaPo", type: "positive" },
        { value: "12.4M", label: "Estimated Reach", sub: "Unique impressions", type: "gold" },
        { value: "+0.62", label: "Net Sentiment", sub: "Score: Positive-leaning", type: "positive" },
    ],

    sentiment: { positive: 148, neutral: 79, negative: 20 },

    executive_summary: [
        "G42 coverage today was dominated by three stories: (1) H.H. Sheikh Tahnoon's keynote at the Global AI Summit in Abu Dhabi drew strong Tier-1 coverage, with Reuters, Bloomberg, and FT running feature pieces; (2) Space42 announced a new satellite constellation partnership picked up across defense and space trades; (3) continued positive framing around G42's US-UAE AI cooperation agreement.",
        "Social volume concentrated on LinkedIn (professional network effect from leadership posts) and X (real-time pickup of summit keynote clips). Instagram and YouTube saw secondary amplification through short-form recaps. No material negative coverage detected in today's window; two minor critical pieces from policy blogs flagged but scored low-impact.",
    ],

    top_takeaways: [
        { label: "Signal 01 · Executive Visibility", text: "Sheikh Tahnoon's AI Summit keynote generated the highest single-day executive mention volume in the last 30 days. Tier-1 outlets framed the address as a sovereign-AI positioning statement for the UAE." },
        { label: "Signal 02 · Portfolio Momentum", text: "Space42 constellation partnership drove a 34% uplift in portfolio-company coverage. Defense trades and space media (SpaceNews, Defense One, Breaking Defense) picked up the announcement within 6 hours." },
        { label: "Signal 03 · Policy Positive", text: "US-UAE AI cooperation narrative remains consistently positive across Washington Post, Reuters, and The Information. No negative framing detected in the US policy press today." },
    ],

    sheikh_tahnoon: {
        total_mentions: 58,
        reach: "4.8M",
        tier1_mentions: 19,
        sentiment_score: "+0.74",
        mentions: [
            {
                source: "Reuters",
                title: "UAE's Sheikh Tahnoon pledges sovereign-AI push, deeper US ties at Abu Dhabi summit",
                snippet: "H.H. Sheikh Tahnoon bin Zayed Al Nahyan, chairman of G42, outlined the UAE's ambitions to become a global AI hub at the Abu Dhabi AI Summit, emphasizing deeper cooperation with US partners and sovereign compute capacity.",
                url: "https://reuters.com/technology/uae-sheikh-tahnoon-sovereign-ai-summit-2026-04-16",
                timestamp: "14:32 GST",
                sentiment: "Positive",
                impact: "Tier 1",
                engagement: { reach: "1.2M", shares: "3.4K", comments: "512" },
            },
            {
                source: "Bloomberg",
                title: "Sheikh Tahnoon Outlines G42's Next Phase as UAE's AI Ambitions Crystallize",
                snippet: "The chairman's keynote positioned G42 at the center of a broader national strategy, with Bloomberg analysts noting the address signals commitment to long-horizon infrastructure bets rather than hype-cycle plays.",
                url: "https://bloomberg.com/news/articles/2026-04-16/uae-g42-sheikh-tahnoon-ai-summit",
                timestamp: "15:10 GST",
                sentiment: "Positive",
                impact: "Tier 1",
                engagement: { reach: "890K", shares: "2.1K", comments: "340" },
            },
            {
                source: "Financial Times",
                title: "G42 chairman maps out AI decade for Gulf",
                snippet: "FT positioned Sheikh Tahnoon's remarks as part of a coherent Gulf strategy, with a detailed sidebar on G42's portfolio architecture and sovereign-compute investments.",
                url: "https://ft.com/content/uae-g42-tahnoon-ai-decade-gulf",
                timestamp: "16:05 GST",
                sentiment: "Positive",
                impact: "Tier 1",
                engagement: { reach: "620K", shares: "1.1K", comments: "180" },
            },
            {
                source: "LinkedIn",
                title: "Sheikh Tahnoon's keynote clip shared by senior tech executives",
                snippet: "High-signal reshare activity from founders and investors across the US and Gulf. Strong positive sentiment in comments focused on the sovereign-AI framing.",
                url: "https://linkedin.com/feed/update/urn:li:activity:7186423890234",
                timestamp: "17:20 GST",
                sentiment: "Positive",
                impact: "Tier 2",
                engagement: { reach: "340K", shares: "1.8K", comments: "420" },
            },
            {
                source: "X",
                title: "Keynote clip trending: 'The UAE will build, not just consume AI'",
                snippet: "Pull-quote reached 2.4M impressions within 4 hours of the keynote. Top repliers include international tech journalists and regional policy commentators.",
                url: "https://x.com/g42ai/status/1783421890234567",
                timestamp: "17:45 GST",
                sentiment: "Positive",
                impact: "Tier 1",
                engagement: { reach: "2.4M", shares: "8.7K", comments: "1.2K" },
            },
        ],
        analyst_note: "Sheikh Tahnoon's summit keynote materially shifted today's coverage profile. Net sentiment on his name jumped +0.11 vs 7-day trailing. Recommendation: monitor X for tail-risk threads from policy-critical accounts over the next 48h, but none detected at time of publication.",
    },

    // ═══════════════════════════════════════════════════════════════════
    // EXECUTIVES — POPULATED BY AGENT AT RUNTIME
    // ═══════════════════════════════════════════════════════════════════
    // The OnDemand OS agent discovers G42 executives dynamically from the
    // mention corpus. The template reserves a structured slot (2-col grid,
    // up to 6 execs per page, auto-splits for overflow).
    //
    // Schema reference for agent output:
    //   g42_executives: [{
    //     name:          "<person's full name>",
    //     title:         "<their role>",
    //     mention_count: "<integer>",
    //     tier1_count:   "<integer>",
    //     sentiment:     "<score or label, e.g. '+0.58' or 'Positive'>",
    //     mentions: [{ outlet, headline, url }]    // up to 3 per exec
    //   }, ... ]
    //
    // Leaving empty here so demo shows the agent is responsible for this.
    g42_executives: [],

    g42_corporate: {
        mentions: [
            {
                source: "Wall Street Journal",
                title: "G42's Microsoft Partnership Enters New Phase With $1.5B Expansion Plan",
                snippet: "WSJ sources report the G42-Microsoft partnership is entering a second phase focused on sovereign-cloud build-out across GCC. The piece frames the deal as a template for US-Gulf tech cooperation.",
                url: "https://wsj.com/articles/g42-microsoft-partnership-expansion-2026",
                timestamp: "12:40 GST",
                sentiment: "Positive",
                impact: "Tier 1",
            },
            {
                source: "Washington Post",
                title: "How G42 Became Central to the US-UAE AI Relationship",
                snippet: "Long-form feature examining G42's role as the anchor institution in the US-UAE AI cooperation agreement, with quotes from White House and UAE officials.",
                url: "https://washingtonpost.com/business/2026/04/16/g42-us-uae-ai/",
                timestamp: "13:15 GST",
                sentiment: "Positive",
                impact: "Tier 1",
            },
            {
                source: "BBC",
                title: "G42: The Emirati AI Group Redrawing Global Tech Maps",
                snippet: "BBC analysis piece walking readers through G42's corporate architecture and its portfolio companies, with a sidebar on Sheikh Tahnoon's chairmanship.",
                url: "https://bbc.com/news/business-g42-emirati-ai-redrawing-maps",
                timestamp: "16:50 GST",
                sentiment: "Neutral",
                impact: "Tier 1",
            },
        ],
    },

    portfolio_companies: [
        { name: "Space42", sector: "Space & Satellite", description: "Satellite operations, Earth observation, sovereign space infrastructure for UAE and MENA customers.", mention_count: 34, sentiment: "+0.68", top_story: "Leo constellation partnership" },
        { name: "Presight", sector: "Big Data & AI", description: "AI-powered big data analytics across government, healthcare, and financial services markets.", mention_count: 12, sentiment: "+0.51", top_story: "New government contract — Gulf News" },
        { name: "Inception", sector: "Applied AI", description: "Domain-focused AI products (Jais Arabic LLM, enterprise verticals). Launched from G42's AI research stack.", mention_count: 18, sentiment: "+0.59", top_story: "Jais benchmark update coverage" },
        { name: "Core42", sector: "Cloud & Compute", description: "Sovereign cloud, compute, and AI infrastructure services. Partnership anchor for Microsoft collaboration.", mention_count: 22, sentiment: "+0.64", top_story: "Compute expansion feature in WSJ" },
        { name: "M42", sector: "Health Tech", description: "Health-tech platform combining Mubadala Health and G42 Healthcare capabilities.", mention_count: 9, sentiment: "+0.72", top_story: "Genomics scale-up in FT Health" },
        { name: "Khazna", sector: "Data Centers", description: "Data center operator; backbone for G42 sovereign compute buildout across UAE.", mention_count: 6, sentiment: "+0.48", top_story: "Abu Dhabi campus expansion" },
    ],

    linkedin_coverage: {
        total_posts: 42,
        reach: "1.8M",
        engagement: "84K",
        sentiment_summary: "+0.71",
        posts: [
            {
                author_name: "Peng Xiao", author_handle: "peng-xiao-g42", author_role: "Group CEO at G42", verified: true, author_is_g42: true,
                timestamp: "6h ago",
                content: "Today at the Abu Dhabi AI Summit, we announced the next phase of sovereign-AI buildout for the UAE. Our thesis: AI sovereignty is not about closing doors — it's about building the infrastructure to be a meaningful partner in global AI. Grateful to our teams across Core42, Inception, and Space42 who make this possible.",
                likes: 12400, reposts: 840, comments: 312,
                url: "https://linkedin.com/posts/peng-xiao-g42_ai-summit-announcement",
                sentiment: "Positive",
            },
            {
                author_name: "G42 Group", author_handle: "g42-group", verified: true, author_is_g42: true,
                timestamp: "9h ago",
                content: "H.H. Sheikh Tahnoon bin Zayed Al Nahyan delivering today's keynote at the Abu Dhabi AI Summit. Watch the full remarks at the link below. #G42 #AbuDhabiAISummit",
                likes: 8900, reposts: 1240, comments: 198,
                url: "https://linkedin.com/company/g42/posts/tahnoon-keynote-summit",
                sentiment: "Positive",
            },
            {
                author_name: "Talal Alkaissi", author_handle: "talal-alkaissi", author_role: "CEO at M42 / G42 Healthcare", verified: true, author_is_g42: true,
                timestamp: "11h ago",
                content: "Announcing M42's expanded partnership with Cleveland Clinic Abu Dhabi on precision medicine. Combining genomics, clinical AI, and the Gulf's largest health dataset — this is the decade when sovereign health AI gets real.",
                likes: 4200, reposts: 380, comments: 87,
                url: "https://linkedin.com/posts/talal-alkaissi_m42-cleveland-partnership",
                sentiment: "Positive",
            },
            {
                author_name: "Kara Swisher", author_handle: "kara-swisher", author_role: "Tech journalist", verified: true,
                timestamp: "14h ago",
                content: "Spent the morning with the G42 team in Abu Dhabi. Honestly — the scope of what they're building is not fully appreciated in the US press yet. Sovereign compute, applied AI, space, health — it's a vertical stack not a series of bets.",
                likes: 6800, reposts: 920, comments: 440,
                url: "https://linkedin.com/posts/kara-swisher_g42-abu-dhabi-observations",
                sentiment: "Positive",
            },
        ],
        analyst_note: "LinkedIn activity is dominated by G42 leadership posts and high-credibility third-party amplifiers (tech journalists, founders). No material negative threads detected. Comment sentiment trending positive across all four top posts.",
    },

    x_coverage: {
        total_posts: 78,
        reach: "6.2M",
        engagement: "142K",
        sentiment_summary: "+0.54",
        posts: [
            {
                author_name: "G42", author_handle: "@G42ai", verified: true, author_is_g42: true,
                timestamp: "5h ago",
                content: "LIVE: H.H. Sheikh Tahnoon bin Zayed Al Nahyan keynote at the Abu Dhabi AI Summit. 'The UAE will build, not just consume, the AI of the next decade.' #G42 #AbuDhabiAISummit",
                likes: 24500, reposts: 8700, comments: 1240, views: 2400000,
                url: "https://x.com/G42ai/status/1783421890234567",
                sentiment: "Positive",
            },
            {
                author_name: "Peng Xiao", author_handle: "@pengxiao_g42", verified: true, author_is_g42: true,
                timestamp: "7h ago",
                content: "What sovereign AI looks like in 2026: partnership, not isolation. Compute at home, global talent networks, open research. The UAE model.",
                likes: 8400, reposts: 2100, comments: 580, views: 720000,
                url: "https://x.com/pengxiao_g42/status/1783401234567890",
                sentiment: "Positive",
            },
            {
                author_name: "Reuters Tech", author_handle: "@ReutersTech", verified: true,
                timestamp: "8h ago",
                content: "BREAKING: UAE's G42 outlines $1.5B second-phase Microsoft partnership focused on sovereign cloud. Full story:",
                likes: 5200, reposts: 3400, comments: 412, views: 890000,
                url: "https://x.com/ReutersTech/status/1783389456789012",
                sentiment: "Neutral",
            },
            {
                author_name: "Ben Thompson", author_handle: "@benthompson", verified: true,
                timestamp: "11h ago",
                content: "G42 is the most interesting company that American tech executives are not paying enough attention to. Stratechery piece in the works on why.",
                likes: 12800, reposts: 1800, comments: 340, views: 1100000,
                url: "https://x.com/benthompson/status/1783356789012345",
                sentiment: "Positive",
            },
        ],
        analyst_note: "X conversation bifurcated: mainstream tech/business audiences are positive; niche policy-hawk accounts posted 3 low-engagement critical threads (aggregate <2% of impressions). No breakout negative narrative; monitor for next 48h.",
    },

    instagram_coverage: {
        total_posts: 18,
        reach: "890K",
        engagement: "42K",
        sentiment_summary: "+0.80",
        posts: [
            {
                author_name: "G42 Official", author_handle: "@g42.ai", verified: true, author_is_g42: true,
                timestamp: "9h ago",
                content: "Behind the scenes at today's Abu Dhabi AI Summit. The future is being built here. 🇦🇪 #G42 #AbuDhabiAISummit",
                likes: 18400, comments: 412, shares: 680,
                url: "https://instagram.com/p/g42-summit-bts-2026",
                sentiment: "Positive",
            },
            {
                author_name: "Space42", author_handle: "@space42_official", verified: true, author_is_g42: true,
                timestamp: "13h ago",
                content: "Our latest satellite render — constellation mission brief dropping next month. Stay tuned.",
                likes: 8200, comments: 178, shares: 240,
                url: "https://instagram.com/p/space42-constellation-reveal",
                sentiment: "Positive",
            },
            {
                author_name: "Abu Dhabi Media", author_handle: "@abudhabi_media", verified: true,
                timestamp: "16h ago",
                content: "A decade of transformation. A new chapter begins today. Full coverage of the Abu Dhabi AI Summit.",
                likes: 4600, comments: 94, shares: 320,
                url: "https://instagram.com/p/abudhabi-media-summit",
                sentiment: "Positive",
            },
        ],
        analyst_note: "Instagram volume lower than LinkedIn/X but engagement rate is highest across platforms (4.7%). Visual-first audience skews positive. No negative signal.",
    },

    youtube_coverage: {
        total_posts: 9,
        reach: "1.4M",
        engagement: "68K",
        sentiment_summary: "+0.62",
        posts: [
            {
                author_name: "G42 Group", author_handle: "@G42Group", verified: true, author_is_g42: true,
                timestamp: "4h ago",
                content: "FULL KEYNOTE: H.H. Sheikh Tahnoon bin Zayed Al Nahyan at the Abu Dhabi AI Summit 2026. 42-min address on sovereign AI, US-UAE cooperation, and the G42 decade roadmap.",
                likes: 32400, comments: 2800, views: 840000,
                url: "https://youtube.com/watch?v=tahnoon-keynote-2026",
                sentiment: "Positive",
            },
            {
                author_name: "Bloomberg Originals", author_handle: "@BloombergOriginals", verified: true,
                timestamp: "7h ago",
                content: "Inside G42: The Company Redefining UAE Tech. 18-min documentary feature with Peng Xiao interview and Abu Dhabi facility tour.",
                likes: 14200, comments: 1840, views: 420000,
                url: "https://youtube.com/watch?v=bloomberg-inside-g42",
                sentiment: "Positive",
            },
            {
                author_name: "CNBC International", author_handle: "@CNBCi", verified: true,
                timestamp: "12h ago",
                content: "G42's Next Chapter: Analysis from Abu Dhabi. 9-min CNBC segment on the Microsoft expansion and sovereign-cloud strategy.",
                likes: 6400, comments: 780, views: 180000,
                url: "https://youtube.com/watch?v=cnbc-g42-next-chapter",
                sentiment: "Neutral",
            },
        ],
        analyst_note: "YouTube is the high-signal, long-form channel today. Bloomberg and CNBC documentary pickups carry durable reach and will continue to accrue views for 5-7 days post-publish.",
    },

    news_coverage: {
        total_posts: 63,
        reach: "8.4M",
        engagement: "—",
        sentiment_summary: "+0.58",
        posts: [
            {
                author_name: "Reuters", author_handle: "reuters.com", verified: true,
                timestamp: "14:32 GST",
                content: "UAE's Sheikh Tahnoon pledges sovereign-AI push, deeper US ties at Abu Dhabi summit. Chairman of G42 outlines ambitions for UAE as global AI hub, emphasizing cooperation with US partners and sovereign compute capacity.",
                url: "https://reuters.com/technology/uae-sheikh-tahnoon-sovereign-ai-summit-2026-04-16",
                sentiment: "Positive",
            },
            {
                author_name: "Bloomberg", author_handle: "bloomberg.com", verified: true,
                timestamp: "15:10 GST",
                content: "Sheikh Tahnoon Outlines G42's Next Phase as UAE's AI Ambitions Crystallize. Keynote positions G42 at center of national strategy, signals long-horizon infrastructure commitment.",
                url: "https://bloomberg.com/news/articles/2026-04-16/uae-g42-sheikh-tahnoon-ai-summit",
                sentiment: "Positive",
            },
            {
                author_name: "Financial Times", author_handle: "ft.com", verified: true,
                timestamp: "16:05 GST",
                content: "G42 chairman maps out AI decade for Gulf. FT positions Sheikh Tahnoon's remarks as part of coherent Gulf strategy, with detailed sidebar on portfolio architecture.",
                url: "https://ft.com/content/uae-g42-tahnoon-ai-decade-gulf",
                sentiment: "Positive",
            },
            {
                author_name: "The Washington Post", author_handle: "washingtonpost.com", verified: true,
                timestamp: "13:15 GST",
                content: "How G42 Became Central to the US-UAE AI Relationship. Long-form feature on G42's role as anchor institution in cooperation agreement with White House quotes.",
                url: "https://washingtonpost.com/business/2026/04/16/g42-us-uae-ai/",
                sentiment: "Positive",
            },
        ],
        analyst_note: "Tier-1 editorial landscape is uniformly positive today. Reuters, Bloomberg, FT, and WaPo all framed G42 coverage around long-term strategic positioning rather than short-term news cycles. Strong signal of institutional press alignment.",
    },

    sentiment_analysis: {
        title: "Sentiment Deep Dive",
        description: "Cross-platform sentiment decomposition. Scoring combines outlet tier, unique reach, and engagement velocity. Weights updated daily against Meltwater baseline.",
        positive: 148, neutral: 79, negative: 20,
        overall_label: "Net Positive",
        by_platform: [
            { name: "LinkedIn", positive: 32, neutral: 9, negative: 1, volume: "42" },
            { name: "X (Twitter)", positive: 48, neutral: 22, negative: 8, volume: "78" },
            { name: "Instagram", positive: 16, neutral: 2, negative: 0, volume: "18" },
            { name: "YouTube", positive: 6, neutral: 3, negative: 0, volume: "9" },
            { name: "Tier-1 News", positive: 46, neutral: 15, negative: 2, volume: "63" },
            { name: "Regional (Gulf)", positive: 28, neutral: 8, negative: 1, volume: "37" },
        ],
        by_topic: [
            { name: "Sovereign AI / UAE Strategy", positive: 52, neutral: 14, negative: 2, volume: "68" },
            { name: "US-UAE Cooperation", positive: 38, neutral: 9, negative: 1, volume: "48" },
            { name: "Microsoft Partnership", positive: 26, neutral: 11, negative: 3, volume: "40" },
            { name: "Space42 / Constellation", positive: 22, neutral: 9, negative: 2, volume: "33" },
            { name: "Leadership Commentary", positive: 18, neutral: 8, negative: 4, volume: "30" },
            { name: "Policy / Regulation", positive: 8, neutral: 12, negative: 8, volume: "28" },
        ],
        trending_topics: [
            { topic: "#AbuDhabiAISummit", sentiment: "Positive", volume: "34K" },
            { topic: "#SovereignAI", sentiment: "Positive", volume: "18K" },
            { topic: "G42 Microsoft", sentiment: "Positive", volume: "12K" },
            { topic: "Space42 LEO", sentiment: "Positive", volume: "8.4K" },
            { topic: "Peng Xiao keynote", sentiment: "Positive", volume: "6.2K" },
            { topic: "UAE AI policy", sentiment: "Neutral", volume: "5.8K" },
        ],
        analyst_note: "Today's sentiment profile is strongly positive (+0.62 net, vs 30-day avg +0.41). Drivers: summit coverage, Microsoft expansion narrative, positive executive posture on LinkedIn/X. The only soft-negative cluster is in Policy/Regulation topic (−0.12), driven by three low-reach critical think-tank pieces. Recommendation: no crisis posture required; monitor policy cluster for 72h.",
    },

    // ═══════════════════════════════════════════════════════════════════
    // AI & POLICY LEADERS — external AI figures relevant to G42's landscape
    // ═══════════════════════════════════════════════════════════════════
    // Structure mirrors the agency report: one bucket per named person.
    // Agent populates `entities` dynamically based on mention detection.
    // Empty entities render with "No notable mentions."
    ai_policy_leaders: {
        title: "AI & Policy Leaders",
        description: "External AI figures, researchers, and policy leaders whose actions shape G42's competitive and regulatory landscape. Tracked daily across Tier-1 editorial.",
        entities: [
            {
                name: "Jensen Huang",
                mentions: [
                    {
                        outlet: "The Wall Street Journal",
                        title: "Trump Names Mark Zuckerberg, Larry Ellison and Jensen Huang to Tech Panel",
                        date: "April 15, 2026",
                        url: "https://wsj.com/politics/trump-tech-panel-zuckerberg-ellison-huang-2026-04-15",
                    },
                ],
            },
            {
                name: "Demis Hassabis",
                mentions: [
                    {
                        outlet: "Financial Times",
                        title: "The Infinity Machine — a deep dive into the mind of Demis Hassabis",
                        date: "April 15, 2026",
                        url: "https://ft.com/content/demis-hassabis-infinity-machine-deepmind",
                    },
                ],
            },
            {
                name: "Sam Altman",
                mentions: [],
            },
            {
                name: "Dario Amodei",
                mentions: [],
            },
            {
                name: "Sundar Pichai",
                mentions: [],
            },
        ],
    },

    // ═══════════════════════════════════════════════════════════════════
    // PARTNERS & PEERS — G42's ecosystem (includes cross-company buckets)
    // ═══════════════════════════════════════════════════════════════════
    partners_peers: {
        title: "Partners & Peers",
        description: "G42's ecosystem partners and peer companies across the AI landscape. Cross-company buckets appear when a story involves two or more (e.g. 'Microsoft and NVIDIA').",
        entities: [
            {
                name: "OpenAI",
                mentions: [
                    { outlet: "Bloomberg", title: "OpenAI Hires CEO of India's JioStar to Head Up Asia-Pacific", date: "April 15, 2026", url: "https://bloomberg.com/news/articles/2026-04-15/openai-hires-jiostar-ceo-asia-pacific" },
                    { outlet: "Financial Times", title: "OpenAI to end Disney deal and Sora video app", date: "April 15, 2026", url: "https://ft.com/content/openai-ends-disney-sora-deal" },
                    { outlet: "Reuters", title: "OpenAI's nonprofit arm names leaders, plans to spend at least $1 billion over next year", date: "April 14, 2026", url: "https://reuters.com/technology/openai-nonprofit-arm-names-leaders-billion-2026-04-14" },
                ],
            },
            {
                name: "Microsoft",
                mentions: [
                    { outlet: "Reuters", title: "Microsoft president says building data centres requires trust of US communities", date: "April 14, 2026", url: "https://reuters.com/technology/microsoft-data-centres-trust-us-communities-2026-04-14" },
                    { outlet: "Bloomberg", title: "Microsoft to Rent Texas Data Center Dropped by Oracle, OpenAI", date: "April 14, 2026", url: "https://bloomberg.com/news/articles/2026-04-14/microsoft-texas-data-center-oracle-openai" },
                ],
            },
            {
                name: "NVIDIA",
                mentions: [
                    { outlet: "Reuters", title: "US lawmakers ask whether Nvidia CEO's smuggling remarks misled regulators", date: "April 14, 2026", url: "https://reuters.com/technology/nvidia-smuggling-remarks-lawmakers-2026-04-14" },
                ],
            },
            {
                name: "AMD",
                mentions: [
                    { outlet: "The Information", title: "AMD-Backed Vultr Seeks $1 Billion for AI Cloud Push", date: "April 15, 2026", url: "https://theinformation.com/articles/amd-vultr-billion-ai-cloud-push" },
                ],
            },
            {
                name: "Amazon",
                mentions: [
                    { outlet: "Bloomberg", title: "Amazon Acquires Fauna Robotics, Entering Consumer Humanoid Market", date: "April 14, 2026", url: "https://bloomberg.com/news/articles/2026-04-14/amazon-acquires-fauna-robotics-humanoid" },
                ],
            },
            {
                name: "xAI",
                mentions: [
                    { outlet: "Reuters", title: "Baltimore sues Elon Musk's xAI over Grok sexual 'deepfakes'", date: "April 14, 2026", url: "https://reuters.com/technology/baltimore-sues-xai-grok-deepfakes-2026-04-14" },
                ],
            },
            // Cross-company buckets (when a story involves two entities)
            {
                name: "Microsoft and NVIDIA",
                mentions: [
                    { outlet: "Axios", title: "Microsoft and Nvidia team up on AI nuclear push", date: "April 14, 2026", url: "https://axios.com/2026/04/14/microsoft-nvidia-ai-nuclear-push" },
                ],
            },
            {
                name: "OpenAI and AWS",
                mentions: [
                    { outlet: "Reuters", title: "EU antitrust chief meets Google, Meta, OpenAI, Amazon CEOs amidst AI scrutiny", date: "April 14, 2026", url: "https://reuters.com/technology/eu-antitrust-meets-openai-amazon-ceos-2026-04-14" },
                ],
            },
            {
                name: "Google",
                mentions: [],
            },
            {
                name: "Anthropic",
                mentions: [],
            },
            {
                name: "Meta",
                mentions: [],
            },
        ],
    },

    // ═══════════════════════════════════════════════════════════════════
    // INDUSTRY NEWS — AI landscape, ESG, sector-specific
    // ═══════════════════════════════════════════════════════════════════
    industry_news: {
        title: "Industry News",
        description: "AI industry landscape, ESG / sustainability signal, and sector-specific AI adoption — the operating environment context around G42.",
        categories: [
            {
                name: "Artificial Intelligence",
                mentions: [
                    { outlet: "Bloomberg", title: "AI Demand Is Shielding China's Booming Trade From Iran War Shock", date: "April 15, 2026", url: "https://bloomberg.com/news/articles/2026-04-15/ai-demand-china-trade-iran-war" },
                ],
            },
            {
                name: "ESG / Sustainability",
                mentions: [],
            },
            {
                name: "AI in Sports",
                mentions: [
                    { outlet: "Sports Business Journal", title: "WSC Sports expands AI work with Cavaliers", date: "April 15, 2026", url: "https://sportsbusinessjournal.com/articles/wsc-sports-ai-cavaliers-2026-04-15" },
                ],
            },
            {
                name: "AI in Healthcare",
                mentions: [],
            },
            {
                name: "AI in Defense",
                mentions: [],
            },
        ],
    },

    // ═══════════════════════════════════════════════════════════════════
    // MARKETS — regional breakdown
    // ═══════════════════════════════════════════════════════════════════
    markets: {
        title: "Markets",
        description: "Regional coverage: USA, UAE, US-China Relations, Europe, India, GCC / Saudi Arabia, Africa. Geopolitics and market signals material to G42's operating footprint.",
        regions: [
            {
                name: "USA",
                mentions: [
                    { outlet: "The Wall Street Journal", title: "U.S. Government's Ban on Anthropic Looks Like Punishment, Judge Says", date: "April 14, 2026", url: "https://wsj.com/politics/us-government-anthropic-ban-punishment-2026-04-14" },
                    { outlet: "Reuters", title: "US will roll out pilot surveys to track energy use by data centers", date: "April 14, 2026", url: "https://reuters.com/sustainability/us-pilot-surveys-data-center-energy-2026-04-14" },
                    { outlet: "Bloomberg", title: "Trump's AI Advisers Urge Congress to Pass National Set of Rules", date: "April 14, 2026", url: "https://bloomberg.com/news/articles/2026-04-14/trump-ai-advisers-national-rules" },
                ],
            },
            {
                name: "UAE",
                mentions: [
                    { outlet: "The Wall Street Journal", title: "The U.A.E. Stands Up to Iran", date: "April 15, 2026", url: "https://wsj.com/world/middle-east/uae-stands-up-iran-2026-04-15" },
                    { outlet: "Gulf News", title: "UAE launches AI-powered platform to develop promising national talent", date: "April 14, 2026", url: "https://gulfnews.com/uae/government/uae-ai-platform-national-talent-2026-04-14" },
                ],
            },
            {
                name: "US-China Relations",
                mentions: [
                    { outlet: "The New York Times", title: "Trump Had His Eye on China, Then Plunged Into a New Mideast War", date: "April 15, 2026", url: "https://nytimes.com/2026/04/15/world/trump-china-mideast-war.html" },
                ],
            },
            {
                name: "Europe",
                mentions: [],
            },
            {
                name: "India",
                mentions: [],
            },
            {
                name: "GCC / Saudi Arabia",
                mentions: [],
            },
            {
                name: "Africa",
                mentions: [],
            },
        ],
    },

    // ═══════════════════════════════════════════════════════════════════
    // FEATURED IMAGERY — visual coverage drawn from Perplexity sources
    // ═══════════════════════════════════════════════════════════════════
    // Agent extracts image URLs + captions + outlet attribution from Perplexity
    // search results. First image becomes the hero; next 4 fill a 2×2 grid below.
    // Using Wikimedia Commons images for reliable demo rendering.
    featured_imagery: {
        title: "Featured Visual Coverage",
        description: "Key images drawn from Perplexity-retrieved Tier-1 editorial sources across the 7-day coverage window.",
        images: [
            {
                url: "https://images.unsplash.com/photo-1518684079-3c830dcef090?w=1200&auto=format&fit=crop&q=75",
                caption: "Abu Dhabi skyline — G42 announces Cisco-backed AI cluster expansion",
                outlet: "Reuters",
                date: "April 14, 2026",
                meta: "~1.2M impressions · Tier 1",
                source_url: "https://reuters.com/technology/uae-g42-cisco-expansion-2026-04-14",
            },
            {
                url: "https://images.unsplash.com/photo-1591808216268-ce0b82787efe?w=960&auto=format&fit=crop&q=75",
                caption: "AMD MI350X GPUs deployed in G42's Regulated Technology Environment",
                outlet: "Bloomberg",
                date: "April 13, 2026",
                source_url: "https://bloomberg.com/news/articles/2026-04-13/amd-g42-mi350x",
            },
            {
                url: "https://images.unsplash.com/photo-1591696205602-2f950c417cb9?w=960&auto=format&fit=crop&q=75",
                caption: "Cisco powers G42 end-to-end AI infrastructure under US-UAE partnership",
                outlet: "Cisco IR",
                date: "April 13, 2026",
                source_url: "https://investor.cisco.com/news/news-details/2026/Cisco-G42-Partnership",
            },
            {
                url: "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=960&auto=format&fit=crop&q=75",
                caption: "Cerebras 8-exaflop compute arrives in India via G42 partnership",
                outlet: "Yahoo Finance",
                date: "April 12, 2026",
                source_url: "https://finance.yahoo.com/news/uae-g42-teams-cerebras-india-deploy",
            },
            {
                url: "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=960&auto=format&fit=crop&q=75",
                caption: "Sheikh Tahnoon's IHC acquires majority stake in The Ivy hospitality group",
                outlet: "Financial Times",
                date: "April 14, 2026",
                source_url: "https://ft.com/content/ihc-richard-caring-ivy-acquisition-2026",
            },
        ],
    },

    sources: [
        { title: "Reuters", text: "UAE's Sheikh Tahnoon pledges sovereign-AI push, deeper US ties at Abu Dhabi summit", url: "https://reuters.com/technology/uae-sheikh-tahnoon-sovereign-ai-summit-2026-04-16" },
        { title: "Bloomberg", text: "Sheikh Tahnoon Outlines G42's Next Phase as UAE's AI Ambitions Crystallize", url: "https://bloomberg.com/news/articles/2026-04-16/uae-g42-sheikh-tahnoon-ai-summit" },
        { title: "Financial Times", text: "G42 chairman maps out AI decade for Gulf", url: "https://ft.com/content/uae-g42-tahnoon-ai-decade-gulf" },
        { title: "The Washington Post", text: "How G42 Became Central to the US-UAE AI Relationship", url: "https://washingtonpost.com/business/2026/04/16/g42-us-uae-ai/" },
        { title: "The Wall Street Journal", text: "G42's Microsoft Partnership Enters New Phase With $1.5B Expansion Plan", url: "https://wsj.com/articles/g42-microsoft-partnership-expansion-2026" },
        { title: "BBC News", text: "G42: The Emirati AI Group Redrawing Global Tech Maps", url: "https://bbc.com/news/business-g42-emirati-ai-redrawing-maps" },
        { title: "The Information", text: "Peng Xiao on why G42 picked Microsoft over alternatives", url: "https://theinformation.com/articles/peng-xiao-g42-microsoft" },
        { title: "Gulf News", text: "G42 Healthcare partners with Cleveland Clinic Abu Dhabi (Alkaissi)", url: "https://gulfnews.com/business/g42-healthcare-cleveland-clinic" },
        { title: "SpaceNews", text: "Space42 announces Leo constellation partnership", url: "https://spacenews.com/space42-leo-constellation-partnership" },
        { title: "Defense One", text: "UAE's Space42 expands ISR footprint", url: "https://defenseone.com/technology/2026/space42-isr-expansion" },
        { title: "LinkedIn — G42 Group", text: "Official G42 and executive posts (15+ posts in window)", url: "https://linkedin.com/company/g42" },
        { title: "X — @G42ai", text: "Official G42 account and executive handles (78 posts)", url: "https://x.com/G42ai" },
        { title: "Instagram — @g42.ai", text: "G42 Official channel and affiliate accounts", url: "https://instagram.com/g42.ai" },
        { title: "YouTube — G42 Group", text: "Keynote full recording and Bloomberg/CNBC features", url: "https://youtube.com/@G42Group" },
    ],

    methodology: "Sources ingested via automated scrape across Meltwater wire + native platform APIs. Mentions deduplicated, entity-linked, sentiment-scored via multi-model ensemble (BERT-NLI + domain-tuned Qwen). Outlet tier weights: Tier-1 = 1.0, Tier-2 = 0.6, Tier-3 = 0.3. Reach and engagement surfaced from platform APIs where available; estimated otherwise. Human-in-loop review: zero interventions during this run.",
};

(async () => {
    console.log(`\n[TEST] Starting G42 report test — session: ${SESSION}\n`);

    try {
        console.log('1. Starting session...');
        let r = await req('POST', '/g42-report/start', { sessionId: SESSION });
        console.log('   →', r.status, r.body.message);

        console.log('2. Uploading report data (single update)...');
        r = await req('POST', '/g42-report/update', reportData);
        console.log('   →', r.status, 'Updated keys:', (r.body.updatedKeys || []).length);

        console.log('3. Generating PDF (this takes ~10-20s)...');
        r = await req('POST', '/g42-report/generate', { sessionId: SESSION });
        console.log('   →', r.status);
        if (r.body.url) {
            console.log('\n✓ PDF GENERATED:');
            console.log('   URL:      ', r.body.url);
            console.log('   Pages:    ', r.body.pages);
            console.log('   PDF ID:   ', r.body.pdfId);
            console.log('   Permanent:', r.body.permanent);
            console.log('\n   Open in browser:', r.body.url);
        } else {
            console.log('   Error:', r.body);
        }
    } catch (e) {
        console.error('TEST ERROR:', e.message);
        process.exit(1);
    }
})();

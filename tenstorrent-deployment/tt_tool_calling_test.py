#!/usr/bin/env python3
"""
AIREV 50-Query Tool Calling Test on Tenstorrent Wormhole
Tests the fine-tuned Qwen 3.5-0.8B model on real OnDemand plugin scenarios.
Proves: correct tool selection, valid JSON output, hardware attestation.
"""
import requests, json, time, sys, os
from datetime import datetime

ENDPOINT = os.environ.get("TT_ENDPOINT", "http://localhost:8600")

# 50 test queries with expected plugin selections
TEST_QUERIES = [
    # 1-10: Single plugin, simple tasks
    {"plugins": [{"name": "perplexity", "pluginId": "plugin-1722260873", "description": "Search the web for real-time information"}],
     "query": "Search for latest AI news in UAE", "expected_plugin": "plugin-1722260873"},
    {"plugins": [{"name": "gmail", "pluginId": "plugin-4827391", "description": "Send and read emails"}],
     "query": "Send an email to john@example.com about the meeting", "expected_plugin": "plugin-4827391"},
    {"plugins": [{"name": "weather_api", "pluginId": "plugin-9382715", "description": "Get weather forecasts for any location"}],
     "query": "What is the weather in Dubai tomorrow", "expected_plugin": "plugin-9382715"},
    {"plugins": [{"name": "calendar", "pluginId": "plugin-1928374", "description": "Manage calendar events and meetings"}],
     "query": "Schedule a meeting for next Monday at 10am", "expected_plugin": "plugin-1928374"},
    {"plugins": [{"name": "slack", "pluginId": "plugin-5738291", "description": "Send messages to Slack channels"}],
     "query": "Post a message in the engineering channel about the deployment", "expected_plugin": "plugin-5738291"},
    {"plugins": [{"name": "jira", "pluginId": "plugin-8273645", "description": "Create and manage Jira tickets"}],
     "query": "Create a bug ticket for the login page issue", "expected_plugin": "plugin-8273645"},
    {"plugins": [{"name": "notion", "pluginId": "plugin-3928174", "description": "Create and edit Notion pages and databases"}],
     "query": "Create a new page in Notion for the project roadmap", "expected_plugin": "plugin-3928174"},
    {"plugins": [{"name": "github", "pluginId": "plugin-6182734", "description": "Manage GitHub repos, issues, and PRs"}],
     "query": "List open pull requests on the main repo", "expected_plugin": "plugin-6182734"},
    {"plugins": [{"name": "google_drive", "pluginId": "plugin-2847362", "description": "Upload, download, and manage files in Google Drive"}],
     "query": "Upload the quarterly report to the shared drive", "expected_plugin": "plugin-2847362"},
    {"plugins": [{"name": "twitter", "pluginId": "plugin-7392841", "description": "Post tweets and read timeline"}],
     "query": "Post a tweet about our new product launch", "expected_plugin": "plugin-7392841"},

    # 11-20: Multi-plugin selection (2-3 plugins available, pick the right one)
    {"plugins": [
        {"name": "weather_api", "pluginId": "plugin-9382715", "description": "Get weather forecasts"},
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send emails"},
        {"name": "calendar", "pluginId": "plugin-1928374", "description": "Manage calendar"}
    ], "query": "What will the weather be like in Abu Dhabi this weekend", "expected_plugin": "plugin-9382715"},
    {"plugins": [
        {"name": "perplexity", "pluginId": "plugin-1722260873", "description": "Search the web"},
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send emails"},
    ], "query": "Search for the latest SpaceX launch details", "expected_plugin": "plugin-1722260873"},
    {"plugins": [
        {"name": "slack", "pluginId": "plugin-5738291", "description": "Send Slack messages"},
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send emails"},
        {"name": "calendar", "pluginId": "plugin-1928374", "description": "Manage calendar"}
    ], "query": "Send a Slack message to the design team about the mockups", "expected_plugin": "plugin-5738291"},
    {"plugins": [
        {"name": "jira", "pluginId": "plugin-8273645", "description": "Manage Jira tickets"},
        {"name": "github", "pluginId": "plugin-6182734", "description": "Manage GitHub repos"},
    ], "query": "Create a Jira task for implementing the new API endpoint", "expected_plugin": "plugin-8273645"},
    {"plugins": [
        {"name": "notion", "pluginId": "plugin-3928174", "description": "Create Notion pages"},
        {"name": "google_drive", "pluginId": "plugin-2847362", "description": "Manage Google Drive files"},
    ], "query": "Create a new Notion wiki page for onboarding documentation", "expected_plugin": "plugin-3928174"},
    {"plugins": [
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send and read emails"},
        {"name": "twitter", "pluginId": "plugin-7392841", "description": "Post tweets"},
    ], "query": "Send an email to the marketing team with the campaign results", "expected_plugin": "plugin-4827391"},
    {"plugins": [
        {"name": "calendar", "pluginId": "plugin-1928374", "description": "Manage calendar events"},
        {"name": "slack", "pluginId": "plugin-5738291", "description": "Send Slack messages"},
    ], "query": "Book a 30 minute meeting with Sarah for Wednesday afternoon", "expected_plugin": "plugin-1928374"},
    {"plugins": [
        {"name": "github", "pluginId": "plugin-6182734", "description": "Manage GitHub repos and PRs"},
        {"name": "jira", "pluginId": "plugin-8273645", "description": "Create Jira tickets"},
    ], "query": "Open a new GitHub issue for the memory leak in the dashboard", "expected_plugin": "plugin-6182734"},
    {"plugins": [
        {"name": "google_drive", "pluginId": "plugin-2847362", "description": "Manage Google Drive"},
        {"name": "notion", "pluginId": "plugin-3928174", "description": "Create Notion pages"},
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send emails"},
    ], "query": "Download the budget spreadsheet from Drive", "expected_plugin": "plugin-2847362"},
    {"plugins": [
        {"name": "perplexity", "pluginId": "plugin-1722260873", "description": "Search the web"},
        {"name": "weather_api", "pluginId": "plugin-9382715", "description": "Get weather"},
    ], "query": "Look up the current stock price of NVIDIA", "expected_plugin": "plugin-1722260873"},

    # 21-30: Multi-step tasks (should select multiple plugins)
    {"plugins": [
        {"name": "weather_api", "pluginId": "plugin-9382715", "description": "Get weather forecasts"},
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send emails"},
    ], "query": "Check the weather in Dubai and email the forecast to my team", "expected_plugin": "plugin-9382715,plugin-4827391"},
    {"plugins": [
        {"name": "perplexity", "pluginId": "plugin-1722260873", "description": "Search the web"},
        {"name": "slack", "pluginId": "plugin-5738291", "description": "Send Slack messages"},
    ], "query": "Research the latest React updates and share in the dev channel", "expected_plugin": "plugin-1722260873,plugin-5738291"},
    {"plugins": [
        {"name": "jira", "pluginId": "plugin-8273645", "description": "Manage Jira tickets"},
        {"name": "slack", "pluginId": "plugin-5738291", "description": "Send Slack messages"},
    ], "query": "Create a Jira ticket for the bug and notify the team on Slack", "expected_plugin": "plugin-8273645,plugin-5738291"},
    {"plugins": [
        {"name": "github", "pluginId": "plugin-6182734", "description": "Manage GitHub"},
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send emails"},
    ], "query": "Check recent PRs on the repo and email a summary to the manager", "expected_plugin": "plugin-6182734,plugin-4827391"},
    {"plugins": [
        {"name": "calendar", "pluginId": "plugin-1928374", "description": "Manage calendar"},
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send emails"},
    ], "query": "Schedule a team standup for tomorrow and send invites via email", "expected_plugin": "plugin-1928374,plugin-4827391"},
    {"plugins": [
        {"name": "notion", "pluginId": "plugin-3928174", "description": "Create Notion pages"},
        {"name": "slack", "pluginId": "plugin-5738291", "description": "Send Slack messages"},
    ], "query": "Create meeting notes in Notion and share the link on Slack", "expected_plugin": "plugin-3928174,plugin-5738291"},
    {"plugins": [
        {"name": "perplexity", "pluginId": "plugin-1722260873", "description": "Search the web"},
        {"name": "notion", "pluginId": "plugin-3928174", "description": "Create Notion pages"},
    ], "query": "Research competitor pricing and document findings in Notion", "expected_plugin": "plugin-1722260873,plugin-3928174"},
    {"plugins": [
        {"name": "google_drive", "pluginId": "plugin-2847362", "description": "Manage Google Drive"},
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send emails"},
    ], "query": "Download the presentation from Drive and email it to the client", "expected_plugin": "plugin-2847362,plugin-4827391"},
    {"plugins": [
        {"name": "weather_api", "pluginId": "plugin-9382715", "description": "Get weather"},
        {"name": "calendar", "pluginId": "plugin-1928374", "description": "Manage calendar"},
    ], "query": "Check if it will rain tomorrow and reschedule outdoor events if needed", "expected_plugin": "plugin-9382715,plugin-1928374"},
    {"plugins": [
        {"name": "twitter", "pluginId": "plugin-7392841", "description": "Post tweets"},
        {"name": "perplexity", "pluginId": "plugin-1722260873", "description": "Search the web"},
    ], "query": "Find trending tech topics and draft a tweet about the top story", "expected_plugin": "plugin-1722260873,plugin-7392841"},

    # 31-40: Edge cases — unusual queries, varied phrasing
    {"plugins": [{"name": "perplexity", "pluginId": "plugin-1722260873", "description": "Search the web for real-time information"}],
     "query": "Who won the Champions League final", "expected_plugin": "plugin-1722260873"},
    {"plugins": [{"name": "gmail", "pluginId": "plugin-4827391", "description": "Send and read emails"}],
     "query": "Draft a thank you email to the vendor for the quick delivery", "expected_plugin": "plugin-4827391"},
    {"plugins": [{"name": "slack", "pluginId": "plugin-5738291", "description": "Send messages to Slack channels"}],
     "query": "Let the ops team know the servers are back online", "expected_plugin": "plugin-5738291"},
    {"plugins": [{"name": "jira", "pluginId": "plugin-8273645", "description": "Create and manage Jira tickets"}],
     "query": "Log a feature request for dark mode support", "expected_plugin": "plugin-8273645"},
    {"plugins": [{"name": "calendar", "pluginId": "plugin-1928374", "description": "Manage calendar events and meetings"}],
     "query": "Block off Friday afternoon for deep work", "expected_plugin": "plugin-1928374"},
    {"plugins": [{"name": "github", "pluginId": "plugin-6182734", "description": "Manage GitHub repos, issues, and PRs"}],
     "query": "Check if the CI pipeline passed on the latest commit", "expected_plugin": "plugin-6182734"},
    {"plugins": [{"name": "notion", "pluginId": "plugin-3928174", "description": "Create and edit Notion pages"}],
     "query": "Add a new section to the product spec for the mobile app", "expected_plugin": "plugin-3928174"},
    {"plugins": [{"name": "google_drive", "pluginId": "plugin-2847362", "description": "Manage files in Google Drive"}],
     "query": "Share the design mockups folder with the frontend team", "expected_plugin": "plugin-2847362"},
    {"plugins": [{"name": "weather_api", "pluginId": "plugin-9382715", "description": "Get weather forecasts for any location"}],
     "query": "Is it going to snow in New York this week", "expected_plugin": "plugin-9382715"},
    {"plugins": [{"name": "twitter", "pluginId": "plugin-7392841", "description": "Post tweets and read timeline"}],
     "query": "Share our Series A announcement on Twitter", "expected_plugin": "plugin-7392841"},

    # 41-50: Production-like real plugin names
    {"plugins": [
        {"name": "openai_dalle", "pluginId": "plugin-3847291", "description": "Generate images using DALL-E"},
        {"name": "stable_diffusion", "pluginId": "plugin-9182736", "description": "Generate images with Stable Diffusion"},
    ], "query": "Generate a logo concept for our new AI product", "expected_plugin": "plugin-3847291"},
    {"plugins": [
        {"name": "stripe", "pluginId": "plugin-2738491", "description": "Process payments and manage subscriptions"},
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send emails"},
    ], "query": "Check the status of the latest payment from Acme Corp", "expected_plugin": "plugin-2738491"},
    {"plugins": [
        {"name": "hubspot", "pluginId": "plugin-8374612", "description": "Manage CRM contacts and deals"},
        {"name": "gmail", "pluginId": "plugin-4827391", "description": "Send emails"},
    ], "query": "Add a new lead to the CRM from today's conference", "expected_plugin": "plugin-8374612"},
    {"plugins": [
        {"name": "youtube", "pluginId": "plugin-6283741", "description": "Search and manage YouTube videos"},
        {"name": "perplexity", "pluginId": "plugin-1722260873", "description": "Search the web"},
    ], "query": "Find the best tutorial videos on Kubernetes deployment", "expected_plugin": "plugin-6283741"},
    {"plugins": [
        {"name": "aws_s3", "pluginId": "plugin-4927381", "description": "Manage AWS S3 buckets and objects"},
        {"name": "google_drive", "pluginId": "plugin-2847362", "description": "Manage Google Drive files"},
    ], "query": "Upload the backup files to our S3 bucket", "expected_plugin": "plugin-4927381"},
    {"plugins": [
        {"name": "datadog", "pluginId": "plugin-7293841", "description": "Monitor application metrics and alerts"},
        {"name": "slack", "pluginId": "plugin-5738291", "description": "Send Slack messages"},
    ], "query": "Check if there are any active alerts on the production dashboard", "expected_plugin": "plugin-7293841"},
    {"plugins": [
        {"name": "confluence", "pluginId": "plugin-3829174", "description": "Create and edit Confluence wiki pages"},
        {"name": "notion", "pluginId": "plugin-3928174", "description": "Create Notion pages"},
    ], "query": "Update the architecture decision record in Confluence", "expected_plugin": "plugin-3829174"},
    {"plugins": [
        {"name": "zapier", "pluginId": "plugin-5928374", "description": "Create and manage automation workflows"},
        {"name": "jira", "pluginId": "plugin-8273645", "description": "Manage Jira tickets"},
    ], "query": "Set up an automation to notify me when high-priority tickets are created", "expected_plugin": "plugin-5928374"},
    {"plugins": [
        {"name": "figma", "pluginId": "plugin-8392741", "description": "Access and manage Figma design files"},
        {"name": "slack", "pluginId": "plugin-5738291", "description": "Send Slack messages"},
    ], "query": "Get the latest version of the homepage design from Figma", "expected_plugin": "plugin-8392741"},
    {"plugins": [
        {"name": "linear", "pluginId": "plugin-2938471", "description": "Manage Linear issues and projects"},
        {"name": "github", "pluginId": "plugin-6182734", "description": "Manage GitHub repos"},
    ], "query": "Create a Linear issue for the mobile responsive fix", "expected_plugin": "plugin-2938471"},
]


def run_test(idx, test_case):
    """Run a single test query and return results."""
    plugins_str = json.dumps(test_case["plugins"])
    system_msg = (
        "You are an AI agent orchestrator. Given available plugins, return a JSON execution plan.\n\n"
        "Available Plugins:\n%s\n\n"
        "Respond with a JSON array of steps." % plugins_str
    )

    try:
        start = time.time()
        r = requests.post(ENDPOINT + "/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": test_case["query"]}
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }, timeout=120)
        elapsed = time.time() - start

        if r.status_code != 200:
            return {"idx": idx, "status": "ERROR", "error": "HTTP %d" % r.status_code,
                    "time": elapsed, "tokens": 0, "tps": 0}

        d = r.json()
        content = d["choices"][0]["message"]["content"]
        perf = d.get("performance", {})
        hw = d.get("hardware", "unknown")
        tokens = perf.get("tokens_generated", 0)
        tps = perf.get("tokens_per_second", 0)

        # Check valid JSON
        valid_json = False
        plugin_ids_found = []
        try:
            parsed = json.loads(content)
            valid_json = True
            if isinstance(parsed, dict) and "plugins" in parsed:
                for p in parsed.get("plugins", []):
                    if "pluginId" in p:
                        plugin_ids_found.append(p["pluginId"])
            elif isinstance(parsed, list):
                for p in parsed:
                    if isinstance(p, dict) and "pluginId" in p:
                        plugin_ids_found.append(p["pluginId"])
        except:
            import re
            m = re.search(r'[\[{].*[}\]]', content, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    valid_json = True
                    if isinstance(parsed, dict) and "plugins" in parsed:
                        for p in parsed.get("plugins", []):
                            if "pluginId" in p:
                                plugin_ids_found.append(p["pluginId"])
                    elif isinstance(parsed, list):
                        for p in parsed:
                            if isinstance(p, dict) and "pluginId" in p:
                                plugin_ids_found.append(p["pluginId"])
                except:
                    pass

        # Check plugin selection
        expected = set(test_case["expected_plugin"].split(","))
        found = set(plugin_ids_found)
        correct_plugin = bool(expected & found)  # At least one expected plugin found

        return {
            "idx": idx + 1,
            "query": test_case["query"][:50],
            "status": "OK",
            "valid_json": valid_json,
            "correct_plugin": correct_plugin,
            "expected": test_case["expected_plugin"],
            "found": ",".join(plugin_ids_found) if plugin_ids_found else "none",
            "tokens": tokens,
            "tps": tps,
            "time": round(elapsed, 2),
            "hardware": hw,
            "output_preview": content[:100],
        }

    except Exception as e:
        return {"idx": idx + 1, "status": "ERROR", "error": str(e)[:100],
                "time": 0, "tokens": 0, "tps": 0}


def main():
    print("=" * 80)
    print("AIREV Tool Calling Test — 50 Queries on Tenstorrent Wormhole")
    print("=" * 80)
    print("Endpoint: %s" % ENDPOINT)
    print("Start: %s" % datetime.now().isoformat())
    print()

    # Get hardware info
    try:
        health = requests.get(ENDPOINT + "/health", timeout=10).json()
        print("Hardware: %s" % health.get("hardware", "unknown"))
        print("Model: %s" % health.get("model", "unknown"))
        print("Backend: %s" % health.get("backend", health.get("acceleration", "unknown")))
        print()
    except:
        print("Could not fetch health info")

    results = []
    total_start = time.time()

    for i, tc in enumerate(TEST_QUERIES):
        result = run_test(i, tc)
        results.append(result)
        status_icon = "PASS" if result.get("valid_json") and result.get("correct_plugin") else "FAIL" if result.get("status") == "OK" else "ERR"
        print("[%02d/50] %s | %-50s | JSON:%s Plugin:%s | %.1fs | %.1f tok/s" % (
            i + 1, status_icon,
            tc["query"][:50],
            "Y" if result.get("valid_json") else "N",
            "Y" if result.get("correct_plugin") else "N",
            result.get("time", 0),
            result.get("tps", 0),
        ))

    total_time = time.time() - total_start

    # Summary
    ok_results = [r for r in results if r.get("status") == "OK"]
    valid_json_count = sum(1 for r in ok_results if r.get("valid_json"))
    correct_plugin_count = sum(1 for r in ok_results if r.get("correct_plugin"))
    avg_tps = sum(r.get("tps", 0) for r in ok_results) / len(ok_results) if ok_results else 0
    avg_time = sum(r.get("time", 0) for r in ok_results) / len(ok_results) if ok_results else 0
    total_tokens = sum(r.get("tokens", 0) for r in ok_results)
    errors = sum(1 for r in results if r.get("status") != "OK")

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("Total queries:     %d" % len(results))
    print("Successful:        %d" % len(ok_results))
    print("Errors:            %d" % errors)
    print()
    print("Valid JSON:        %d/%d (%.1f%%)" % (valid_json_count, len(ok_results), 100*valid_json_count/len(ok_results) if ok_results else 0))
    print("Correct Plugin:    %d/%d (%.1f%%)" % (correct_plugin_count, len(ok_results), 100*correct_plugin_count/len(ok_results) if ok_results else 0))
    print()
    print("Avg tokens/query:  %.1f" % (total_tokens / len(ok_results) if ok_results else 0))
    print("Avg tok/s:         %.1f" % avg_tps)
    print("Avg latency:       %.2fs" % avg_time)
    print("Total tokens:      %d" % total_tokens)
    print("Total time:        %.1fs" % total_time)
    print()
    print("Hardware:          %s" % (ok_results[0].get("hardware", "unknown") if ok_results else "unknown"))
    print("End:               %s" % datetime.now().isoformat())

    # Save report
    report = {
        "test_name": "AIREV 50-Query Tool Calling Test",
        "hardware": ok_results[0].get("hardware", "unknown") if ok_results else "unknown",
        "model": "qwen-0.8b-agentjson-grpo-v9-ext",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_queries": len(results),
            "successful": len(ok_results),
            "errors": errors,
            "valid_json_pct": round(100*valid_json_count/len(ok_results), 1) if ok_results else 0,
            "correct_plugin_pct": round(100*correct_plugin_count/len(ok_results), 1) if ok_results else 0,
            "avg_tokens_per_second": round(avg_tps, 1),
            "avg_latency_seconds": round(avg_time, 2),
            "total_tokens": total_tokens,
            "total_time_seconds": round(total_time, 1),
        },
        "results": results,
    }

    report_path = os.environ.get("REPORT_PATH", "tt_test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\nReport saved: %s" % report_path)


if __name__ == "__main__":
    main()

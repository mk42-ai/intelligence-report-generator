#!/usr/bin/env python3
"""
Tenstorrent Wormhole Hardware Attestation Report
Proves the model is running on Tenstorrent hardware with device diagnostics.
"""
import subprocess, json, os, platform, time
from datetime import datetime


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=30).decode().strip()
    except:
        return "N/A"


def get_tt_device_info():
    """Get Tenstorrent device info via tt-smi."""
    info = {}
    info["tt_smi_version"] = run("tt-smi --version 2>/dev/null || echo N/A")

    # Parse tt-smi output
    smi_output = run("tt-smi 2>/dev/null")
    info["tt_smi_raw"] = smi_output[:2000] if smi_output != "N/A" else "N/A"

    # Device files
    info["device_files"] = run("ls -la /dev/tenstorrent/ 2>/dev/null")

    # Kernel module
    info["kernel_module"] = run("lsmod | grep tenstorrent 2>/dev/null")

    # PCI devices
    info["pci_devices"] = run("lspci | grep -i tenstorrent 2>/dev/null")

    # Firmware
    info["firmware"] = run("tt-smi -f 2>/dev/null | head -20")

    # Board info
    info["board_type"] = run("tt-smi -b 2>/dev/null | head -10")

    return info


def get_system_info():
    """Get system information."""
    return {
        "hostname": platform.node(),
        "os": run("cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2"),
        "kernel": platform.release(),
        "cpu": run("lscpu | grep 'Model name' | cut -d: -f2 | xargs"),
        "memory_total": run("free -h | grep Mem | awk '{print $2}'"),
        "memory_available": run("free -h | grep Mem | awk '{print $7}'"),
        "disk": run("df -h / | tail -1 | awk '{print $2, $3, $4, $5}'"),
        "uptime": run("uptime -p"),
    }


def get_inference_proof(endpoint="http://localhost:8600"):
    """Run inference and capture hardware attestation."""
    import requests
    proofs = []

    # Health check
    try:
        r = requests.get(endpoint + "/health", timeout=10)
        proofs.append({"type": "health_check", "data": r.json()})
    except Exception as e:
        proofs.append({"type": "health_check", "error": str(e)})

    # Model info
    try:
        r = requests.get(endpoint + "/v1/models", timeout=10)
        proofs.append({"type": "model_info", "data": r.json()})
    except Exception as e:
        proofs.append({"type": "model_info", "error": str(e)})

    # Inference test with hardware field
    try:
        start = time.time()
        r = requests.post(endpoint + "/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": "You are an AI assistant. Say hello."},
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7,
            "max_tokens": 50
        }, timeout=120)
        elapsed = time.time() - start
        data = r.json()
        proofs.append({
            "type": "inference_test",
            "hardware": data.get("hardware"),
            "performance": data.get("performance"),
            "output": data["choices"][0]["message"]["content"][:200],
            "wall_time": round(elapsed, 2),
        })
    except Exception as e:
        proofs.append({"type": "inference_test", "error": str(e)})

    # Tool calling test
    try:
        start = time.time()
        r = requests.post(endpoint + "/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": "You are an AI agent orchestrator. Given available plugins, return a JSON execution plan.\n\nAvailable Plugins:\n[{\"name\": \"perplexity\", \"pluginId\": \"plugin-1722260873\", \"description\": \"Search the web\"}]\n\nRespond with a JSON array of steps."},
                {"role": "user", "content": "Search for AI news"}
            ],
            "temperature": 0.3,
            "max_tokens": 200
        }, timeout=120)
        elapsed = time.time() - start
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            valid_json = True
        except:
            valid_json = False

        proofs.append({
            "type": "tool_calling_test",
            "hardware": data.get("hardware"),
            "performance": data.get("performance"),
            "output": content[:300],
            "valid_json": valid_json,
            "wall_time": round(elapsed, 2),
        })
    except Exception as e:
        proofs.append({"type": "tool_calling_test", "error": str(e)})

    return proofs


def generate_report():
    """Generate complete hardware attestation report."""
    print("=" * 70)
    print("TENSTORRENT WORMHOLE HARDWARE ATTESTATION REPORT")
    print("=" * 70)
    print("Generated:", datetime.now().isoformat())
    print()

    report = {
        "report_type": "Tenstorrent Wormhole Hardware Attestation",
        "generated_at": datetime.now().isoformat(),
        "system": get_system_info(),
        "tenstorrent": get_tt_device_info(),
        "inference_proofs": get_inference_proof(),
    }

    # Print summary
    sys = report["system"]
    print("SYSTEM:")
    print("  Hostname: %s" % sys["hostname"])
    print("  OS: %s" % sys["os"])
    print("  CPU: %s" % sys["cpu"])
    print("  Memory: %s (available: %s)" % (sys["memory_total"], sys["memory_available"]))
    print()

    tt = report["tenstorrent"]
    print("TENSTORRENT HARDWARE:")
    print("  Device files: %s" % ("PRESENT" if tt["device_files"] != "N/A" else "NOT FOUND"))
    print("  Kernel module: %s" % ("LOADED" if tt["kernel_module"] != "N/A" else "NOT FOUND"))
    print("  PCI devices: %s" % tt["pci_devices"][:200])
    print()

    print("INFERENCE PROOFS:")
    for proof in report["inference_proofs"]:
        print("  [%s] hardware=%s" % (
            proof["type"],
            proof.get("hardware", proof.get("error", "N/A"))
        ))
        if "performance" in proof:
            perf = proof["performance"]
            print("    tokens=%s, speed=%s tok/s" % (
                perf.get("tokens_generated", "?"),
                perf.get("tokens_per_second", "?"),
            ))
        if "valid_json" in proof:
            print("    valid_json=%s" % proof["valid_json"])
    print()

    # Save
    report_path = os.environ.get("REPORT_PATH", "tt_hardware_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("Report saved: %s" % report_path)

    return report


if __name__ == "__main__":
    generate_report()

"""
src/llm_agent.py - Gemini-Powered OS Root Cause Analysis (RCA) & Remediation Agent
Translates ML bottleneck classifications and raw telemetry vectors into actionable sysadmin runbooks.
Includes real-time terminal telemetry logging, model status telemetry, and latency tracking.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import yaml

load_dotenv()


class BottleneckDiagnosisReport(BaseModel):
    severity_level: str = Field(
        default="MEDIUM",
        description="Severity classification: LOW, MEDIUM, HIGH, or CRITICAL.",
    )
    root_cause: str = Field(
        default="Operating system performance bottleneck detected.",
        description="1-2 sentence precise technical description of kernel/hardware saturation.",
    )
    technical_analysis: str = Field(
        default="Telemetry indicators show elevated resource pressure across system subsystems.",
        description="2-3 sentence explanation directly linking the metrics to the active bottleneck.",
    )
    investigation_commands: List[str] = Field(
        default_factory=lambda: ["top -b -n 1", "vmstat 1 5", "iostat -xz 1 3"],
        description="Targeted CLI inspection commands.",
    )
    remediation_commands: List[str] = Field(
        default_factory=lambda: ["# Inspect high resource processes."],
        description="Actionable shell commands to relieve the condition.",
    )


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


CONFIG = load_config()
LLM_CFG = CONFIG.get("llm", {})
DEFAULT_MODEL_NAME = LLM_CFG.get("model_name", "gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(LLM_CFG.get("temperature", 0.2))
DEFAULT_MAX_OUTPUT_TOKENS = int(LLM_CFG.get("max_output_tokens", 1000))


def _clean_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    cleaned_key = api_key.strip().strip("'\"")
    if not cleaned_key or cleaned_key == "your_actual_gemini_api_key_here":
        return None

    from google import genai
    return genai.Client(api_key=cleaned_key)

def get_agent_status() -> Dict[str, Any]:
    """Inspects environment and reports live LLM client operational status."""
    api_key = os.getenv("GEMINI_API_KEY")
    client = get_gemini_client()
    return {
        "api_key_configured": bool(api_key and api_key != "your_actual_gemini_api_key_here"),
        "client_initialized": client is not None,
        "active_model": DEFAULT_MODEL_NAME,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "status_badge": "ONLINE (Gemini)" if client is not None else "OFFLINE (Heuristic Mode)"
    }

def get_agent_status() -> Dict[str, Any]:
    """Inspects environment and reports live LLM client operational status."""
    api_key = os.getenv("GEMINI_API_KEY")
    client = get_gemini_client()
    is_ready = bool(client is not None)
    
    return {
        # Keys expected by app.py
        "ready": is_ready,
        "model": DEFAULT_MODEL_NAME,
        "status": "ready" if is_ready else "fallback",
        
        # Extended metadata keys
        "api_key_configured": bool(api_key and api_key != "your_actual_gemini_api_key_here"),
        "client_initialized": is_ready,
        "active_model": DEFAULT_MODEL_NAME,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "status_badge": "ONLINE (Gemini)" if is_ready else "OFFLINE (Heuristic Mode)"
    }

def gemini_status() -> Dict[str, Any]:
    """Exposes live Gemini agent status dictionary for Streamlit UI."""
    return get_agent_status()

def _get_offline_fallback(
    predicted_class: str,
    confidence: float,
    top_features: Dict[str, float],
    error_reason: str = "Client offline"
) -> Dict[str, Any]:
    feature_summary = ", ".join(list(top_features.keys())[:3]) or "primary system metrics"

    kb = {
        "CPU_Bound": {
            "severity_level": "HIGH",
            "root_cause": "Saturated CPU execution pipelines; user-space computational threads exceed scheduler capacity.",
            "technical_analysis": (
                f"Active compute saturation driven by elevated metrics in {feature_summary}. "
                "The process run queue exceeds available logical cores, inducing thread preemption stalls."
            ),
            "investigation_commands": [
                "top -b -n 1 -o %CPU | head -n 15",
                "pidstat -u 1 3",
                "perf top --sort comm,dso",
            ],
            "remediation_commands": [
                "renice +10 -p <PID>",
                "taskset -cp <core_ids> <PID>",
                "systemctl set-property <service_name> CPUQuota=200%",
            ],
        },
        "Memory_Thrashing": {
            "severity_level": "CRITICAL",
            "root_cause": "Physical RAM exhaustion forcing high active page-cache eviction and continuous swap paging.",
            "technical_analysis": (
                f"The kernel memory subsystem is thrashing due to pressure on {feature_summary}. "
                "Dirty pages are being cycled to swap space, causing severe memory latency."
            ),
            "investigation_commands": [
                "vmstat 1 5",
                "ps -eo pid,ppid,cmd,%mem --sort=-%mem | head -n 10",
                "grep -i oom /var/log/messages",
            ],
            "remediation_commands": [
                "sysctl -w vm.swappiness=10",
                "echo 3 > /proc/sys/vm/drop_caches",
                "kill -15 <PID>",
            ],
        },
        "IO_Contention": {
            "severity_level": "HIGH",
            "root_cause": "Block device storage queue saturation with tasks blocked in uninterruptible sleep state (D-state).",
            "technical_analysis": (
                f"Storage pipeline backpressure is dominating system latency, indicated by {feature_summary}. "
                "Tasks are stalling on synchronous write flushes and block allocation."
            ),
            "investigation_commands": [
                "iostat -xz 1 3",
                "iotop -b -n 2 -d 1 -o",
                "lsof +D /data",
            ],
            "remediation_commands": [
                "ionice -c 3 -p <PID>",
                "sync && echo 1 > /proc/sys/vm/drop_caches",
                "blockdev --setra 4096 /dev/sdX",
            ],
        },
        "Normal": {
            "severity_level": "LOW",
            "root_cause": "System telemetry is operating within nominal operating bounds.",
            "technical_analysis": (
                "Scheduler queues, memory usage, and storage throughput are within baseline bounds. "
                "No resource starvation or thrashing detected."
            ),
            "investigation_commands": [
                "uptime",
                "sar -u 1 1",
            ],
            "remediation_commands": [
                "# System healthy. No remediation necessary.",
            ],
        },
    }

    base = kb.get(predicted_class, kb["Normal"]).copy()
    base["status"] = "offline_heuristic_fallback"
    base["model_used"] = "Local Rule Engine"
    base["error_details"] = error_reason
    base["confidence_pct"] = round(confidence * 100.0, 2)
    base["latency_ms"] = 0.5
    return base


def diagnose_bottleneck(
    predicted_class: str,
    confidence: float,
    telemetry_snapshot: Dict[str, Any],
    top_features: Dict[str, float],
) -> Dict[str, Any]:
    client = get_gemini_client()
    if client is None:
        print("[!] Notice: Gemini client offline or API key missing. Serving heuristic fallback.")
        return _get_offline_fallback(predicted_class, confidence, top_features, "No API key configured")

    system_instruction = (
        "You are an expert Linux/Unix Site Reliability Engineer and Kernel Performance Analyst.\n"
        "Analyze the provided operating system telemetry vector, the active machine learning classification result, "
        "and the top anomalous features.\n\n"
        "You MUST respond with STRICTLY valid JSON matching the following schema:\n"
        "{\n"
        '  "severity_level": "LOW | MEDIUM | HIGH | CRITICAL",\n'
        '  "root_cause": "<1-2 sentence precise technical description of kernel/hardware saturation>",\n'
        '  "technical_analysis": "<2-3 sentence explanation directly linking the metrics to the active bottleneck>",\n'
        '  "investigation_commands": ["<CLI command 1>", "<CLI command 2>", "<CLI command 3>"],\n'
        '  "remediation_commands": ["<Bash action 1>", "<Bash action 2>", "<Bash action 3>"]\n'
        "}\n"
        "Do NOT output markdown backticks (```json) or commentary outside the JSON object."
    )

    user_payload = {
        "predicted_state": predicted_class,
        "prediction_confidence": f"{confidence * 100:.2f}%",
        "top_contributing_metrics": top_features,
        "full_system_telemetry": telemetry_snapshot,
    }

    prompt_text = (
        "Telemetry Incident Snapshot:\n"
        f"{json.dumps(user_payload, indent=2)}\n\n"
        "Diagnose this system bottleneck and provide root cause analysis with investigation and remediation commands."
    )

    model_name = DEFAULT_MODEL_NAME
    start_time = time.perf_counter()

    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=DEFAULT_TEMPERATURE,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=config,
            )
            raw_text = response.text or ""
        else:
            model = client.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
            )
            response = model.generate_content(
                prompt_text,
                generation_config={
                    "temperature": DEFAULT_TEMPERATURE,
                    "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                },
            )
            raw_text = response.text or ""

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        cleaned_text = _clean_markdown_fences(raw_text)
        parsed_dict = json.loads(cleaned_text)

        validated_report = BottleneckDiagnosisReport.model_validate(parsed_dict)
        report = validated_report.model_dump()
        report["status"] = "online_gemini_agent"
        report["model_used"] = model_name
        report["confidence_pct"] = round(confidence * 100.0, 2)
        report["latency_ms"] = elapsed_ms
        return report

    except Exception as error:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        print(f"[!] Warning: Gemini API request encountered an error: {error}. Falling back.")
        fallback = _get_offline_fallback(predicted_class, confidence, top_features, str(error))
        fallback["latency_ms"] = elapsed_ms
        return fallback


if __name__ == "__main__":
    print("=" * 70)
    print("      TESTING GEMINI SYSADMIN ROOT CAUSE AGENT")
    print("=" * 70)

    # 1. Print Model & Agent Health Status
    status_info = get_agent_status()
    print("[*] Agent Operational Telemetry:")
    for k, v in status_info.items():
        print(f"    - {k:<24}: {v}")

    print("\n[*] Dispatching Diagnostic Request to Gemini...")

    sample_telemetry = {
        "cpu_user_pct": 12.4,
        "cpu_system_pct": 28.1,
        "cpu_iowait_pct": 58.6,
        "cpu_idle_pct": 0.9,
        "mem_used_pct": 64.2,
        "swap_used_pct": 8.0,
        "disk_write_kb_per_sec": 98450.0,
        "context_switches_per_sec": 4200.0,
    }

    sample_top_features = {
        "cpu_iowait_pct": 58.6,
        "disk_write_kb_per_sec": 98450.0,
        "cpu_idle_pct": 0.9,
    }

    report = diagnose_bottleneck(
        predicted_class="IO_Contention",
        confidence=0.985,
        telemetry_snapshot=sample_telemetry,
        top_features=sample_top_features,
    )

    print("\n[✓] Diagnostic Response Received:")
    print(json.dumps(report, indent=2))
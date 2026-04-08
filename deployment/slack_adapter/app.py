"""Cloud Run adapter that transforms Monitoring webhooks into Slack messages."""

import datetime as dt
import json
import os
from typing import Any

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)


def _truncate(text: str, limit: int = 2800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _extract_text(payload: dict[str, Any]) -> str:
    incident = payload.get("incident", {}) if isinstance(payload.get("incident"), dict) else {}
    summary = payload.get("summary") or incident.get("summary") or "Monitoring alert triggered"
    policy = incident.get("policy_name") or incident.get("policyDisplayName") or "unknown-policy"
    state = incident.get("state") or payload.get("state") or "unknown"
    project = (
        incident.get("resource", {}).get("labels", {}).get("project_id")
        if isinstance(incident.get("resource"), dict)
        else None
    )
    started_at = incident.get("started_at") or incident.get("startedAt")

    lines = [
        f":rotating_light: *{summary}*",
        f"Policy: `{policy}`",
        f"State: `{state}`",
    ]

    if project:
        lines.append(f"Project: `{project}`")

    if started_at:
        lines.append(f"Started: `{started_at}`")
    else:
        lines.append(f"Received: `{dt.datetime.utcnow().isoformat()}Z`")

    url = incident.get("url") or payload.get("url")
    if url:
        lines.append(f"<{url}|Open incident>")

    return _truncate("\n".join(lines))


@app.post("/")
def notify() -> Any:
    """Accept a Monitoring webhook payload and forward a Slack-formatted message."""
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    if not slack_webhook:
        return jsonify({"error": "SLACK_WEBHOOK_URL is not configured"}), 500

    payload = request.get_json(silent=True) or {}
    text = _extract_text(payload if isinstance(payload, dict) else {})

    response = requests.post(
        slack_webhook,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"text": text}),
        timeout=10,
    )

    if response.status_code >= 300:
        return (
            jsonify(
                {
                    "error": "Slack post failed",
                    "status": response.status_code,
                    "body": response.text,
                }
            ),
            502,
        )

    return jsonify({"status": "ok"}), 200


@app.get("/healthz")
def healthz() -> Any:
    """Return adapter liveness status for health checks."""
    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

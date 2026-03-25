"""Alerting integration for Phase 9.

Supports email and Slack alerts for major risk conditions.
"""

import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


def format_alert_message(alerts, metrics) -> str:
    lines = ["Trading system alert triggered:"]
    for a in alerts:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("Metrics")
    for key, value in metrics.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def send_email_alert(subject: str,
                     body: str,
                     sender: str,
                     recipient: str,
                     smtp_server: str = 'localhost',
                     smtp_port: int = 25,
                     username: Optional[str] = None,
                     password: Optional[str] = None) -> bool:
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient
        msg.set_content(body)

        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            if username and password:
                server.starttls()
                server.login(username, password)
            server.send_message(msg)

        logger.info('Email alert sent')
        return True
    except Exception as e:
        logger.error(f'Failed to send email alert: {e}')
        return False


def send_slack_alert(webhook_url: str, message: str) -> bool:
    try:
        payload = {"text": message}
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()

        logger.info('Slack alert sent')
        return True
    except Exception as e:
        logger.error(f'Failed to send Slack alert: {e}')
        return False

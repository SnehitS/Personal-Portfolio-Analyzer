"""Send the day's alerts somewhere you'll see them: a Windows toast, your phone (ntfy) or email.

Every channel is optional and configured under `notify` in settings; secrets come from .env.
A channel that fails is reported back, never raised, so one bad channel can't sink the run.
"""

from __future__ import annotations

import os
import smtplib
import subprocess
import sys
from dataclasses import dataclass
from email.message import EmailMessage
from xml.sax.saxutils import escape

import requests

from src.update_engine.alerts import Alert

DEFAULTS = {
    "channels": ["desktop"],   # any of: desktop, ntfy, email
    "min_level": "watch",      # "high" to only hear about alerts in big positions
    "always": False,           # also send on quiet days (a one-line "all clear")
    "ntfy_server": "https://ntfy.sh",
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
}


@dataclass
class Message:
    title: str
    body: str
    urgent: bool = False


def compose(alerts: list[Alert], failures: list[str], cfg: dict | None = None,
            summary: str = "") -> Message | None:
    """The notification to send, or None when there's nothing worth interrupting you for."""
    c = {**DEFAULTS, **(cfg or {})}
    wanted = [a for a in alerts if a.level == "high" or c["min_level"] == "watch"]
    if not wanted and not failures and not c["always"]:
        return None
    high = sum(a.level == "high" for a in wanted)
    watch = len(wanted) - high
    parts = [f"{high} high" if high else "", f"{watch} watch" if watch else ""]
    counts = ", ".join(p for p in parts if p)
    if counts:
        title = f"Portfolio: {counts} alert{'s' if len(wanted) > 1 else ''}"
    elif failures:
        title = "Portfolio: run had problems"
    else:
        title = "Portfolio: all clear"
    lines = [summary] if summary else []
    lines += [f"{'!' if a.level == 'high' else '-'} {a.message}" for a in wanted]
    lines += [f"x {f}" for f in failures]
    return Message(title, "\n".join(lines) or "Nothing needs attention today.", urgent=high > 0)


def _desktop(msg: Message, c: dict) -> None:
    if sys.platform != "win32":
        raise RuntimeError("desktop notifications are Windows only; use ntfy or email")
    body = "\n".join(msg.body.splitlines()[:4])  # toasts show about four lines
    xml = (f"<toast><visual><binding template='ToastGeneric'><text>{escape(msg.title)}</text>"
           f"<text>{escape(body)}</text></binding></visual></toast>")
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null;"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] > $null;"
        "$x = New-Object Windows.Data.Xml.Dom.XmlDocument; $x.LoadXml($env:TOAST_XML);"
        "$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe';"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app)"
        ".Show([Windows.UI.Notifications.ToastNotification]::new($x))"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                   env={**os.environ, "TOAST_XML": xml}, check=True, capture_output=True,
                   timeout=30)


def _ntfy(msg: Message, c: dict) -> None:
    topic = os.getenv("NTFY_TOPIC")
    if not topic:
        raise RuntimeError("set NTFY_TOPIC in .env")
    headers = {"Title": msg.title.encode("utf-8"), "Priority": "high" if msg.urgent else "default",
               "Tags": "chart_with_downwards_trend" if msg.urgent else "bar_chart"}
    if os.getenv("NTFY_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['NTFY_TOKEN']}"
    r = requests.post(f"{c['ntfy_server'].rstrip('/')}/{topic}", data=msg.body.encode("utf-8"),
                      headers=headers, timeout=20)
    r.raise_for_status()


def _email(msg: Message, c: dict) -> None:
    user, password = os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD")
    to = os.getenv("NOTIFY_EMAIL_TO") or user
    if not (user and password and to):
        raise RuntimeError("set SMTP_USERNAME, SMTP_PASSWORD (and optionally NOTIFY_EMAIL_TO) in .env")
    em = EmailMessage()
    em["Subject"], em["From"], em["To"] = msg.title, user, to
    em.set_content(msg.body)
    with smtplib.SMTP(c["smtp_host"], c["smtp_port"], timeout=30) as s:
        s.starttls()
        s.login(user, password)
        s.send_message(em)


CHANNELS = {"desktop": _desktop, "ntfy": _ntfy, "email": _email}


def send(msg: Message, cfg: dict | None = None) -> dict[str, str]:
    """Send to every configured channel. Returns {channel: "sent" or the error}."""
    c = {**DEFAULTS, **(cfg or {})}
    results = {}
    for name in c["channels"]:
        fn = CHANNELS.get(name)
        if fn is None:
            results[name] = f"unknown channel (expected one of {', '.join(CHANNELS)})"
            continue
        try:
            fn(msg, c)
            results[name] = "sent"
        except Exception as e:  # noqa: BLE001 - report and carry on with the other channels
            results[name] = f"failed: {e}"
    return results

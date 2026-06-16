"""Daily report via email (SMTP) or Telegram — picks based on env vars."""
import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Any

log = logging.getLogger(__name__)


def _send_email(subject: str, body: str) -> None:
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASS"]
    to_addr = os.environ["NOTIFY_EMAIL"]

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr

    with smtplib.SMTP(host, port) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    log.info("Email report sent to %s", to_addr)


def _send_telegram(text: str) -> None:
    import requests
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
    log.info("Telegram report sent")


def send_daily_report(report: dict[str, Any]) -> None:
    """Send daily trading summary — tries Telegram first, then email."""
    date = report.get("date", "N/A")
    body = (
        f"brain-topology-trader Daily Report — {date}\n"
        f"{'='*50}\n"
        f"Portfolio Value:   ${report.get('portfolio_value', 0):,.0f}\n"
        f"Tickers Analyzed:  {report.get('tickers_analyzed', 0)}\n"
        f"Orders Placed:     {report.get('orders_placed', 0)}\n"
        f"\nTop Longs:  {', '.join(report.get('top_longs', []))}\n"
        f"Top Shorts: {', '.join(report.get('top_shorts', []))}\n"
    )
    subject = f"[brain-topology-trader] {date}"

    sent = False
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        try:
            _send_telegram(body)
            sent = True
        except Exception as exc:
            log.warning("Telegram failed: %s", exc)

    if not sent and os.environ.get("SMTP_USER") and os.environ.get("NOTIFY_EMAIL"):
        try:
            _send_email(subject, body)
        except Exception as exc:
            log.error("Email failed: %s", exc)
    elif not sent:
        log.warning("No notification credentials configured — report logged only")
        log.info("REPORT:\n%s", body)

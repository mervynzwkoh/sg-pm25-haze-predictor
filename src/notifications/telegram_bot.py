"""Telegram bot notification dispatcher for Singapore athlete haze advisories.

Dispatches scheduled morning/evening training briefings (9pm SGT & 12pm SGT)
and real-time hazard spike alerts if PM2.5 crosses safe limits (>55 µg/m³).
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import requests
from pytz import timezone

from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SCHEDULED_HOURS_SGT, SPIKE_THRESHOLD_PM25

logger = logging.getLogger(__name__)
SGT = timezone("Asia/Singapore")


class TelegramNotifier:
    """Dispatches haze forecasts and athletic alerts to Telegram group or chat."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID

    @property
    def is_configured(self) -> bool:
        """Check if Telegram bot credentials are set."""
        return bool(self.token and self.chat_id)

    def send_message(self, text: str) -> bool:
        """Send a Markdown-formatted message to the configured Telegram chat."""
        if not self.is_configured:
            logger.info("Telegram bot not configured; previewing message in logs.")
            preview = "\n" + "=" * 40 + "\n[TELEGRAM PREVIEW]\n" + text + "\n" + "=" * 40 + "\n"
            try:
                print(preview)
            except UnicodeEncodeError:
                import sys
                sys.stdout.buffer.write(preview.encode("utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram notification sent successfully.")
                return True
            else:
                logger.error(f"Telegram API error {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Exception sending Telegram message: {e}")
            return False

    def format_forecast_message(self, forecast_data: Dict[str, Any], trigger_type: str = "scheduled") -> str:
        """Construct a high-readability athlete briefing message."""
        now = datetime.now(SGT)
        hour = now.hour

        # Determine target session context
        if trigger_type == "spike":
            header = "🚨 *URGENT SG HAZE ALERT: SPIKE DETECTED* 🚨"
            context_note = "A rapid haze elevation has been detected for the next 3 hours."
        elif hour >= 18 or hour < 4:
            header = "🌅 *SG HAZE BRIEFING: TOMORROW MORNING TRAINING (~7:00 AM)*"
            context_note = "Night-before forecast for planning tomorrow's morning workout."
        else:
            header = "🌇 *SG HAZE BRIEFING: TODAY EVENING TRAINING (~6:00 PM)*"
            context_note = "Mid-day forecast for planning today's evening workout."

        # Pick forecast horizon for the training session
        timeline = forecast_data.get("hourly_timeline", [])
        current_readings = forecast_data.get("current_readings", {})

        # Find target hour or default to +6h
        target_step = timeline[min(5, len(timeline) - 1)] if timeline else {}
        target_display = target_step.get("target_time_display", "Upcoming Hours")
        reg_preds = target_step.get("regional_predictions", {})

        lines = [
            header,
            f"_{context_note}_\n",
            f"⏱ *Forecast for:* {target_display}",
            f"📊 *Singapore Average:* {target_step.get('sg_average', 0)} µg/m³\n",
            "*Regional Outlook & Confidence:*",
        ]

        # Status emoji map
        for reg_key in ["central", "north", "south", "east", "west"]:
            info = reg_preds.get(reg_key, {})
            val = info.get("pm25_median", 0.0)
            conf = info.get("confidence_score", 0.0)
            band = info.get("health_band", "Normal")
            reg_name = reg_key.capitalize()

            if val <= 55:
                status_icon = "🟢"
                tag = "Safe"
            elif val <= 150:
                status_icon = "🟡"
                tag = "Elevated"
            else:
                status_icon = "🔴"
                tag = "Unhealthy"

            lines.append(f"• {status_icon} *{reg_name}*: {val:.0f} µg/m³ ({tag}) | Conf: {conf:.0f}%")

        # Add overall athletic advisory
        overall_advisory = forecast_data.get("overall_advisory", "")
        lines.append(f"\n🏃‍♂️ *Athletic Advisory:*\n{overall_advisory}")

        # Add link to dashboard
        lines.append("\n🌐 *Interactive Map & 12h Timeline:* Check repo / dashboard for live regional maps.")

        return "\n".join(lines)

    def check_and_notify(self, forecast_data: Dict[str, Any], force: bool = False) -> bool:
        """Evaluate if notification conditions are met and dispatch."""
        now = datetime.now(SGT)
        hour = now.hour

        is_scheduled = hour in SCHEDULED_HOURS_SGT

        # Check for immediate spike in next 3 hours
        spike_detected = False
        timeline = forecast_data.get("hourly_timeline", [])[:3]
        for step in timeline:
            if step.get("max_pm25", 0.0) > SPIKE_THRESHOLD_PM25:
                # Check if current reading was previously normal to prevent duplicate spike flood
                spike_detected = True
                break

        if force or is_scheduled or spike_detected:
            trigger_type = "spike" if spike_detected and not is_scheduled else "scheduled"
            msg = self.format_forecast_message(forecast_data, trigger_type=trigger_type)
            return self.send_message(msg)

        logger.info(f"Current hour {hour} SGT is not a scheduled notification window and no spike detected.")
        return False

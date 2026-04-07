"""Slack integration for anomaly alert notifications."""

from __future__ import annotations

import os
from typing import Any

import structlog

from pipeline_anomaly_detector.models.base_detector import AnomalyScore

log = structlog.get_logger(__name__)


class SlackIntegration:
    """Send anomaly alert notifications to Slack via Incoming Webhooks.

    Constructs rich Block Kit messages with pipeline name, anomaly score,
    contributing features, optional historical context, and a timestamp.

    Args:
        webhook_url: Slack Incoming Webhook URL.  Falls back to the
            ``SLACK_WEBHOOK_URL`` environment variable if not provided.

    Example::

        slack = SlackIntegration()
        success = slack.send(score, historical_context={"avg_duration": 3600})
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        """Initialise the Slack integration.

        Args:
            webhook_url: Webhook URL. If ``None``, reads from the
                ``SLACK_WEBHOOK_URL`` environment variable.
        """
        self._webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        if not self._webhook_url:
            log.warning(
                "slack_no_webhook_url",
                message="Set SLACK_WEBHOOK_URL env var or pass webhook_url to SlackIntegration",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        score: AnomalyScore,
        historical_context: dict | None = None,
    ) -> bool:
        """Post an anomaly alert to Slack.

        Args:
            score: The :class:`~pipeline_anomaly_detector.models.AnomalyScore`
                to report.
            historical_context: Optional dict with historical statistics, e.g.
                ``{"avg_duration": 3600, "avg_rows": 100_000}``.

        Returns:
            ``True`` if the message was sent successfully, ``False`` otherwise.
            Never raises.
        """
        if not self._webhook_url:
            log.warning(
                "slack_send_skipped",
                reason="no webhook URL configured",
                run_id=score.run_id,
            )
            return False

        try:
            import requests
        except ImportError:
            log.error("slack_requests_not_installed")
            return False

        blocks = self._build_blocks(score, historical_context)
        payload = {"blocks": blocks}

        try:
            response = requests.post(
                self._webhook_url,
                json=payload,
                timeout=10,
            )
            success = response.status_code == 200
            if not success:
                log.warning(
                    "slack_post_failed",
                    status_code=response.status_code,
                    body=response.text[:200],
                )
            else:
                log.info(
                    "slack_alert_sent",
                    run_id=score.run_id,
                    pipeline_name=score.pipeline_name,
                )
            return success
        except Exception as exc:  # noqa: BLE001
            log.error(
                "slack_send_exception",
                error=str(exc),
                run_id=score.run_id,
            )
            return False

    # ------------------------------------------------------------------
    # Internal block building
    # ------------------------------------------------------------------

    def _build_blocks(
        self,
        score: AnomalyScore,
        historical_context: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Build a Slack Block Kit message body.

        Args:
            score: The anomaly score to render.
            historical_context: Optional dict with keys like
                ``"avg_duration"`` and ``"avg_rows"`` for comparison context.

        Returns:
            List of Slack Block Kit block dicts.
        """
        pct = int(score.anomaly_score * 100)
        badge = "🚨 ANOMALY" if score.is_anomaly else "✅ Normal"

        # Header
        blocks: list[dict] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{badge} — {score.pipeline_name}",
                    "emoji": True,
                },
            },
            {"type": "divider"},
        ]

        # Score section
        score_bar = self._score_bar(score.anomaly_score)
        blocks.append(
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Anomaly Score*\n{score_bar} {pct}%",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Run ID*\n`{score.run_id}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Detector*\n`{score.detector_name}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Timestamp*\n{score.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
                    },
                ],
            }
        )

        # Contributing features
        if score.contributing_features:
            feature_list = "\n".join(
                f"• `{feat}`" for feat in score.contributing_features
            )
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Top Contributing Features*\n{feature_list}",
                    },
                }
            )

        # Historical context
        if historical_context:
            ctx_lines = []
            if "avg_duration" in historical_context:
                ctx_lines.append(
                    f"• Avg duration (last 30): *{historical_context['avg_duration']:.1f}s*"
                )
            if "avg_rows" in historical_context:
                ctx_lines.append(
                    f"• Avg rows (last 30): *{historical_context['avg_rows']:,.0f}*"
                )
            for k, v in historical_context.items():
                if k not in ("avg_duration", "avg_rows"):
                    ctx_lines.append(f"• {k}: *{v}*")

            if ctx_lines:
                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Historical Context*\n" + "\n".join(ctx_lines),
                        },
                    }
                )

        blocks.append({"type": "divider"})
        return blocks

    @staticmethod
    def _score_bar(anomaly_score: float, width: int = 10) -> str:
        """Build a simple text progress bar for the anomaly score.

        Args:
            anomaly_score: Score in [0.0, 1.0].
            width: Total bar width in characters.

        Returns:
            A string like ``"████░░░░░░ 40%"``.
        """
        filled = int(round(anomaly_score * width))
        bar = "█" * filled + "░" * (width - filled)
        return bar

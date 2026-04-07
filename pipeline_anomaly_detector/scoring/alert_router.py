"""Alert routing for anomaly scores."""

from __future__ import annotations

import fnmatch
from datetime import datetime, timezone
from typing import Any

import structlog

from pipeline_anomaly_detector.models.base_detector import AnomalyScore

log = structlog.get_logger(__name__)

# Type alias for routing configuration.
# Keys are pipeline_name glob patterns; values are lists of channel configs.
RoutingConfig = dict[str, list[dict[str, Any]]]


class AlertRouter:
    """Routes anomaly alerts to configured channels.

    The router matches a pipeline name against a set of glob patterns and
    sends alerts to the configured channel(s) for each match.  Deduplication
    prevents the same pipeline from triggering repeated alerts within a
    configurable time window.

    Channel config examples::

        {
            "orders_pipeline": [
                {"type": "slack", "channel": "#data-alerts"},
                {"type": "log_only"}
            ],
            "*": [{"type": "log_only"}]
        }

    Args:
        config: :data:`RoutingConfig` mapping glob patterns to channel configs.
        dedup_window_minutes: Alerts for the same pipeline are suppressed if
            one has already been sent within this window. Defaults to ``60``.

    Example::

        router = AlertRouter(config={"*": [{"type": "log_only"}]})
        router.route(score)
    """

    def __init__(
        self,
        config: RoutingConfig,
        dedup_window_minutes: int = 60,
    ) -> None:
        """Initialise the alert router.

        Args:
            config: Routing configuration mapping glob patterns to channel
                lists.
            dedup_window_minutes: Deduplication window in minutes.
        """
        self._config = config
        self._dedup_window = dedup_window_minutes
        # Tracks (pipeline_name, window_bucket) tuples that have already been alerted.
        self._sent: set[tuple[str, int]] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, score: AnomalyScore) -> None:
        """Route an anomaly alert to the appropriate channel(s).

        Only anomalous scores (``score.is_anomaly is True``) trigger alerts.
        Deduplication: if an alert for the same pipeline has already been
        sent in the current time window the alert is suppressed.

        Args:
            score: The anomaly score to potentially route.
        """
        if not score.is_anomaly:
            log.debug(
                "alert_router_skipped_non_anomaly",
                run_id=score.run_id,
                pipeline_name=score.pipeline_name,
            )
            return

        if self._is_duplicate(score.pipeline_name):
            log.info(
                "alert_router_dedup_suppressed",
                pipeline_name=score.pipeline_name,
                dedup_window_minutes=self._dedup_window,
            )
            return

        self._mark_sent(score.pipeline_name)

        # Find matching channel configs
        matched_channels = self._match_channels(score.pipeline_name)

        if not matched_channels:
            log.warning(
                "alert_router_no_matching_channels",
                pipeline_name=score.pipeline_name,
            )
            return

        for channel_config in matched_channels:
            channel_type = channel_config.get("type", "log_only")
            if channel_type == "slack":
                self._send_slack(score, channel_config)
            elif channel_type == "log_only":
                self._send_log(score)
            else:
                log.warning(
                    "alert_router_unknown_channel_type",
                    channel_type=channel_type,
                    pipeline_name=score.pipeline_name,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_window_bucket(self) -> int:
        """Return the current time bucket for deduplication.

        Returns:
            An integer representing the current time window bucket.
        """
        now_ts = datetime.now(tz=timezone.utc).timestamp()
        bucket_seconds = self._dedup_window * 60
        return int(now_ts // bucket_seconds)

    def _is_duplicate(self, pipeline_name: str) -> bool:
        """Check if an alert for this pipeline has been sent in the current window.

        Args:
            pipeline_name: The pipeline to check.

        Returns:
            ``True`` if a duplicate alert should be suppressed.
        """
        bucket = self._current_window_bucket()
        return (pipeline_name, bucket) in self._sent

    def _mark_sent(self, pipeline_name: str) -> None:
        """Record that an alert has been sent for this pipeline in the current window.

        Args:
            pipeline_name: The pipeline that was alerted.
        """
        bucket = self._current_window_bucket()
        self._sent.add((pipeline_name, bucket))

    def _match_channels(self, pipeline_name: str) -> list[dict]:
        """Find all channel configs whose pattern matches *pipeline_name*.

        Args:
            pipeline_name: The pipeline name to match.

        Returns:
            Flat list of channel config dicts from all matching patterns.
        """
        matched: list[dict] = []
        for pattern, channels in self._config.items():
            if fnmatch.fnmatch(pipeline_name, pattern):
                matched.extend(channels)
        return matched

    def _send_slack(self, score: AnomalyScore, channel_config: dict) -> None:
        """Send a Slack alert for the anomaly score.

        Args:
            score: The anomaly score to alert on.
            channel_config: Channel configuration including optional
                ``webhook_url``.
        """
        from pipeline_anomaly_detector.integrations.slack_integration import (
            SlackIntegration,
        )

        webhook_url = channel_config.get("webhook_url")
        slack = SlackIntegration(webhook_url=webhook_url)
        success = slack.send(score)
        log.info(
            "alert_router_slack_sent",
            pipeline_name=score.pipeline_name,
            run_id=score.run_id,
            success=success,
        )

    def _send_log(self, score: AnomalyScore) -> None:
        """Log the anomaly score via structlog.

        Args:
            score: The anomaly score to log.
        """
        log.warning(
            "anomaly_detected",
            run_id=score.run_id,
            pipeline_name=score.pipeline_name,
            anomaly_score=round(score.anomaly_score, 4),
            contributing_features=score.contributing_features,
            detector_name=score.detector_name,
            timestamp=score.timestamp.isoformat(),
        )

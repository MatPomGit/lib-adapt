# Stability monitor

Pure Kotlin/JVM state machine with explicit threshold, consecutive qualifying updates, timeout and invalidation. No study-specific defaults, score models or participant assignment. Timestamp units: monotonic milliseconds. Equality at the deadline is censored, not observed. Historical study scorer and controller stay in migration snapshots. No clinical validation.

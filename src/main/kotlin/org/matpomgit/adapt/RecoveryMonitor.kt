package org.matpomgit.adapt

data class RecoveryMonitorConfig(
    val delta: Double,
    val stabilityWindowUpdates: Int,
    val censorLimitMs: Long,
) {
    init {
        require(delta.isFinite() && delta >= 0)
        require(stabilityWindowUpdates > 0)
        require(censorLimitMs > 0)
    }
}

enum class RecoveryMonitorStatus { IN_PROGRESS, OBSERVED, CENSORED, DATA_INVALID }

data class RecoveryMonitorState(
    val monitoringStartedAtMs: Long,
    val config: RecoveryMonitorConfig,
    val status: RecoveryMonitorStatus = RecoveryMonitorStatus.IN_PROGRESS,
    val qualifyingRunLength: Int = 0,
    val qualifyingRunStartedAtMs: Long? = null,
    val observedAtMs: Long? = null,
    val recoveryLatencyMs: Long? = null,
    val lastUpdateMs: Long = monitoringStartedAtMs,
    val invalidReason: String? = null,
) {
    init {
        require(monitoringStartedAtMs >= 0)
        require(monitoringStartedAtMs <= Long.MAX_VALUE - config.censorLimitMs)
        require(lastUpdateMs >= monitoringStartedAtMs)
        require(qualifyingRunLength >= 0)
    }

    fun update(timestampMs: Long, q: Double): RecoveryMonitorState {
        if (status != RecoveryMonitorStatus.IN_PROGRESS) return this
        if (timestampMs < lastUpdateMs || !q.isFinite()) {
            return invalidate(timestampMs.coerceAtLeast(lastUpdateMs), "invalid_recovery_update")
        }
        val censorBoundary = monitoringStartedAtMs + config.censorLimitMs
        if (timestampMs >= censorBoundary) {
            return copy(
                status = RecoveryMonitorStatus.CENSORED,
                qualifyingRunLength = 0,
                qualifyingRunStartedAtMs = null,
                lastUpdateMs = timestampMs,
            )
        }

        if (q >= -config.delta) {
            val runStart = qualifyingRunStartedAtMs ?: timestampMs
            val newLength = qualifyingRunLength + 1
            if (newLength >= config.stabilityWindowUpdates) {
                return copy(
                    status = RecoveryMonitorStatus.OBSERVED,
                    qualifyingRunLength = newLength,
                    qualifyingRunStartedAtMs = runStart,
                    observedAtMs = timestampMs,
                    recoveryLatencyMs = runStart - monitoringStartedAtMs,
                    lastUpdateMs = timestampMs,
                )
            }
            return copy(
                qualifyingRunLength = newLength,
                qualifyingRunStartedAtMs = runStart,
                lastUpdateMs = timestampMs,
            )
        }

        return copy(
            qualifyingRunLength = 0,
            qualifyingRunStartedAtMs = null,
            lastUpdateMs = timestampMs,
        )
    }

    fun tick(timestampMs: Long): RecoveryMonitorState {
        if (status != RecoveryMonitorStatus.IN_PROGRESS || timestampMs < lastUpdateMs) return this
        return if (timestampMs >= monitoringStartedAtMs + config.censorLimitMs) {
            copy(
                status = RecoveryMonitorStatus.CENSORED,
                qualifyingRunLength = 0,
                qualifyingRunStartedAtMs = null,
                lastUpdateMs = timestampMs,
            )
        } else {
            copy(lastUpdateMs = timestampMs)
        }
    }

    fun invalidate(timestampMs: Long, reason: String): RecoveryMonitorState {
        if (status != RecoveryMonitorStatus.IN_PROGRESS || reason.isBlank()) return this
        if (timestampMs < lastUpdateMs) return this
        return copy(
            status = RecoveryMonitorStatus.DATA_INVALID,
            qualifyingRunLength = 0,
            qualifyingRunStartedAtMs = null,
            lastUpdateMs = timestampMs,
            invalidReason = reason,
        )
    }
}

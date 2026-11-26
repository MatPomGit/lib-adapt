package org.matpomgit.adapt

import org.junit.Assert.*
import org.junit.Test

class RecoveryMonitorTest {
    @Test
    fun stableRunUsesFirstQualifyingUpdateAsRecoveryLatency() {
        val config = RecoveryMonitorConfig(delta = 0.5, stabilityWindowUpdates = 3, censorLimitMs = 20_000)
        var state = RecoveryMonitorState(monitoringStartedAtMs = 10_000, config = config)
        state = state.update(11_000, -0.2)
        state = state.update(12_000, -0.3)
        state = state.update(13_000, -0.1)

        assertEquals(RecoveryMonitorStatus.OBSERVED, state.status)
        assertEquals(1_000L, state.recoveryLatencyMs)
        assertEquals(13_000L, state.observedAtMs)
    }

    @Test
    fun failedUpdateResetsStabilityRun() {
        val config = RecoveryMonitorConfig(delta = 0.5, stabilityWindowUpdates = 2, censorLimitMs = 20_000)
        var state = RecoveryMonitorState(monitoringStartedAtMs = 10_000, config = config)
        state = state.update(11_000, -0.2)
        state = state.update(12_000, -0.8)
        state = state.update(13_000, -0.1)

        assertEquals(RecoveryMonitorStatus.IN_PROGRESS, state.status)
        assertEquals(1, state.qualifyingRunLength)
        assertEquals(13_000L, state.qualifyingRunStartedAtMs)
    }

    @Test
    fun censorBoundaryIsNotConvertedToObservedRecovery() {
        val config = RecoveryMonitorConfig(delta = 0.5, stabilityWindowUpdates = 1, censorLimitMs = 2_000)
        val state = RecoveryMonitorState(monitoringStartedAtMs = 10_000, config = config)
            .update(12_000, 0.0)

        assertEquals(RecoveryMonitorStatus.CENSORED, state.status)
        assertNull(state.recoveryLatencyMs)
    }

    @Test fun nonfiniteAndBackwardUpdatesInvalidate() {
        val c = RecoveryMonitorConfig(0.5, 2, 100)
        val state = RecoveryMonitorState(10, c).update(20, 0.0)
        assertEquals(RecoveryMonitorStatus.DATA_INVALID, state.update(19, 0.0).status)
        assertEquals(RecoveryMonitorStatus.DATA_INVALID, state.update(21, Double.NaN).status)
    }
    @Test fun terminalStateIsImmutable() {
        val s = RecoveryMonitorState(0, RecoveryMonitorConfig(0.5, 1, 100)).update(1, 0.0)
        assertEquals(s, s.update(100, -10.0))
        assertEquals(s, s.tick(100))
    }
    @Test(expected = IllegalArgumentException::class)
    fun overflowBoundaryRejected() {
        RecoveryMonitorState(Long.MAX_VALUE - 1, RecoveryMonitorConfig(0.5, 1, 10))
    }
}

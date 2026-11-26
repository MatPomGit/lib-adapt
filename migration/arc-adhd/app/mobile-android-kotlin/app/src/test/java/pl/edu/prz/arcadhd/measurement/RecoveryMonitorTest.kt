package pl.edu.prz.arcadhd.measurement

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RecoveryMonitorTest {
    private val reference = RecoveryReference(
        goCorrect = 4,
        goTotal = 4,
        noGoCorrect = 4,
        noGoTotal = 4,
        correctGoRtMs = listOf(500.0, 510.0, 490.0, 505.0, 495.0),
        deviceTimingFloorMs = 10.0,
    )

    @Test
    fun behaviorOnlyScoreIsNonPositiveAndComputable() {
        val score = RecoveryScorer.score(
            reference,
            RecoveryWindow(
                goCorrect = 4,
                goTotal = 4,
                noGoCorrect = 4,
                noGoTotal = 4,
                correctGoRtMs = emptyList(),
            ),
            RecoveryFeatureModel.B,
        )
        assertNotNull(score)
        assertTrue(requireNotNull(score).q <= 0.0)
    }

    @Test
    fun timingModelRejectsInsufficientCorrectGoTiming() {
        val score = RecoveryScorer.score(
            reference,
            RecoveryWindow(
                goCorrect = 2,
                goTotal = 2,
                noGoCorrect = 2,
                noGoTotal = 2,
                correctGoRtMs = listOf(700.0, 710.0),
            ),
            RecoveryFeatureModel.B_S,
        )
        assertNull(score)
    }

    @Test
    fun slowingCreatesDeficitWithoutPositiveCompensation() {
        val slowed = requireNotNull(
            RecoveryScorer.score(
                reference,
                RecoveryWindow(
                    goCorrect = 4,
                    goTotal = 4,
                    noGoCorrect = 4,
                    noGoTotal = 4,
                    correctGoRtMs = listOf(700.0, 710.0, 690.0, 705.0),
                ),
                RecoveryFeatureModel.B_S,
            ),
        )
        assertTrue(requireNotNull(slowed.slowingDeficit) > 0.0)
        assertTrue(slowed.q < 0.0)
    }

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
}

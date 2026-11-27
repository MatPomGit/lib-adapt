# API

`RecoveryMonitorConfig` sets the threshold tolerance, required consecutive updates and timeout in milliseconds. `RecoveryMonitorState.update`, `tick` and `invalidate` return immutable state copies. `RecoveryMonitorStatus` distinguishes in-progress, observed, censored and invalid data. Exact deadline is censored. See [contract](../contracts/api.json). No scientific scoring policy or participant allocation is defined by this API.

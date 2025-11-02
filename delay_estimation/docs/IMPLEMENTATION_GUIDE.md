# Implementation Guide

This guide explains how to implement the delay estimation algorithm from the paper:
"Measurement Delay Estimation for Kalman Filter in Networked Control Systems"

## Current Status

The framework is set up with placeholder implementations. The following need to be implemented based on the paper:

### 1. Delay Estimation Algorithms

**File**: `estimators/delay_estimator.py`

#### Innovation-Based Estimator (`InnovationBasedEstimator`)

Implement the `estimate_delay()` method:

```python
def estimate_delay(self, innovation: np.ndarray, innovation_covariance: np.ndarray) -> int:
    """
    TODO: Implement innovation-based delay estimation

    Algorithm from paper:
    1. Compute innovation sequence statistics
    2. Compare with expected statistics for each delay hypothesis
    3. Select delay with best match

    Args:
        innovation: Current innovation (measurement residual)
        innovation_covariance: Innovation covariance matrix

    Returns:
        estimated_delay: Estimated delay in time steps
    """
    # Update history
    self.update_history(innovation)

    # YOUR IMPLEMENTATION HERE

    return estimated_delay
```

#### Maximum Likelihood Estimator (`MLDelayEstimator`)

Implement the `estimate_delay()` method:

```python
def estimate_delay(self, innovation: np.ndarray, innovation_covariance: np.ndarray) -> int:
    """
    TODO: Implement ML-based delay estimation

    Algorithm from paper:
    1. Compute likelihood for each delay hypothesis d = 0, 1, ..., max_delay
    2. Likelihood based on innovation sequence
    3. Select delay with maximum likelihood

    Returns:
        estimated_delay: Delay with maximum likelihood
    """
    # Update history
    self.update_history(innovation)

    # Compute likelihoods for each delay hypothesis
    for d in range(self.max_delay + 1):
        # YOUR IMPLEMENTATION HERE
        self.likelihoods[d] = ...

    return int(np.argmax(self.likelihoods))
```

#### Bayesian Estimator (`BayesianDelayEstimator`)

Implement the `estimate_delay()` method:

```python
def estimate_delay(self, innovation: np.ndarray, innovation_covariance: np.ndarray) -> int:
    """
    TODO: Implement Bayesian delay estimation

    Algorithm from paper:
    1. Update posterior probability using Bayes rule
    2. P(d|y) ∝ P(y|d) * P(d)
    3. Return MAP (maximum a posteriori) estimate

    Returns:
        estimated_delay: MAP estimate of delay
    """
    # Update history
    self.update_history(innovation)

    # Compute posterior for each delay
    for d in range(self.max_delay + 1):
        # YOUR IMPLEMENTATION HERE
        # likelihood = P(y|d)
        # posterior[d] = likelihood * prior[d]
        pass

    # Normalize
    self.posterior = self.posterior / np.sum(self.posterior)

    # Update prior for next step
    self.prior = self.posterior.copy()

    return int(np.argmax(self.posterior))
```

### 2. Kalman Filter with Delay Compensation

**New File**: `estimators/delay_compensated_kf.py`

Create a Kalman filter that uses the delay estimate to properly handle delayed measurements:

```python
class DelayCompensatedKF(KalmanFilter):
    """
    Kalman Filter with delay compensation

    Uses estimated delay to properly propagate state and covariance
    """

    def __init__(self, ...):
        super().__init__(...)
        self.state_history = deque(maxlen=max_delay + 1)
        self.covariance_history = deque(maxlen=max_delay + 1)

    def update_with_delay(self, y: np.ndarray, estimated_delay: int):
        """
        Update with delayed measurement

        1. Retrieve state/covariance from 'estimated_delay' steps ago
        2. Perform update with that state
        3. Re-propagate to current time
        """
        # YOUR IMPLEMENTATION HERE
        pass
```

### 3. Paper-Specific Details to Implement

Based on the paper you'll provide, implement:

1. **Innovation statistics**: How to compute and compare innovation sequences
2. **Likelihood computation**: Exact formula for P(y|d) given delay d
3. **Delay compensation**: How to use the delay estimate in the Kalman filter
4. **Threshold detection**: Any adaptive thresholds for delay change detection

### 4. Testing

Create test scenarios in `scenarios/`:

```python
# scenarios/test_constant_delay.py
# scenarios/test_varying_delay.py
# scenarios/test_step_delay.py
```

### 5. Analysis

Add analysis scripts in `utils/`:

```python
# utils/delay_estimation_analysis.py
# - Analyze delay estimation accuracy
# - Compare different methods
# - Statistical analysis of performance
```

## Running Experiments

After implementation:

```bash
# Single run with specific method
cd delay_estimation
uv run python main.py --method innovation --show

# Compare standard KF vs delay-compensated KF
uv run python main.py --method ml --compare --show

# Test different methods
uv run python main.py --method bayesian --compare
```

## Integration Checklist

- [ ] Implement innovation-based estimator
- [ ] Implement ML estimator
- [ ] Implement Bayesian estimator
- [ ] Create delay-compensated Kalman filter
- [ ] Add unit tests
- [ ] Create validation scenarios
- [ ] Compare with paper results
- [ ] Document algorithm details

## References

Add the paper details and key equations here once you provide them.

#!/usr/bin/env python3
"""Monte Carlo methods — estimation, integration, MCMC, importance sampling.

Implements: pi estimation, numerical integration, Metropolis-Hastings MCMC,
importance sampling, bootstrap confidence intervals, rejection sampling.

Usage: python monte_carlo.py [--test]
"""

import sys, math, random

def estimate_pi(n=100000):
    inside = sum(1 for _ in range(n) if random.random()**2 + random.random()**2 <= 1)
    return 4.0 * inside / n

def mc_integrate(f, a, b, n=100000):
    """Monte Carlo integration of f over [a,b]."""
    total = sum(f(random.uniform(a, b)) for _ in range(n))
    return (b - a) * total / n

def mc_integrate_2d(f, x_range, y_range, n=100000):
    """2D Monte Carlo integration."""
    x0, x1 = x_range
    y0, y1 = y_range
    area = (x1 - x0) * (y1 - y0)
    total = sum(f(random.uniform(x0, x1), random.uniform(y0, y1)) for _ in range(n))
    return area * total / n

def importance_sampling(f, proposal_sample, proposal_pdf, target_pdf, n=100000):
    """Importance sampling: E_target[f] using proposal distribution."""
    total = 0.0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / max(proposal_pdf(x), 1e-30)
        total += f(x) * w
    return total / n

def rejection_sampling(target_pdf, proposal_sample, proposal_pdf, M, n=1000):
    """Rejection sampling: generate samples from target using proposal."""
    samples = []
    attempts = 0
    while len(samples) < n:
        x = proposal_sample()
        u = random.random()
        attempts += 1
        if u <= target_pdf(x) / (M * proposal_pdf(x)):
            samples.append(x)
    return samples, attempts

def metropolis_hastings(target_log_pdf, initial, n=10000, proposal_std=1.0, burn_in=1000):
    """Metropolis-Hastings MCMC sampler."""
    samples = []
    current = initial
    current_log_p = target_log_pdf(current)
    accepted = 0
    
    for i in range(n + burn_in):
        proposed = current + random.gauss(0, proposal_std)
        proposed_log_p = target_log_pdf(proposed)
        log_alpha = proposed_log_p - current_log_p
        
        if math.log(random.random()) < log_alpha:
            current = proposed
            current_log_p = proposed_log_p
            accepted += 1
        
        if i >= burn_in:
            samples.append(current)
    
    return samples, accepted / (n + burn_in)

def bootstrap_ci(data, stat_fn=None, n_bootstrap=10000, alpha=0.05):
    """Bootstrap confidence interval."""
    if stat_fn is None:
        stat_fn = lambda d: sum(d) / len(d)
    
    n = len(data)
    stats = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n-1)] for _ in range(n)]
        stats.append(stat_fn(sample))
    
    stats.sort()
    lo = stats[int(n_bootstrap * alpha / 2)]
    hi = stats[int(n_bootstrap * (1 - alpha / 2))]
    return lo, hi, stat_fn(data)

def monte_carlo_tree(f, n=10000):
    """Estimate E[f(X)] where X ~ Uniform(0,1) with variance estimate."""
    vals = [f(random.random()) for _ in range(n)]
    mean = sum(vals) / n
    var = sum((v - mean)**2 for v in vals) / (n - 1)
    se = math.sqrt(var / n)
    return mean, se

# --- Tests ---

def test_pi():
    random.seed(42)
    pi = estimate_pi(500000)
    assert abs(pi - math.pi) < 0.02, f"Pi estimate {pi} too far from {math.pi}"

def test_integration():
    random.seed(42)
    result = mc_integrate(math.sin, 0, math.pi, 200000)
    assert abs(result - 2.0) < 0.05, f"Integral of sin(x) over [0,π] = {result}, expected 2"

def test_2d_integration():
    random.seed(42)
    result = mc_integrate_2d(lambda x, y: 1 if x**2 + y**2 <= 1 else 0,
                             (-1, 1), (-1, 1), 200000)
    assert abs(result - math.pi) < 0.05

def test_importance_sampling():
    random.seed(42)
    # E[x^2] where x ~ N(0,1), use uniform proposal
    result = importance_sampling(
        lambda x: x**2,
        lambda: random.uniform(-5, 5),
        lambda x: 0.1,
        lambda x: math.exp(-x**2/2) / math.sqrt(2*math.pi),
        n=200000
    )
    assert abs(result - 1.0) < 0.1, f"E[X²] = {result}, expected 1.0"

def test_rejection_sampling():
    random.seed(42)
    # Sample from truncated normal using uniform proposal
    def target(x): return math.exp(-x**2/2) if -3 <= x <= 3 else 0
    samples, _ = rejection_sampling(
        target,
        lambda: random.uniform(-3, 3),
        lambda x: 1/6,
        M=7,
        n=1000
    )
    assert len(samples) == 1000
    mean = sum(samples) / len(samples)
    assert abs(mean) < 0.2

def test_mcmc():
    random.seed(42)
    # Sample from N(3, 1)
    samples, accept_rate = metropolis_hastings(
        lambda x: -(x - 3)**2 / 2,
        initial=0, n=20000, proposal_std=1.5, burn_in=2000
    )
    mean = sum(samples) / len(samples)
    assert abs(mean - 3.0) < 0.2, f"MCMC mean {mean}, expected 3.0"
    assert 0.1 < accept_rate < 0.9

def test_bootstrap():
    random.seed(42)
    data = [random.gauss(5, 1) for _ in range(100)]
    lo, hi, point = bootstrap_ci(data)
    assert lo < 5.0 < hi
    assert abs(point - 5.0) < 0.5

def test_variance_estimate():
    random.seed(42)
    mean, se = monte_carlo_tree(lambda x: x**2, n=50000)
    # E[X²] where X~U(0,1) = 1/3
    assert abs(mean - 1/3) < 0.01
    assert se < 0.01

if __name__ == "__main__":
    if "--test" in sys.argv or len(sys.argv) == 1:
        test_pi()
        test_integration()
        test_2d_integration()
        test_importance_sampling()
        test_rejection_sampling()
        test_mcmc()
        test_bootstrap()
        test_variance_estimate()
        print("All tests passed!")

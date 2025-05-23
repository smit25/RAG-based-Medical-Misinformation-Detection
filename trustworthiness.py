import math
from typing import List, Tuple, Optional



def exponential_penalty_ratio(
    n_correct: int,
    n_unverif: int,
    n_wrong: int,
    severities: Optional[List[Optional[float]]],
    alpha: float = 0.5,
    beta: float = 1.0
) -> float:
    """
    EPR: (Correct + α·Unverif) / N  ×  exp(−β·avg_severity)
    """
    N = n_correct + n_unverif + n_wrong
    if N == 0:
        return 0.0
    R = (n_correct + alpha * n_unverif) / N
    sv = [s for s in (severities or []) if isinstance(s, (int, float))]
    avg_sev = sum(sv) / len(sv) if sv else 0.0
    return R * math.exp(-beta * avg_sev)


def compute_uncertainty(
    confidences: Optional[List[Optional[float]]],
    lambda_uncertainty: float = 0.5
) -> float:
    """
    Uncertainty = λ_u × average_confidence_unverif
    """
    cf = [c for c in (confidences or []) if isinstance(c, (int, float))]
    return lambda_uncertainty * (sum(cf) / len(cf)) if cf else 0.0


def compute_trust_and_uncertainty(
    n_correct: int,
    n_unverif: int,
    n_wrong: int,
    severities: Optional[List[Optional[float]]],
    confidences_unverif: Optional[List[Optional[float]]],
    alpha: float = 0.5,
    beta: float = 5.0,
    lambda_uncertainty: float = 0.5
) -> Tuple[float, float, float, float]:
    
    """
    Returns:
      - trust_score
      - uncertainty (±)
      - lower_bound (trust - uncertainty)
      - upper_bound (trust + uncertainty)
    """
    trust = exponential_penalty_ratio(n_correct, n_unverif, n_wrong, severities, alpha, beta)
    uncertainty = compute_uncertainty(confidences_unverif, lambda_uncertainty)
    lower = max(0.0, trust - uncertainty)
    upper = min(1.0, trust + uncertainty)
    return trust, uncertainty, lower, upper


def beta_posterior_trust(
    n_correct: int,
    n_unverif: int,
    n_wrong: int,
    severities: List[float],
    alpha_credit: float = 0.5,
    gamma: float = 2.0,
    alpha0: float = 2.0,
    beta0: float = 2.0
) -> Tuple[float, float, float]:
    """
    Returns (posterior_mean, lower95, upper95) for the Beta-posterior metric.

    Each claim contributes a fractional 'success' t_i:
      correct         → t = 1
      unverifiable    → t = α_credit
      wrong, severity → t = (1 - severity)**gamma

    Prior:  Beta(alpha0, beta0)

    The 95 % CI is mean ± 1.96·sqrt(var).
    """

    successes = (
        n_correct * 1.0
        + n_unverif * alpha_credit
        + sum((1 - s)**gamma for s in severities)
    )
    failures = (
        n_unverif * (1 - alpha_credit)
        + sum(1 - (1 - s)**gamma for s in severities)
    )

    a_post = alpha0 + successes
    b_post = beta0 + failures

    mean = a_post / (a_post + b_post)
    var = (a_post * b_post) / ((a_post + b_post) ** 2 * (a_post + b_post + 1))
    half_width = 1.96 * math.sqrt(var)
    lower = max(0.0, mean - half_width)
    upper = min(1.0, mean + half_width)

    return mean, half_width, lower, upper


def test():
    correct = 80
    unverif = 10
    wrong   = 10
    sevs    = [0.9, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1]

    epr_score = exponential_penalty_ratio(correct, unverif, wrong, sevs,
                                        alpha=0.5, beta=5)
    print(f"EPR trustworthiness: {epr_score:.3f}")

    mean, uncer, lo, hi = beta_posterior_trust(correct, unverif, wrong, sevs,
                                        alpha_credit=0.5, gamma=2,
                                        alpha0=2, beta0=2)
    print(f"Beta-posterior: {mean:.3f} +- {uncer:.3f}      (95 % CI {lo:.3f}–{hi:.3f})")



def test2():
    n_correct = 70
    severities_wrong = [0.9, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1]  # 10 wrong claims
    confidences_unverif = [0.9, 0.8, 0.3, 0.1, 0.6, 0.4, 0.2, 0.7, 0.5, 0.1,
                           0.4, 0.2, 0.3, 0.6, 0.5, 0.1, 0.2, 0.1, 0.4, 0.3]  # 20 unverifiable

    trust_score, uncertainty, lower, upper = exponential_penalty_ratio(
        n_correct,
        severities_wrong,
        confidences_unverif,
        gamma=2.0,
        lambda_penalty=0.5,
        lambda_uncertainty=0.5
    )

    print(f"Trust Score     : {trust_score:.3f}")
    print(f"Uncertainty     : ±{uncertainty:.3f}")
    print(f"Confidence Band : [{lower:.3f} – {upper:.3f}]")


if __name__ == "__main__":
    test()

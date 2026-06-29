"""Basic probabilistic Ball Mapper comparison example."""

import numpy as np

from probabilistic_ball_mapper import (
    FGWConfig,
    MembershipConfig,
    fgw_distance,
    fit,
    summarize,
)


def main() -> None:
    rng = np.random.default_rng(42)
    X_a = rng.normal(size=(500, 2))
    X_b = rng.normal(loc=0.3, size=(500, 2))

    membership = MembershipConfig(eps=0.6, membership="gaussian")
    pbm_a = fit(X_a, config=membership)
    pbm_b = fit(X_b, config=membership)

    distance, info = fgw_distance(
        pbm_a,
        pbm_b,
        config=FGWConfig(alpha=0.5, relation="soft_overlap"),
        return_plan=True,
    )

    print("PBM A:", summarize(pbm_a))
    print("PBM B:", summarize(pbm_b))
    print("FGW distance:", distance)
    print("FGW objective:", info["objective"])


if __name__ == "__main__":
    main()

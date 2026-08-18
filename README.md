# fabio

## The Problem

> Let N be the total number of players out of the Serie A dataset. Let x in {0,1}^n be the binary decision vector:
> x_i = {1 if the player is drafted; 0 else}
> The goal is to maximise the total roaster xPts (Expected Points) calculated as:
> max(sum_{i=1}^{n} xP_i * x_i) iff min_x f^T x) where f = -xP

The constraints are:
1. Budget (max 500 "crediti")
2. Positional quotas: Ax = b such as, with 3 GK, 8 D, 8 C and 6 A (25 players)
3. Binary bounds (0 <= x_i <= 1)

## The Structure:
- The Pre-season squad builder (namely, phase_1)
- The Pre-match formation builder (namely, phase_2)
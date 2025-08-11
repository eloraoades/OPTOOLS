# Game Plan for Simulator Functions

## P0 — Correctness & Consistency (do these first)

1. Units audit (degrees vs radians)
    - Surface_from_Efth* assumes θ in degrees (because of ±360 wrap).
    - Add an explicit theta_units: Literal["deg","rad"] param (default "deg"), or at least assert degrees.
    - Acceptance: a test that converting a known bimodal spectrum in deg vs rad (converted) yields identical Hs (within 1e-6) and similar Qkk.

2. FFT normalization is mixed
    - Some functions use norm="forward", others multiply by (nx*ny). Pick one convention (recommend norm="forward" everywhere).
    - After standardizing, verify variance energy: var(η) ≈ ∬ Z1 dkx dky (within tolerance).
    - Acceptance: unit tests pass for all four generators with same Z1.

3. Spectral step conversions (cycles/m vs rad/m)
    - In surface_from_Efth, wavespec_Efth_to_Ekxky expects cycles/m; you pass dkx/(2π). You then divide Ekxky by (2π)^2. Keep this but document it clearly at top-of-file.
    - Acceptance: numeric equality of ∑∑Ekxky·dkx·dky before/after helper call (post conversions).

4. floor() in spatial step
    - In two places: dy = np.floor(2*np.pi/((kY0[1]-kY0[0])*ny)). This silently distorts grid spacing.
    - Mark as bug for future removal; for now add a warning.
    - Acceptance: removing floor doesn’t change Hs by more than ~1e-6 in test spectra.

5. Phase distribution inconsistency
    - One path uses Gaussian phases in exp(i·2π·rg) (nonstandard), others use uniform [0,1).
    - Decide policy (likely uniform everywhere). Until then, document the intentional difference.
    - Acceptance: with uniform everywhere, output variance and spectra unchanged; only spatial realization changes.

6. Ambiguous names (kF = |k|)
    - kF is actually the wavenumber magnitude |k| [rad/m], not frequency.
    - Plan to rename to k_mag later, but for now add a #TODO rename + doc.

## P1 — API, Safety Nets, and Clarity

1. Input validation & assertions
    - Check monotonic spacing for kX0, kY0, f_vec, th_vec.
    - Assert positive dx, dy, dkx, dky, and consistency: abs(dkx - 2π/(dx·nx)) < tol.
    - Assert shapes: Z1 (ny,nx), kX,kY either (ny,nx) or (nx)/(ny).
    - Acceptance: meaningful ValueError with helpful messages.

2. Randomness control
    - Standardize RNG usage: always np.random.default_rng(seed); pass seed through; don’t call global RNG.
    - Optional future: accept an rng: np.random.Generator | None param.
    - Acceptance: same seed → identical fields bit-for-bit.

3. Consistent centering strategy
    - Use ifftshift consistently (preferred) instead of a mix of roll and ifftshift.
    - For now, keep both (per your request) but add #TODO unify centering.
    - Acceptance (future): replacing roll with ifftshift doesn’t change variance/energy.

4. Logging vs prints

    - Replace print(...) with a tiny logger (or verbose → logger.debug).
    - Acceptance: silence by default; user can enable verbose messages.

5. Type hints & imports cleanup

    - Currently, still have "from typing import Optional, tuple"—the lowercase tuple import is unnecessary/wrong. Use built-in tuple[...] (Py≥3.9) or from typing import Tuple (Py<3.9).
    - Find location wehere duplicate import numpy as np. Remove one.
    - Acceptance: mypy/pyright clean on this module.

## P2 — Performance & Ergonomics (safe to do later)

1. Grid reuse
    - Building kX0,kY0,kX,kY happens in multiple functions. Provide a small helper (but don’t change calls yet).
    - Acceptance: microbenchmark shows fewer allocations; no API changes.

2. Interpolation strategy
    - griddata(..., method='nearest') is robust but rough. Consider 'linear' with NaN handling + energy renorm.
    - Acceptance: Hs conservation still holds after renormalization; visual artifacts reduced.

3. dtypes
    - Consider float32 option for large grids to reduce memory.
    - Acceptance: outputs within tolerance; memory drop.

4. Return signature consistency
    - Return ordering/contents differ across functions (e.g., S1 vs (S1,S2), kz arrays included or not).
    - Plan a future API that returns a small dataclass (but not now).
    - Acceptance: thin wrappers maintain old signature while adding a canonical object.

## P3 — Documentation & Tests ( guardrails)

1. Module header with conventions
    Include at top of file:
    - θ default units (degrees).
    - kx,ky,dkx,dky units (rad/m).
    - FFT normalization policy (norm="forward").
    - Spectral integral convention: Var[η] = ∬ Z1 dkx dky.
    - Conversions when helpers expect cycles/m.

2. Unit tests
    - Variance conservation: For a synthetic Gaussian ring spectrum Z1, var(η) ≈ ∬ Z1 dkx dky within 1e-6.
    - Hs match: For Efth with known total variance, Hs_out ≈ Hs_in.
    - Symmetry/Reality: Output field is real-valued (imag residual ~ numerical noise).
    - Reproducibility: Same seed → identical output. Different seed → variance same.
    - Grid relations: dkx ≈ 2π/(dx·nx) and dky ≈ 2π/(dy·ny).
    - θ wrap: A narrow lobe near 0° equals same near 360° after wrap.
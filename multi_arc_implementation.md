# Constrained Multi-Arc Continuity for Tudat Orbit Determination

**Self-contained implementation specification.** Read this cold and you have
everything needed to implement the feature. Source-code anchors below are
verified against the current tree on branch
`constrained-multi-arc-orbit-estimation`. All paths are absolute under
`/Users/markusreichel/PhD/tudatpy`.

---

## 1. Background

### 1.1 The problem (Rosetta motivation)

Orbit determination of the ESA Rosetta spacecraft from radiometric tracking
data (range and Doppler) is being implemented in tudatpy. When the trajectory
is estimated as a **single unconstrained arc**, the least-squares estimator
converges to a global residual minimum that corresponds to a **physically
implausible trajectory**. The natural regularization is to split the data into
multiple arcs and add a soft continuity constraint between consecutive arcs.

### 1.2 Orbit14 constrained multi-arc strategy (Cicalò et al. 2021, §3.3)

Cicalò et al., *"Orbit determination methods for interplanetary missions:
development and use of the Orbit14 software"*, Experimental Astronomy 53
(2022) 159–208 — DOI 10.1007/s10686-021-09823-8. Notation reproduced here
exactly from the paper:

For two adjacent arcs `j` and `j+1`, the **jump** at the **conjunction time**
`t_c^j` is

```
d^{j,j+1} = Φ_{t_{j+1,0}}^{t_c^j}(y_{j+1,0}) − Φ_{t_{j,0}}^{t_c^j}(y_{j,0})      (Eq. 27)
```

i.e. the right-arc state propagated *backwards* to `t_c` minus the left-arc
state propagated *forwards* to `t_c`. In the paper's "extended arc" setting
both sides need to be propagated through an otherwise unobserved gap to meet
at `t_c`.

The constraint adds to the target function a new piece

```
Q_d = (1 / m_d) · (1 / μ) · Σ_{j=1..n-1}  (d^{j,j+1})^T  C_{j,j+1}  d^{j,j+1}    (Eq. 28)
```

where:

- `C_{j,j+1}` is a 6×6 symmetric positive-semidefinite weight matrix per
  boundary; the paper says it "sets the desired level of the jumps' size
  (e.g., meters in position, centimeters per second in velocity)".
- `μ > 0` is a single global scalar that "fixes the global weight of the
  jumps constraint in the target function". **Larger μ weakens the
  penalty.** This implementation follows the paper convention exactly.
- `m_d` is the total number of constrained components — e.g. `6·(n-1)` for
  full-state continuity across `n` arcs, `3·(n-1)` for position-only.

The new design matrix in the paper is `B = ∂(ξ, d) / ∂x` — i.e. the paper
**row-stacks** continuity rows under the observation rows and uses a
block-diagonal weight. This implementation prefers **direct normal-equation
injection** (mathematically equivalent, numerically more robust for
rank-deficient `C`; see §4.9).

### 1.3 Why the weight matrix C matters — design intent

This must be a **general-purpose, reusable** capability. The 6×6 PSD matrix
`C` controls **which components are constrained and how tightly**, so a single
code path covers (at minimum):

- **Full 6-DOF continuity** — both position and velocity (`C = diag(σ_r², σ_r², σ_r², σ_v², σ_v², σ_v²)` or similar).
- **Position-only continuity** — `C = diag(σ_r², σ_r², σ_r², 0, 0, 0)`. Rank 3.
- **Velocity-only continuity** — the inverse. Rank 3.
- **Arbitrary per-component or dense PSD weighting**, per-boundary if needed.

Users configure by supplying `C` (or via a convenience preset) and `μ`, both
as global values or per-boundary overrides.

### 1.4 The Rosetta configuration (one specialization)

- Arcs split at **Orbit Correction Maneuvers (OCMs)**. The connection epoch
  `t_c` between arc `k` and arc `k+1` **is** the OCM time — adjacent arcs,
  no unobserved gap.
- The OCM Δv is **not estimated** (its uncertainty is too large), so there
  is a genuine velocity discontinuity at each boundary.
- Therefore: **position-only continuity** — `C` weights the 3 position
  components, velocity rows/columns are zero.

### 1.5 Shared-boundary vs mid-gap connection epoch — both supported

| Case                  | `t_c` location              | Left arc                       | Right arc                       |
|-----------------------|-----------------------------|--------------------------------|---------------------------------|
| Shared OCM boundary   | `arc_left.end == arc_right.start` | propagated to its end          | trivially at its start (Φ = I)  |
| Orbit14 mid-gap       | Inside a gap between arcs   | extended forward past its data | extended backward before its data |

**Both cases use the same code path.** The only user responsibility is to
extend each arc's propagation window to cover `t_c`. The validator enforces
`t_c ∈ [arc.start, arc.end]` for both adjacent arcs.

---

## 2. Math used by the implementation

### 2.1 Discrepancy and partials

For a constrained pair `(k_left, k_right)` at epoch `t_c`:

```
d = x_{right}(t_c) − x_{left}(t_c)              ∈ ℝ^6      (physical units)

D = ∂d / ∂x_estimated  =  M_{right}(t_c) − M_{left}(t_c)              (6 × N_params)
```

where `M_k(t)` is the per-arc full-parameter mapping matrix — the
`(6 × N_params)` block consisting of:

- The arc's 6×6 STM `Φ_k(t)` placed at the arc-k initial-state column slot.
- Zeros at all other arcs' initial-state column slots.
- The arc's `6 × P_global` sensitivity matrix `S_k(t)` placed at the global
  parameter columns.

The multi-arc STM interface in tudat already returns exactly this layout —
see §3.4.

**At a shared OCM boundary** (`t_c == arc_right.start_time`):
- `Φ_{right}(t_c)` = `I_6`, `S_{right}(t_c)` = `0`. Falls out of the
  variational propagator with no special-casing — but verify with a unit
  test (§6 test 7).

### 2.2 Cost contribution

With paper convention `Q_d = (1/m_d)(1/μ) Σ d^T C d`, the per-pair
contribution to the cost is

```
q_pair = (1 / (μ · m_d)) · d^T C d
```

### 2.3 Normal-equation injection

Tudat residual convention is `r = observed − computed`. For the continuity
target the "observed" jump is zero and the "computed" jump is `d`, so the
constraint residual is `−d`. The paper's row-stacking
`B = ∂(ξ, d)/∂x` with diagonal weight blocks `W_obs ⊕ W_d` produces normal
equations whose **continuity contribution** is

```
H_constraint += D^T  W_d  D
g_constraint += − D^T  W_d  d                  with  W_d = (1 / (μ · m_d)) · C
```

Direct injection adds these to the assembled `H = B^T W B (+ a-priori)` and
`g = B^T W r` of the observation-only problem. See §4.2 / §4.4 for the API.

### 2.4 Normalization

Tudat column-normalizes the observation design matrix:

```
H_obs_norm = D_obs^T W D_obs ,  where each column of D_obs is divided by a
                                 column scale stored in `normalizationTerms`.
```

To keep the continuity contribution in the same normalized coordinates,
divide each column of the continuity Jacobian by the *same* normalization
factor before computing `H_constraint`, `g_constraint`:

```
D_norm(:, j) = D(:, j) / normalizationTerms(j)

H_constraint += D_norm^T  W_d  D_norm
g_constraint += − D_norm^T  W_d  d
```

`d` itself stays in **physical units** — the scaling already lives in
`D_norm`.

The cost `q_pair` is computed from the physical-units `d` and `W_d` so its
reported value is scale-invariant.

### 2.5 m_d accounting

`m_d = Σ_pair rank(C_pair)`, where the rank is the numerical rank within a
small PSD tolerance (e.g. 1e-12 relative to the largest eigenvalue). For
`diag(1,1,1,0,0,0)` rank is 3; for full `I_6` rank is 6. This makes the
behavior identical to the paper for paper-style symmetric weighting and
falls back gracefully for mixed configurations.

---

## 3. How tudat works today (verified anchors)

### 3.1 Least-squares solver

`src/tudat/math/basic/leastSquaresEstimation.cpp:124-162` — primary entry:

```cpp
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
performLeastSquaresAdjustmentFromDesignMatrix(
    const Eigen::MatrixXd& designMatrix,
    const Eigen::VectorXd& observationResiduals,
    const Eigen::VectorXd& diagonalOfWeightMatrix,
    const Eigen::MatrixXd& inverseOfAPrioriCovarianceMatrix,
    const double limitConditionNumberForWarning = 1.0E8,
    const Eigen::MatrixXd& constraintMultiplier   = Eigen::MatrixXd(0,0),
    const Eigen::VectorXd& constraintRightHandside = Eigen::VectorXd(0),
    const Eigen::MatrixXd& designMatrixConsiderParameters = Eigen::MatrixXd(0,0),
    const Eigen::VectorXd& considerParametersDeviations    = Eigen::VectorXd(0));
```

Header declaration: `include/tudat/math/basic/leastSquaresEstimation.h:129-138`.

Key facts:

- Weights are diagonal only (`VectorXd diagonalOfWeightMatrix`,
  multiplication at cpp:52-64).
- Normal matrix `H` is built in
  `calculateInverseOfUpdatedCovarianceMatrix(...)` at cpp:66-99,
  line 74-75: `H = inverseOfAPrioriCovarianceMatrix + D^T W D`.
- Hard-equality constraints
  (`constraintMultiplier` / `constraintRightHandside`) are Lagrange-multiplier
  rows appended to LHS and RHS, **resizing both** to `(n + n_c)`:
  - LHS resize at cpp:91-95 inside
    `calculateInverseOfUpdatedCovarianceMatrix(...)`.
  - RHS resize at cpp:156-157 in
    `performLeastSquaresAdjustmentFromDesignMatrix(...)`.
- RHS proper is built at cpp:135-145 as `D^T W r`.

**Other overloads** (cpp:165-186, header:152-172): without a-priori, without
weights — all delegate to the primary. They need not be touched.

### 3.2 OD manager iteration loop

`include/tudat/simulation/estimation_setup/orbitDeterminationManagerEstimationImplementation.h`:

- Loop start: line 105.
- `performPreEstimationSteps(...)` call: lines 112-117 — returns
  `((designMatrixEstimatedParameters, designMatrixConsiderParameters), residuals)`.
- Column-normalize design matrix: line 130
  `normalizeDesignMatrix(designMatrixEstimatedParameters)` returns
  `normalizationTerms` (Eigen::VectorXd).
- Normalize a-priori covariance with the same factors: lines 131-133.
- Hard-constraint multiplier from parameters: line 166
  `parametersToEstimate_->getConstraints(constraintStateMultiplier, constraintRightHandSide)`.
- LSQ call: lines 174-183.
- Cost computation: lines 221-222
  `costFunction = linear_algebra::computeLeastSquaresCostFunction(weightsMatrixDiagonals, residuals)`.
- Best-iteration check: line 258
  `if (costFunction < bestCostFunction || !(bestCostFunction == bestCostFunction))`.
- EstimationOutput construction: lines 322-338.

**Access to multi-arc machinery from inside the loop:**

- The manager exposes
  `getVariationalEquationsSolver()`
  (`include/tudat/simulation/estimation_setup/orbitDeterminationManager.h:213`)
  and
  `getStateTransitionAndSensitivityMatrixInterface()`
  (`...orbitDeterminationManager.h:244`) — the latter returns the **base**
  `CombinedStateTransitionAndSensitivityMatrixInterface`.
- Inside the iteration template the corresponding members
  `variationalEquationsSolver_` and
  `stateTransitionsAndSensitivityMatrixInterface_` are visible directly.
- Downcast both to the multi-arc variant:
  ```cpp
  auto multiArcStmInterface = std::dynamic_pointer_cast<
      propagators::MultiArcCombinedStateTransitionAndSensitivityMatrixInterface >(
          stateTransitionsAndSensitivityMatrixInterface_ );
  auto multiArcSimulator = std::dynamic_pointer_cast<
      propagators::MultiArcDynamicsSimulator<ObservationScalarType, TimeType> >(
          variationalEquationsSolver_->getDynamicsSimulator( ) );
  ```
  Throw if either is null with constraints present.

### 3.3 Multi-arc parameter layout

`include/tudat/astro/orbit_determination/estimatable_parameters/initialTranslationalState.h:138-349`

`ArcWiseInitialTranslationalStateParameter`:
- Members: `initialTranslationalState_` (`Eigen::Matrix`, 6×n_arcs concatenated as a column vector), `arcStartTimes_`, `centralBodies_`, `frameOrientation_`.
- `getParameterSize()` → `6 * arcStartTimes_.size()`.
- `getArcStartTimes()` → `std::vector<double>` (lines 275-284).

Global parameter vector ordering in `EstimatableParameterSet`
(`include/tudat/astro/orbit_determination/estimatable_parameters/estimatableParameterSet.h:91-122`):

```
[ single-arc initial states | multi-arc initial states | scalar params | vector params ]
```

with multi-arc states inserted via lines 91-95. Each `ArcWiseInitialTranslationalStateParameter` is a single 6·n_arcs block; arc-k 6-block sits at `block_offset + 6·k` inside it. **Use the parameter set's index map (`parameterIndices_`) to locate the block offset — do not assume offsets by hand.**

### 3.4 Multi-arc STM and sensitivity interface

`include/tudat/astro/propagators/stateTransitionMatrixInterface.h:247-931`

`MultiArcCombinedStateTransitionAndSensitivityMatrixInterface`:

- `getCurrentArc(double evaluationTime)` (lines 624-641): returns
  `std::pair<int, double>` (arcIndex, propagationStartTime). Uses
  `findNearestLowerNeighbour` over arc start times then checks
  `t ∈ [arcStart, arcEnd]`. **At a shared boundary it returns the left
  arc.** Returns `(-1, NaN)` outside any arc.
- `getCombinedStateTransitionAndSensitivityMatrix(double time)` (lines
  457-526): returns `(6 × (6 + P_global))` for the time-resolved arc only.
- `getFullCombinedStateTransitionAndSensitivityMatrix(double time)` (lines
  536-616): returns
  `(fullStateSize_, fullStateTransitionMatrixSize_ + fullSensitivityMatrixSize_)` —
  inactive arc rows/columns zero. **This is the right per-time shape**, but
  cannot retrieve `M_right(t_c)` at a shared boundary because the time
  lookup picks the left arc.
- Per-arc-index accessors exist for sizes (lines 696-709) but **not** for
  evaluating the matrix at a specific time. We add one (§4.1).
- Backing storage: per-arc interpolators built during multi-arc variational
  propagation, defined on `[arcStart, arcEnd]`. Arc bounds in
  `arcStartTimes_`, `arcEndTimes_` (line 872), private.

### 3.5 Multi-arc dynamics simulator and per-arc results

`include/tudat/simulation/propagation_setup/multiArcDynamicsSimulator.h` and
`include/tudat/simulation/propagation_setup/propagationResults.h`.

- `MultiArcSimulationResults::getSingleArcResults()` → vector of
  per-arc `SingleArcSimulationResults`.
- `SingleArcSimulationResults::getEquationsOfMotionNumericalSolution()` →
  `std::map<TimeType, StateVector>` of propagated state at integrator steps.
- `MultiArcSimulationResults::getArcStartTimes()` / `getArcEndTimes()`
  (lines 702-710 of multiArcDynamicsSimulator.h).
- **Interpolation required** for arbitrary `t_c` — the state map is at
  integrator steps only.
- Reusable Lagrange (8th-order) helper for arc-to-arc handoff:
  `getArcInitialStateFromPreviousArcResult(...)` at lines 48-107 of
  `multiArcDynamicsSimulator.h`. Same pattern works for `t_c` evaluation.

### 3.6 Input / output types

`include/tudat/astro/orbit_determination/podInputOutputTypes.h`:

- `CovarianceAnalysisInput` (lines 35-521): base class; holds observation
  collection, `inverseOfAprioriCovariance_` (line 497),
  `considerCovariance_`, etc.
- `EstimationInput` (lines 595-722): extends with `convergenceChecker_`,
  `saveResidualsAndParametersFromEachIteration_`, etc. Constructor lines
  606-641.
- `EstimationOutput` (lines 1043-1227): extends `CovarianceAnalysisOutput`.
  Constructor at 1062-1091 has 14+ positional args. Per-iteration fields:
  `residualHistory_` (1209), `parameterHistory_` (1212),
  `simulationResultsPerIteration_` (1219), `bestIteration_` (1203).

### 3.7 Settings-object pattern to mirror

`src/tudatpy/estimation/observations_setup/viability/expose_viability.cpp:77-215`
is the canonical pattern: a `py::class_` for the settings type plus free
factory functions (`elevation_angle_viability(...)`,
`body_avoidance_viability(...)`). Mirror it for the new settings.

### 3.8 Covariance-only path

`include/tudat/simulation/estimation_setup/orbitDeterminationManagerCovarianceImplementation.h:82-88`
— one-shot normal-matrix assembly via
`calculateInverseOfUpdatedCovarianceMatrix(...)`. No residual / RHS; needs
the `H_constraint` contribution only.

### 3.9 No existing inter-arc continuity utility

Verified by grep: no `interArc`, `ArcContinuity`, `ArcConnect`, `ArcBridge`,
`continuityConstraint` symbols in the tree. Clean gap to fill.

---

## 4. Design

### 4.1 New per-arc-index STM accessor

Add two methods on `MultiArcCombinedStateTransitionAndSensitivityMatrixInterface`
(`include/tudat/astro/propagators/stateTransitionMatrixInterface.h`):

```cpp
// Returns the 6 × (6 + P_global) STM+sensitivity for the explicitly named arc,
// evaluated at time t. t must lie in [arcStartTimes_[arcIndex], arcEndTimes_[arcIndex]].
Eigen::MatrixXd getCombinedStateTransitionAndSensitivityMatrixForArc(
    int arcIndex, double evaluationTime ) const;

// Returns the (fullStateSize_, fullStateTransitionMatrixSize_ + fullSensitivityMatrixSize_)
// padded layout (zero in inactive arcs' state-block columns), for a specific arcIndex.
Eigen::MatrixXd getFullCombinedStateTransitionAndSensitivityMatrixForArc(
    int arcIndex,
    double evaluationTime,
    bool addCentralBodyDependency = true,
    const std::vector<std::string>& arcDefiningBodies = {} ) const;
```

These are direct generalizations of the existing time-keyed methods
(lines 457-526 and 536-616) that skip the `getCurrentArc(...)` lookup and
take `arcIndex` as input. The shared-boundary identity-Φ / zero-S property
falls out of the underlying per-arc interpolator at `t == arcStart`.

### 4.2 Least-squares solver — optional normal/RHS additions

Extend the primary overload in
`src/tudat/math/basic/leastSquaresEstimation.cpp:124-162` and its declaration
in `include/tudat/math/basic/leastSquaresEstimation.h:129-138` with two
optional trailing arguments:

```cpp
const Eigen::MatrixXd& additionalNormalMatrix   = Eigen::MatrixXd(0,0),
const Eigen::VectorXd& additionalRightHandSide  = Eigen::VectorXd(0)
```

Apply them in the **parameter block** of the assembled matrices, so the
addition composes correctly with the hard-equality Lagrange augmentation:

```cpp
// After calculateInverseOfUpdatedCovarianceMatrix(...) and the constraint RHS resize:
const int n = designMatrix.cols();
if( additionalNormalMatrix.size() > 0 ) {
    if( additionalNormalMatrix.rows() != n || additionalNormalMatrix.cols() != n ) {
        throw std::runtime_error("additionalNormalMatrix wrong shape");
    }
    inverseOfCovarianceMatrix.topLeftCorner(n, n) += additionalNormalMatrix;
}
if( additionalRightHandSide.size() > 0 ) {
    if( additionalRightHandSide.size() != n ) {
        throw std::runtime_error("additionalRightHandSide wrong size");
    }
    rightHandSide.head(n) += additionalRightHandSide;
}
```

Defaults are empty → zero behavior change for existing callers.

### 4.3 Settings object

New header
`include/tudat/simulation/estimation_setup/interArcStateContinuityConstraintSettings.h`:

```cpp
class InterArcStateContinuityConstraintSettings {
public:
    InterArcStateContinuityConstraintSettings(
        std::string body,
        std::vector<double> connectionEpochs,
        std::vector<Eigen::Matrix<double,6,6>> weightMatrices,   // size 1 or n_pairs
        std::vector<double> muValues,                            // size 1 or n_pairs
        std::vector<std::pair<int,int>> arcPairs = {} );         // empty → all (i, i+1)

    // Validated in constructor; throws on inconsistency.

    std::string body_;
    std::vector<std::pair<int,int>> arcPairs_;
    std::vector<double> connectionEpochs_;
    std::vector<Eigen::Matrix<double,6,6>> weightMatrices_;
    std::vector<double> muValues_;
};
```

**Validation rules** (settings-time, runtime-independent):

- `mu > 0` for every entry.
- Every `C` is exactly 6×6 and symmetric within a tolerance
  (`(C − C^T).norm() < 1e-12 · C.norm()`).
- Every `C` has eigenvalues ≥ `−1e-12 · |λ_max|` (PSD within tolerance).
  Rank-deficient `C` is allowed.
- `connectionEpochs_.size() == arcPairs_.size()`.
- `weightMatrices_.size() ∈ {1, arcPairs_.size()}`.
- `muValues_.size() ∈ {1, arcPairs_.size()}`.
- If `arcPairs_` is non-empty, each entry has `right == left + 1` (v1
  restriction: consecutive arcs only). Both indices in `[0, n_arcs)` —
  this index check is deferred to assembly time once n_arcs is known.

**Convenience preset builders** (free functions in same header):

```cpp
std::shared_ptr<InterArcStateContinuityConstraintSettings>
fullStateContinuity(
    std::string body, std::vector<double> epochs,
    double positionWeight = 1.0, double velocityWeight = 1.0,
    double mu = 1.0, std::vector<std::pair<int,int>> arcPairs = {} );

std::shared_ptr<InterArcStateContinuityConstraintSettings>
positionOnlyContinuity(
    std::string body, std::vector<double> epochs,
    double positionWeight = 1.0,
    double mu = 1.0, std::vector<std::pair<int,int>> arcPairs = {} );

std::shared_ptr<InterArcStateContinuityConstraintSettings>
velocityOnlyContinuity(
    std::string body, std::vector<double> epochs,
    double velocityWeight = 1.0,
    double mu = 1.0, std::vector<std::pair<int,int>> arcPairs = {} );

std::shared_ptr<InterArcStateContinuityConstraintSettings>
generalContinuity(
    std::string body, std::vector<double> epochs,
    std::vector<Eigen::Matrix<double,6,6>> weightMatrices,    // size 1 or n_pairs
    double mu = 1.0, std::vector<std::pair<int,int>> arcPairs = {} );
```

### 4.4 Per-iteration assembly module

New files:
- `include/tudat/simulation/estimation_setup/interArcContinuityConstraint.h`
- `src/tudat/simulation/estimation_setup/interArcContinuityConstraint.cpp`

Public types and function:

```cpp
struct InterArcConstraintContribution {
    Eigen::MatrixXd additionalNormalMatrix;        // n × n, normalized
    Eigen::VectorXd additionalRightHandSide;       // n, normalized
    double          totalConstraintCost;           // Σ_pair (1/(μ·m_d)) d^T C d
    std::vector<Eigen::Matrix<double,6,1>> perPairDiscrepancies;  // d for each pair, this iter
};

template<typename ObservationScalarType, typename TimeType>
InterArcConstraintContribution assembleInterArcContinuityContribution(
    const std::vector< std::shared_ptr<InterArcStateContinuityConstraintSettings> >&
        constraintSettings,
    const std::shared_ptr< estimatable_parameters::EstimatableParameterSet<ObservationScalarType> >&
        parametersToEstimate,
    const std::shared_ptr< propagators::MultiArcDynamicsSimulator<ObservationScalarType, TimeType> >&
        multiArcSimulator,
    const std::shared_ptr< propagators::MultiArcCombinedStateTransitionAndSensitivityMatrixInterface >&
        stmInterface,
    const Eigen::VectorXd& columnNormalizationFactors,
    int totalParameterSize );
```

Algorithm, per `InterArcStateContinuityConstraintSettings` in the vector:

1. Resolve the body's `ArcWiseInitialTranslationalStateParameter` from
   `parametersToEstimate`. Locate its global column offset via the
   parameter set's index helpers. Read `n_arcs` from the parameter.
2. Resolve the per-pair `(C_k, μ_k)` via the broadcasting rule.
3. Compute `m_d_total = Σ_pair rank(C_k)` where rank is the count of
   eigenvalues > `1e-12 · |λ_max|`.
4. For each pair `(k_left, k_right)`:
   - **Validate** `t_c ∈ [arc.start, arc.end]` for both arcs. Throw
     with codex's wording:
     ```
     Inter-arc continuity connection epoch <t_c> for body <name> and arc pair
     (<left>, <right>) is outside the propagated interval of arc <which>
     [<start>, <end>]. Extend the arc propagation interval or change the
     connection epoch.
     ```
     Validate `right == left + 1` and `0 ≤ left, right < n_arcs`.
   - **Evaluate states**: `x_left(t_c)` and `x_right(t_c)` via
     `evaluateArcStateAtTime(...)` (see §4.5). Compute
     `d = x_right − x_left`.
   - **Evaluate partials**: `M_left = getFullCombinedStateTransitionAndSensitivityMatrixForArc(k_left, t_c)`
     and `M_right = getFullCombinedStateTransitionAndSensitivityMatrixForArc(k_right, t_c)`.
   - **Build sparse D** (6 × N_params): start from
     `M_right - M_left` over the `getFullCombined...` layout, then map
     columns into the LSQ's `totalParameterSize` if the global parameter
     vector ordering differs. (In practice the layouts agree — verify
     with a finite-difference test in §6.)
   - **Normalize columns**:
     `D_norm.col(j) = D.col(j) / columnNormalizationFactors(j)`.
   - **Accumulate**:
     ```
     W_d = (1.0 / (mu_k * m_d_total)) * C_k     // 6×6
     H_constraint += D_norm^T * W_d * D_norm
     g_constraint += − D_norm^T * (W_d * d)
     totalConstraintCost += d^T * W_d * d
     perPairDiscrepancies.push_back(d);
     ```

Multiple settings entries (multi-body) accumulate into the same returned
contribution.

### 4.5 Per-arc state evaluator

Free helper, in the same `.cpp` (private translation-unit scope or in a
detail namespace):

```cpp
Eigen::Matrix<double,6,1> evaluateArcStateAtTime(
    const std::shared_ptr< propagators::SingleArcSimulationResults<...> >& arc,
    double t,
    double arcInitialTime,
    double arcFinalTime );
```

Implementation:

- If `|t − arcInitialTime| < 1e-9` → return the stored initial state
  (handles the right-arc side of a shared OCM boundary exactly,
  avoiding interpolation noise at the arc boundary).
- Else if `|t − arcFinalTime| < 1e-9` → return the last sample directly.
- Else → 8th-order Lagrange interpolation against the local neighborhood
  of `arc->getEquationsOfMotionNumericalSolution()`, mirroring
  `getArcInitialStateFromPreviousArcResult` at
  `multiArcDynamicsSimulator.h:48-107`.

### 4.6 OD manager wiring — iteration loop

Edit
`include/tudat/simulation/estimation_setup/orbitDeterminationManagerEstimationImplementation.h`
inside `estimateParameters`:

1. **Once before the loop** (after `estimateParameters` start, ~line 90):
   - If `estimationInput->getInterArcContinuityConstraints().size() > 0`:
     downcast `stateTransitionsAndSensitivityMatrixInterface_` and
     `variationalEquationsSolver_->getDynamicsSimulator()` to multi-arc
     types. Throw if either cast is null.

2. **Inside the loop**, right after normalization (line ~133):
   ```cpp
   InterArcConstraintContribution interArcContribution;
   if( !estimationInput->getInterArcContinuityConstraints().empty() ) {
       interArcContribution = assembleInterArcContinuityContribution(
           estimationInput->getInterArcContinuityConstraints(),
           parametersToEstimate_,
           multiArcSimulator,
           multiArcStmInterface,
           normalizationTerms,
           numberEstimatedParameters_ );
   }
   ```

3. **Pass to the LSQ call** (line ~174-183) — append the two optional
   args after the existing `normalizedConsiderParametersDeviation`:
   ```cpp
   ..., normalizedConsiderParametersDeviation,
   interArcContribution.additionalNormalMatrix,
   interArcContribution.additionalRightHandSide );
   ```

4. **Combine into cost** (line ~221, just after `costFunction = ...`):
   ```cpp
   costFunction += interArcContribution.totalConstraintCost;
   ```
   (This is the cost used at the best-iteration check on line 258. The
   observation residual vector and `residualRms` are unchanged.)

5. **Record history** — push back into two new vectors local to
   `estimateParameters`, then pass into `EstimationOutput` (§4.8):
   - `interArcContinuityCostHistory.push_back(interArcContribution.totalConstraintCost);`
   - `interArcContinuityDiscrepancyHistory.push_back(interArcContribution.perPairDiscrepancies);`

### 4.7 OD manager wiring — covariance-only path

Edit
`include/tudat/simulation/estimation_setup/orbitDeterminationManagerCovarianceImplementation.h`
around lines 82-88: after the `calculateInverseOfUpdatedCovarianceMatrix(...)`
call, if constraints are present, recompute the same `H_constraint` from
§4.4 (RHS is unused for covariance) and add to the returned normal matrix:

```cpp
inverseNormalizedCovariance += interArcContribution.additionalNormalMatrix;
```

### 4.8 EstimationInput / EstimationOutput extensions

`include/tudat/astro/orbit_determination/podInputOutputTypes.h`:

- Add to `CovarianceAnalysisInput`:
  ```cpp
  private:
      std::vector< std::shared_ptr<InterArcStateContinuityConstraintSettings> >
          interArcContinuityConstraints_;
  public:
      void setInterArcContinuityConstraints(
          const std::vector< std::shared_ptr<InterArcStateContinuityConstraintSettings> >& s )
      { interArcContinuityConstraints_ = s; }
      const std::vector< std::shared_ptr<InterArcStateContinuityConstraintSettings> >&
          getInterArcContinuityConstraints() const
      { return interArcContinuityConstraints_; }
  ```
  Visible through inheritance on `EstimationInput`.

- Add to `EstimationOutput`:
  ```cpp
  std::vector<double> interArcContinuityCostHistory_;
  std::vector< std::vector<Eigen::Matrix<double,6,1>> >
      interArcContinuityDiscrepancyHistory_;
  ```
  Extend the constructor with two **trailing optional** args defaulting to
  empty vectors. All existing call sites continue to compile unchanged.
  Add getters.

### 4.9 Rejected alternatives (do not implement these)

- **`EstimatableParameterSet::getConstraints()` →
  `constraintMultiplier`/`constraintRightHandside`** is a *hard equality*
  via Lagrange multipliers. Cannot express tunable μ; diverges from the
  Orbit14 formulation.
- **`inverseOfAPrioriCovariance` alone** adds a PSD block to the LHS but
  has no companion RHS contribution; can only express priors centered on
  the current estimate, not relative inter-arc constraints with nonzero
  `d`.
- **Whitened pseudo-observation row stacking** (Cholesky-factor
  `(C/μ) = L L^T`, append `L^T D` rows to the design matrix and `L^T d`
  to residuals with unit weights) is mathematically equivalent but
  requires rank-revealing Cholesky for rank-deficient `C`
  (position-only is rank 3) and pollutes the residual vector with
  non-observations.

### 4.10 Python exposure

New file
`src/tudatpy/estimation/estimation_analysis/expose_inter_arc_constraints.cpp`:

- `py::class_` for `InterArcStateContinuityConstraintSettings` (no methods
  exposed — used as the return type from factories).
- Free functions:
  - `full_state_continuity(body, epochs, position_weight=1.0, velocity_weight=1.0, mu=1.0, arc_pairs=None)`
  - `position_only_continuity(body, epochs, position_weight=1.0, mu=1.0, arc_pairs=None)`
  - `velocity_only_continuity(body, epochs, velocity_weight=1.0, mu=1.0, arc_pairs=None)`
  - `general_continuity(body, epochs, weight_matrices, mu=1.0, arc_pairs=None)`
- All factory functions accept `position_weight`/`velocity_weight` as
  either scalar or length-3 sequence (anisotropic) — convert to a
  diagonal 6×6 inside the binding.

Auto-collected into the kernel by `update_sources(estimation sources)` in
`src/tudatpy/CMakeLists.txt` — **no CMake edit needed for Python.**

Edit `src/tudatpy/estimation/estimation_analysis/expose_estimation_analysis.cpp`:

- Extend the `CovarianceAnalysisInput` binding around lines 337-549 with:
  ```cpp
  .def("set_inter_arc_continuity_constraints",
       &tss::CovarianceAnalysisInput<...>::setInterArcContinuityConstraints,
       py::arg("constraints"))
  .def_property_readonly("inter_arc_continuity_constraints",
       &tss::CovarianceAnalysisInput<...>::getInterArcContinuityConstraints)
  ```
  (Inherited by `EstimationInput`.)
- Extend the `EstimationOutput` binding around lines 884-948 with two
  read-only properties:
  - `inter_arc_continuity_cost_history`
  - `inter_arc_continuity_discrepancy_history`

---

## 5. Files to create and modify

### 5.1 New C++ files

| Path | Contents |
|------|----------|
| `include/tudat/simulation/estimation_setup/interArcStateContinuityConstraintSettings.h` | Settings class, validation, four preset builders. |
| `include/tudat/simulation/estimation_setup/interArcContinuityConstraint.h` | `InterArcConstraintContribution` struct + `assembleInterArcContinuityContribution(...)` decl. |
| `src/tudat/simulation/estimation_setup/interArcContinuityConstraint.cpp` | Validation, `evaluateArcStateAtTime`, sparse `D` assembly, normalization, accumulation. |

### 5.2 Modified C++ files (with verified line anchors)

| Path | Anchor | Change |
|------|--------|--------|
| `include/tudat/math/basic/leastSquaresEstimation.h` | 129-138 | Add 2 optional trailing args to primary overload declaration. |
| `src/tudat/math/basic/leastSquaresEstimation.cpp` | 124-162 | Add 2 optional args; insert top-left `n×n` / `head(n)` additions after the constraint resize. |
| `include/tudat/astro/propagators/stateTransitionMatrixInterface.h` | 457-616 | Add `*ForArc(arcIndex, t, ...)` per-arc-index accessors on `MultiArcCombined...`. |
| `include/tudat/astro/orbit_determination/podInputOutputTypes.h` | 35-722, 1043-1227 | Add `interArcContinuityConstraints_` member + accessors on `CovarianceAnalysisInput`; add two trailing optional ctor args + members on `EstimationOutput`. |
| `include/tudat/simulation/estimation_setup/orbitDeterminationManagerEstimationImplementation.h` | 90, 130, 174, 221, 322 | Downcast multi-arc; assemble per iteration; pass to LSQ; combine cost; populate output history. |
| `include/tudat/simulation/estimation_setup/orbitDeterminationManagerCovarianceImplementation.h` | 82-88 | Add `H_constraint` to one-shot covariance assembly. |

### 5.3 CMake

| Path | Change |
|------|--------|
| `src/tudat/simulation/estimation_setup/CMakeLists.txt` | Add `interArcContinuityConstraint.cpp` to that target's `_SOURCES` set; add the new headers to its `_HEADERS` set. |
| `tests/test_tudat/src/astro/orbit_determination/CMakeLists.txt` | Add `TUDAT_ADD_TEST_CASE(InterArcContinuityConstraint ...)` line next to the existing `MultiArcStateEstimation` test. |

No CMake edit for Python — `src/tudatpy/CMakeLists.txt` auto-collects via
`update_sources(estimation sources)`.

### 5.4 New / modified Python files

| Path | Change |
|------|--------|
| `src/tudatpy/estimation/estimation_analysis/expose_inter_arc_constraints.cpp` | New file: bind settings class + four factories. |
| `src/tudatpy/estimation/estimation_analysis/expose_estimation_analysis.cpp` | Extend `CovarianceAnalysisInput` and `EstimationOutput` bindings. |

### 5.5 New tests

| Path | Coverage |
|------|----------|
| `tests/test_tudat/src/astro/orbit_determination/unitTestInterArcContinuityConstraint.cpp` | All tests in §6. |

---

## 6. Tests (C++ unit tests; pattern: existing `unitTestMultiArcStateEstimation.cpp`)

1. **Position-only OCM shared boundary**. Two arcs adjacent at `t_c`, no
   observations, strong penalty (small μ). Assert position discrepancy
   goes to zero, velocity discrepancy unchanged.
2. **Full-state preset**. Both position and velocity discrepancies driven
   to zero.
3. **Velocity-only preset**. Symmetric inverse of test 1.
4. **Dense PSD weight**. Use a rank-deficient symmetric PSD 6×6. Assert
   no Cholesky failure, no artificial rank requirement.
5. **Per-boundary heterogeneous weights**. Two boundaries with different
   `(C, μ)` — verify linear superposition of contributions.
6. **Validation: `t_c` outside arc**. Assert the exact error message,
   naming arc index and bounds.
7. **Shared-boundary identity Φ**. At `t_c == arc_right.start`, assert
   `M_right(t_c)` has `I_6` in arc_right's 6-block and zeros elsewhere
   for non-arc state columns and zeros in the sensitivity block.
8. **Mid-gap Orbit14 case**. Arcs explicitly extended through a gap,
   `t_c` at midpoint, both Φ blocks non-trivial. Compare against an
   analytic reference.
9. **Finite-difference `D`**. Numerically perturb each multi-arc initial
   state and global parameter; compare with the analytical `D`.
   Tolerance ≤ `1e-6` relative. Catches multi-arc parameter offset
   bugs.
10. **Normalization invariance**. Run identical problems with different
    column normalization (artificially scaled parameters); assert the
    final parameter update is identical to working tolerance.
11. **Empty additions no-op (LSQ unit test)**. With empty
    `additionalNormalMatrix`/`additionalRightHandSide`, the LSQ output
    bit-equals the un-extended call.
12. **Covariance-only tightens**. Run `computeCovariance(...)` with and
    without continuity; assert the constrained covariance shrinks in the
    weighted directions.
13. **Best-iteration selection**. Construct a case where observation-only
    cost and combined cost prefer different iterations. Assert
    `bestIteration_` reflects combined cost.
14. **Python round-trip**. Build settings via each factory; attach to
    `EstimationInput`; read back `EstimationOutput.inter_arc_continuity_cost_history`.

---

## 7. End-to-end Rosetta verification

After unit tests pass:

1. Build a multi-arc estimator with arcs split at OCMs (epochs from the
   Rosetta SPICE kernels).
2. Configure:
   ```python
   from tudatpy.estimation import estimation_analysis as ea

   constraints = [
       ea.position_only_continuity(
           body="Rosetta",
           epochs=ocm_times,
           position_weight=1.0,  # 1 unit penalty per (1 m)^2 jump
           mu=mu_user,
       )
   ]
   estimation_input.set_inter_arc_continuity_constraints(constraints)
   ```
3. Run `estimator.perform_estimation(estimation_input)` and confirm:
   - Estimator converges to a physically plausible trajectory (visually
     compare against SPICE reference).
   - Position discrepancies at OCM boundaries shrink as μ decreases.
   - Velocity discrepancies at OCM boundaries remain free.
   - Observation residuals degrade smoothly with shrinking μ.
4. Inspect `EstimationOutput.inter_arc_continuity_cost_history` and
   `inter_arc_continuity_discrepancy_history` to verify per-iteration
   evolution.

---

## 8. Implementation order

1. **STM `ForArc` accessors** (§4.1) + identity-Φ-at-arc-start unit test
   (test 7).
2. **LSQ optional additions** (§4.2) + empty-additions-no-op unit test
   (test 11).
3. **Settings class** (§4.3) + validation unit tests.
4. **Assembly module** (§4.4 / §4.5) + finite-difference + normalization
   tests (tests 9, 10).
5. **Input/Output wiring** (§4.8): `CovarianceAnalysisInput` member +
   `EstimationOutput` extended constructor.
6. **OD manager iteration wiring** (§4.6) + tests 1–5, 8, 13.
7. **Covariance-only path** (§4.7) + test 12.
8. **Python bindings** (§4.10) + test 14.
9. **Validation tests** (test 6).
10. **End-to-end Rosetta verification** (§7).

---

## 9. Assumptions and out-of-scope

**In scope (v1):**

- Translational 6-state multi-arc continuity for one or more bodies via
  `ArcWiseInitialTranslationalStateParameter`.
- Both shared-boundary (Rosetta OCM) and mid-gap (Orbit14) connection
  epochs.
- Per-boundary `C` and per-boundary μ (generalization of the paper's
  single-μ formulation).
- `m_d`-normalized cost matching paper Eq. (28).

**Out of scope (v1):**

- Rotational and mass-state continuity.
- Hybrid-arc dynamics.
- Non-consecutive arc pairs (validator rejects `right != left + 1`).
- Automatic arc-window extension into Orbit14 gaps — users explicitly
  extend each arc's propagation interval to cover its connection epoch.

**Conventions:**

- Cost: `Q_d = (1/m_d) · (1/μ) · Σ d^T C d` — matches Cicalò et al. 2021
  Eq. (28). Larger μ weakens.
- Residual sign: continuity residual is `−d` (verified at
  `orbitDeterminationManagerHelpers.h:103-104`); hence
  `g_constraint += −D^T W_d d` and `H_constraint += D^T W_d D`.
- Connection epoch placement is user-supplied; no midpoint or OCM
  convention hard-coded.

---

## 10. Critical files reference (quick index)

- `src/tudat/math/basic/leastSquaresEstimation.cpp:66-162` — LSQ solver + hard-constraint augmentation.
- `include/tudat/math/basic/leastSquaresEstimation.h:129-172` — declarations.
- `include/tudat/simulation/estimation_setup/orbitDeterminationManagerEstimationImplementation.h:100-340` — iteration loop.
- `include/tudat/simulation/estimation_setup/orbitDeterminationManager.h:213, 244, 283` — member accessors to downcast.
- `include/tudat/simulation/estimation_setup/orbitDeterminationManagerCovarianceImplementation.h:82-88` — covariance assembly.
- `include/tudat/astro/propagators/stateTransitionMatrixInterface.h:247-931` — multi-arc STM interface (extend here).
- `include/tudat/astro/orbit_determination/podInputOutputTypes.h:35-1227` — input/output storage (extend here).
- `include/tudat/astro/orbit_determination/estimatable_parameters/initialTranslationalState.h:138-349` — arc-wise translational parameter.
- `include/tudat/astro/orbit_determination/estimatable_parameters/estimatableParameterSet.h:91-122` — global parameter ordering; use `parameterIndices_` to locate block offsets.
- `include/tudat/simulation/propagation_setup/multiArcDynamicsSimulator.h:48-107, 702-710` — per-arc results + 8th-order Lagrange handoff helper.
- `include/tudat/simulation/propagation_setup/propagationResults.h` — per-arc result accessors.
- `src/tudatpy/estimation/estimation_analysis/expose_estimation_analysis.cpp:337-948` — Python bindings to extend.
- `src/tudatpy/estimation/observations_setup/viability/expose_viability.cpp:77-215` — settings/factory pattern to mirror.
- `tests/test_tudat/src/astro/orbit_determination/unitTestMultiArcStateEstimation.cpp` — multi-arc test pattern to follow.

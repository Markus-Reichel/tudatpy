import numpy as np
import pytest

from tudatpy.dynamics import environment_setup, parameters_setup, propagation_setup
from tudatpy.estimation import estimation_analysis
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation.observations_setup import (
    observations_simulation_settings,
    observations_wrapper,
)
from tudatpy.interface import spice


def _create_estimation():
    """Create an Earth orbiter with degree and order 2 gravity, observed through its position, and a function that
    estimates its initial state from a perturbed value with a given solver."""

    spice.load_standard_kernels()
    bodies = environment_setup.create_system_of_bodies(
        environment_setup.get_default_body_settings(["Earth"], "Earth", "J2000")
    )
    bodies.create_empty_body("Delfi")
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        {"Delfi": {"Earth": [propagation_setup.acceleration.spherical_harmonic_gravity(2, 2)]}},
        ["Delfi"],
        ["Earth"],
    )
    initial_state = np.array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    propagator_settings = propagation_setup.propagator.translational(
        ["Earth"],
        acceleration_models,
        ["Delfi"],
        initial_state,
        0.0,
        propagation_setup.integrator.runge_kutta_fixed_step(
            100.0, propagation_setup.integrator.CoefficientSets.rkf_78
        ),
        propagation_setup.propagator.time_termination(3600.0),
    )
    parameter_set = parameters_setup.create_parameter_set(
        parameters_setup.initial_states(propagator_settings, bodies), bodies
    )

    link_ends = links.link_definition({links.observed_body: links.body_origin_link_end_id("Delfi")})
    estimator = estimation_analysis.Estimator(
        bodies, parameter_set, [model_settings.cartesian_position(link_ends)], propagator_settings
    )
    observations = observations_wrapper.simulate_observations(
        [
            observations_simulation_settings.tabulated_simulation_settings(
                model_settings.ObservableType.position_observable_type,
                link_ends,
                np.arange(0.0, 3600.0, 300.0),
                links.observed_body,
            )
        ],
        estimator.observation_simulators,
        bodies,
    )

    def estimate(solver_settings, sparse_design_matrix=False):
        parameter_set.parameter_vector = initial_state + np.array(
            [100.0, 100.0, 100.0, 0.1, 0.1, 0.1]
        )
        estimation_input = estimation_analysis.EstimationInput(
            observations, convergence_checker=estimation_analysis.estimation_convergence_checker(3)
        )
        estimation_input.define_estimation_settings(print_output_to_terminal=False)
        estimation_input.normal_equations_solver_settings = solver_settings
        estimation_input.sparse_design_matrix = sparse_design_matrix
        return estimation_input, estimator.perform_estimation(estimation_input)

    return estimate


@pytest.mark.parametrize(
    "solver_name, sparse_design_matrix",
    [
        ("cholesky_solver", False),
        ("sparse_qr_solver", True),
        ("sparse_cholesky_solver", True),
        ("bdc_svd_solver", True),
        ("cholesky_solver", True),
    ],
)
def test_solver_matches_default(solver_name, sparse_design_matrix):
    """Check that a solver, with a dense or sparse design matrix, gives the same estimate and covariance as the default
    solver with a dense design matrix."""

    estimate = _create_estimation()
    _, reference = estimate(None)
    _, output = estimate(getattr(estimation_analysis, solver_name)(), sparse_design_matrix)

    # Compare the estimates relative to their formal errors, since some state components are zero
    estimate_difference = (
        output.final_parameters - reference.final_parameters
    ) / reference.formal_errors
    assert np.max(np.abs(estimate_difference)) < 1.0e-6
    assert np.allclose(output.formal_errors, reference.formal_errors, rtol=1.0e-8, atol=0.0)
    assert np.allclose(output.correlations, reference.correlations, rtol=0.0, atol=1.0e-8)


def test_cost_history():
    """Check that the reported cost of the best iteration is half the weighted sum of squared final residuals."""

    estimation_input, output = _create_estimation()(None)
    weights = estimation_input.weight_matrix_diagonal
    expected_cost = 0.5 * np.sum(weights * output.final_residuals**2)
    best_cost = output.cost_history[output.best_iteration]
    assert np.isclose(best_cost.observation_cost, expected_cost, rtol=1.0e-12, atol=0.0)
    assert best_cost.total_cost == best_cost.observation_cost
    assert min(cost.total_cost for cost in output.cost_history) == best_cost.total_cost

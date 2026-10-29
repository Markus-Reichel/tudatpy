import numpy as np

from tudatpy.dynamics import environment_setup, parameters_setup, propagation_setup
from tudatpy.estimation import estimation_analysis
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation.observations_setup import (
    observations_simulation_settings,
    observations_wrapper,
)
from tudatpy.interface import spice


def test_cholesky_solver_matches_default():
    """Check that an estimation with the Cholesky solver gives the same estimate and covariance as the default solver."""

    # Earth orbiter with degree and order 2 gravity, observed through its position
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

    # Estimate from the same perturbed initial state with the given solver
    def estimate(solver_settings):
        parameter_set.parameter_vector = initial_state + np.array(
            [100.0, 100.0, 100.0, 0.1, 0.1, 0.1]
        )
        estimation_input = estimation_analysis.EstimationInput(
            observations, convergence_checker=estimation_analysis.estimation_convergence_checker(3)
        )
        estimation_input.define_estimation_settings(print_output_to_terminal=False)
        estimation_input.normal_equations_solver_settings = solver_settings
        return estimator.perform_estimation(estimation_input)

    reference = estimate(None)
    cholesky = estimate(estimation_analysis.cholesky_solver())
    # Compare the estimates relative to their formal errors, since some state components are zero
    estimate_difference = (
        cholesky.final_parameters - reference.final_parameters
    ) / reference.formal_errors
    assert np.max(np.abs(estimate_difference)) < 1.0e-6
    assert np.allclose(cholesky.formal_errors, reference.formal_errors, rtol=1.0e-8, atol=0.0)
    assert np.allclose(cholesky.correlations, reference.correlations, rtol=0.0, atol=1.0e-8)

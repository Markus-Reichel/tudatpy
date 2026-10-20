import numpy as np
import pytest

from tudatpy.astro import element_conversion
from tudatpy.dynamics import environment_setup, parameters_setup, propagation_setup
from tudatpy.estimation import estimation_analysis, observable_models_setup, observations_setup
from tudatpy.estimation import observations as observation_data
from tudatpy.estimation.observations import ObservationSelectionCondition as Condition


def test_orbit_sigma_recovery():
    """Range and position noise levels, started 10x off, are recovered by the estimator."""

    # Vehicle around a fixed Earth, observed with range from two stations and with its position.
    gravitational_parameter = 3.986004418e14
    body_settings = environment_setup.BodyListSettings("SSB", "J2000")
    body_settings.add_empty_settings("Earth")
    earth = body_settings.get("Earth")
    earth.ephemeris_settings = environment_setup.ephemeris.constant(np.zeros(6), "SSB", "J2000")
    earth.gravity_field_settings = environment_setup.gravity_field.central(gravitational_parameter)
    earth.shape_settings = environment_setup.shape.spherical(6.371e6)
    earth.rotation_model_settings = environment_setup.rotation_model.constant_rotation_model(
        "J2000", "IAU_Earth", np.eye(3)
    )
    body_settings.add_empty_settings("Vehicle")
    bodies = environment_setup.create_system_of_bodies(body_settings)
    for i in range(2):
        environment_setup.add_ground_station(
            bodies.get("Earth"),
            environment_setup.ground_station.basic_station(
                f"Station{i}",
                [0.0, 0.3 * i, 1.0 + 2.0 * i],
                element_conversion.PositionElementTypes.geodetic_position_type,
            ),
        )

    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        {"Vehicle": {"Earth": [propagation_setup.acceleration.point_mass_gravity()]}},
        ["Vehicle"],
        ["Earth"],
    )
    end_time = 4.0 * 3600.0
    propagator_settings = propagation_setup.propagator.translational(
        ["Earth"],
        acceleration_models,
        ["Vehicle"],
        element_conversion.keplerian_to_cartesian(
            np.array([7200.0e3, 0.05, 1.2, 0.5, 0.3, 0.0]), gravitational_parameter
        ),
        0.0,
        propagation_setup.integrator.runge_kutta_fixed_step(
            10.0, propagation_setup.integrator.CoefficientSets.rk_4
        ),
        propagation_setup.propagator.time_termination(end_time),
    )
    parameter_set = parameters_setup.create_parameter_set(
        parameters_setup.initial_states(propagator_settings, bodies), bodies
    )

    links = observable_models_setup.links
    model_settings = observable_models_setup.model_settings
    range_links = [
        links.LinkDefinition(
            {
                links.transmitter: links.body_reference_point_link_end_id("Earth", f"Station{i}"),
                links.receiver: links.body_origin_link_end_id("Vehicle"),
            }
        )
        for i in range(2)
    ]
    position_link = links.LinkDefinition(
        {links.observed_body: links.body_origin_link_end_id("Vehicle")}
    )
    estimator = estimation_analysis.Estimator(
        bodies,
        parameter_set,
        [model_settings.one_way_range(link) for link in range_links]
        + [model_settings.cartesian_position(position_link)],
        propagator_settings,
    )

    # Noise of 1 m on range and 50 m on each position component.
    rng = np.random.default_rng(1234)
    simulation_settings = observations_setup.observations_simulation_settings
    settings_list = [
        simulation_settings.tabulated_simulation_settings(
            model_settings.one_way_range_type,
            link,
            np.arange(60.0, end_time - 60.0, 30.0),
            links.receiver,
            noise_function=lambda time: rng.normal(0.0, 1.0, 1),
        )
        for link in range_links
    ]
    settings_list.append(
        simulation_settings.tabulated_simulation_settings(
            model_settings.position_observable_type,
            position_link,
            np.arange(60.0, end_time - 60.0, 600.0),
            links.observed_body,
            noise_function=lambda time: rng.normal(0.0, 50.0, 3),
        )
    )
    observations = observation_data.simulate_observation_dataset(
        settings_list, estimator.observation_simulators, bodies
    )

    groups = [
        Condition.observable_type(model_settings.one_way_range_type),
        Condition.observable_type(model_settings.position_observable_type),
    ]
    initial_sigmas = np.array([10.0, 5.0])
    for group, sigma in zip(groups, initial_sigmas):
        observations.set_constant_single_observation_scalar_weight(group, 1.0 / sigma**2)

    perturbed_parameters = parameter_set.parameter_vector
    perturbed_parameters[:3] += 100.0
    perturbed_parameters[3:] += 0.1
    parameter_set.parameter_vector = perturbed_parameters

    estimation_input = estimation_analysis.EstimationInput(
        observations, convergence_checker=estimation_analysis.estimation_convergence_checker(6)
    )
    estimation_input.define_estimation_settings(print_output_to_terminal=False)
    output = estimator.perform_estimation_with_variance_components(
        estimation_input,
        estimation_analysis.variance_component_settings(groups, maximum_iterations=15),
    )

    assert output.is_converged
    estimated_sigmas = initial_sigmas * np.sqrt(output.total_variance_factor_history[-1])
    assert estimated_sigmas[0] == pytest.approx(1.0, rel=0.08)
    assert estimated_sigmas[1] == pytest.approx(50.0, rel=0.3)

    # Applied to the final estimation, the standalone function finds no further change.
    estimate = estimation_analysis.compute_variance_components(
        output.estimation_output, observations, groups
    )
    factors = [result.variance_factor for result in estimate.group_results]
    assert np.allclose(factors, 1.0, atol=0.01)

"""Regression tests for aerodynamic dependent variables in Estimator construction.

Mirrors the symptom from the Rosetta IFMS Doppler estimation pipeline, where
``tudatpy.estimation.estimation_analysis.Estimator(...)`` failed with:

    RuntimeError: Error: Lagrange interpolator cannot identify zero entry.

The error comes from ``lagrangeInterpolator.h:131`` and is a downstream NaN
detector. The Rosetta reproducer isolated one NaN source to propagation output:
``local_wind_velocity`` saved in a non-corotating aerodynamic frame. The
corotating wind vector and the rotation matrix were both finite, but returning
the unevaluated Eigen product from the dependent-variable lambda produced NaN
entries in the saved dependent-variable history.

This test wires the same coma/drag chain the Rosetta script uses:
``coma_model_file_processor`` → ``coma_model_from_stokes_data`` →
``set_density_correction_harmonic_degree`` → ``custom_parameter`` +
``custom_analytical_partial`` pointing at
``get_density_correction_acceleration_partial``. It also saves vertical-frame
``local_wind_velocity`` so estimator construction exercises the dependent
variable interpolator that used to fail.
"""
from pathlib import Path

import numpy as np
import pytest

from tudatpy.astro.time_representation import DateTime
from tudatpy.data import coma_model as coma_data
from tudatpy.dynamics import (
    environment_setup,
    parameters_setup,
    propagation_setup,
)
from tudatpy.dynamics.environment_setup import aerodynamic_coefficients as aero_coef_setup
from tudatpy.estimation import estimation_analysis
from tudatpy.estimation.observations_setup import observations_wrapper
from tudatpy.interface import spice


SPACECRAFT_NAME = "ProbeForBugRegression"
COMET_BODY = "ComaTestComet"

_COMA_DATA = Path(__file__).resolve().parents[1] / "data" / "coma"
_DENSITY_FILE = _COMA_DATA / "density" / "polynomial" / "input_poly_coef_test_file.txt"
_WIND_X_FILE = _COMA_DATA / "wind" / "polynomial" / "input_poly_wind_x.txt"
_WIND_Y_FILE = _COMA_DATA / "wind" / "polynomial" / "input_poly_wind_y.txt"
_WIND_Z_FILE = _COMA_DATA / "wind" / "polynomial" / "input_poly_wind_z.txt"


def _build_bodies_with_coma(initial_time, final_time):
    """Build a minimal Sun + comet + spacecraft system, mirroring Rosetta scripts.

    Uses ``coma_model_file_processor`` to load the test polynomial coefficient
    file, then wires the resulting Stokes dataset onto the comet's atmosphere
    via ``coma_model_from_stokes_data``. A constant Python wind callback is
    attached so the bug's full ingredient list (ComaModel + wind + drag + Cd)
    is reproduced.
    """
    density_proc = coma_data.coma_model_file_processor([str(_DENSITY_FILE)])
    density_stokes = density_proc.create_coma_stokes_dataset(
        [4000.0, 10000.0], [0.0, 30.0], 10, 10
    )

    body_settings = environment_setup.get_default_body_settings(["Sun"], "SSB", "ECLIPJ2000")

    # Comet at the Sun-system origin (constant ephemeris) with identity rotation
    # — same solar-longitude convention as the Rosetta scripts (Sun on +X,
    # identity comet rotation).
    body_settings.add_empty_settings(COMET_BODY)
    body_settings.get(COMET_BODY).ephemeris_settings = environment_setup.ephemeris.constant(
        np.zeros(6), "SSB", "ECLIPJ2000"
    )
    body_settings.get(COMET_BODY).rotation_model_settings = environment_setup.rotation_model.simple(
        "ECLIPJ2000", "ComaTestComet_Fixed", np.eye(3), initial_time, 0.0
    )
    body_settings.get(COMET_BODY).gravity_field_settings = environment_setup.gravity_field.central(670.0)
    body_settings.get(COMET_BODY).shape_settings = environment_setup.shape.spherical(2000.0)
    body_settings.get(COMET_BODY).atmosphere_settings = environment_setup.atmosphere.coma_model_from_stokes_data(
        stokes_data=density_stokes,
        molecular_weight=18.01528e-3,
        max_degree=10,
        max_order=10,
        is_log2=True,
    )
    # ComaWindModel via SH wind datasets — same code path as failing Rosetta Cases 6-7.
    wind_proc = coma_data.coma_wind_file_processor(
        [str(_WIND_X_FILE)], [str(_WIND_Y_FILE)], [str(_WIND_Z_FILE)]
    )
    wind_datasets = wind_proc.create_coma_stokes_dataset(
        [4000.0, 10000.0], [0.0, 30.0], 10, 10
    )
    body_settings.get(COMET_BODY).atmosphere_settings.wind_settings = (
        environment_setup.atmosphere.coma_wind_model(
            dataset_collection=wind_datasets,
            requested_max_degree=10,
            requested_max_order=10,
            associated_reference_frame=environment_setup.aerodynamic_coefficients.AerodynamicsReferenceFrames.vertical_frame,
        )
    )

    body_settings.add_empty_settings(SPACECRAFT_NAME)
    body_settings.get(SPACECRAFT_NAME).constant_mass = 100.0

    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Spacecraft aero coefficients (constant Cd) — matches failing Rosetta cases.
    aero_settings = aero_coef_setup.constant(reference_area=10.0, constant_force_coefficient=[2.0, 0.0, 0.0])
    environment_setup.add_aerodynamic_coefficient_interface(bodies, SPACECRAFT_NAME, aero_settings)

    # Cannonball SRP — present in all failing Rosetta cases.
    target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        reference_area=4.0,
        radiation_pressure_coefficient=1.2,
    )
    environment_setup.add_radiation_pressure_target_model(bodies, SPACECRAFT_NAME, target_settings)

    return bodies


def _build_propagator_settings(bodies, initial_time, final_time):
    bodies_to_propagate = [SPACECRAFT_NAME]
    central_bodies = [COMET_BODY]

    acceleration_settings = {
        SPACECRAFT_NAME: {
            COMET_BODY: [
                propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.aerodynamic(),
            ],
            "Sun": [
                propagation_setup.acceleration.radiation_pressure(),
            ],
        }
    }
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    # Initial state ~5 km from the comet — inside the valid coma-dataset range (4-10 km).
    # Tangential velocity small enough that wind dominates airspeed (like Rosetta).
    initial_state = np.array([5000.0, 0.0, 0.0,
                              0.0, 0.4, 0.0])

    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        time_step=30.0,
        coefficient_set=propagation_setup.integrator.rkf_78,
    )
    termination = propagation_setup.propagator.time_termination(final_time)

    output_variables = [
        propagation_setup.dependent_variable.local_wind_velocity(
            SPACECRAFT_NAME,
            COMET_BODY,
            target_frame=environment_setup.aerodynamic_coefficients.AerodynamicsReferenceFrames.vertical_frame,
        ),
        propagation_setup.dependent_variable.number_density(SPACECRAFT_NAME, COMET_BODY),
        propagation_setup.dependent_variable.density(SPACECRAFT_NAME, COMET_BODY),
        propagation_setup.dependent_variable.dynamic_pressure(SPACECRAFT_NAME, COMET_BODY),
    ]

    return propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        initial_time,
        integrator_settings,
        termination,
        propagator=propagation_setup.propagator.cowell,
        output_variables=output_variables,
    )


@pytest.mark.skipif(not _DENSITY_FILE.exists(), reason="coma test data not present")
def test_estimator_constructs_with_coma_density_correction():
    """Estimator with coma drag, density correction, and wind output must construct.

    This catches NaNs in the dependent-variable history before they reach the
    Lagrange interpolator used by the estimator's dependent-variable interface.
    """
    spice.load_standard_kernels()

    # Initial epoch must fall inside the coma-test-file time validity window
    # (2015-07-21 → 2015-08-21, per the header of input_poly_coef_test_file.txt).
    initial_time = DateTime(2015, 7, 22, 5, 0, 0).to_epoch()
    final_time = initial_time + 1800.0  # 30 minutes — enough integration steps to expose the bug

    bodies = _build_bodies_with_coma(initial_time, final_time)
    propagator_settings = _build_propagator_settings(bodies, initial_time, final_time)

    parameter_settings = parameters_setup.initial_states(propagator_settings, bodies)
    parameter_settings.append(parameters_setup.radiation_pressure_coefficient(SPACECRAFT_NAME))
    parameter_settings.append(parameters_setup.constant_drag_coefficient(SPACECRAFT_NAME))

    # Wire density correction estimation — the trigger of the bug (failing Cases 6-7).
    coma_model = bodies.get(COMET_BODY).atmosphere_model
    coma_model.set_density_correction_harmonic_degree(0)
    coma_model.set_density_correction_parameters(
        np.zeros(coma_model.get_density_correction_parameter_size())
    )
    density_parameter = parameters_setup.custom_parameter(
        "global_coma_log_density_correction",
        coma_model.get_density_correction_parameter_size(),
        coma_model.get_density_correction_parameters,
        coma_model.set_density_correction_parameters,
    )
    density_parameter.add_custom_partial_settings(
        parameters_setup.custom_analytical_partial(
            coma_model.get_density_correction_acceleration_partial,
            SPACECRAFT_NAME,
            COMET_BODY,
            propagation_setup.acceleration.AvailableAcceleration.aerodynamic_type,
        )
    )
    parameter_settings.append(density_parameter)

    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )

    observation_model_settings, _ = observations_wrapper.create_pseudo_observations_and_models(
        bodies,
        [SPACECRAFT_NAME],
        [COMET_BODY],
        initial_time,
        final_time,
        60.0,
    )

    # The actual bug surface: this raises if the propagated dependent-variable
    # history contains NaNs and the Lagrange interpolator rejects it.
    try:
        estimator = estimation_analysis.Estimator(
            bodies,
            parameters_to_estimate,
            observation_model_settings,
            propagator_settings,
        )
    except RuntimeError as exc:
        pytest.fail(
            "Estimator construction raised RuntimeError — the variational-equations "
            "NaN bug is still present. Reported error: " + str(exc)
        )

    assert estimator is not None

    # Probe the state-transition + sensitivity interpolator at a few epochs. If the
    # upstream catch block silently swallowed the Lagrange-interpolator exception
    # (singleArcVariationalEquationsSolver.h:336-347), the interpolator entries
    # are null and querying produces NaN.
    state_transition = estimator.state_transition_interface
    assert state_transition is not None

    for t in (initial_time + 120.0, initial_time + 600.0, initial_time + 1500.0):
        matrix = state_transition.full_state_transition_sensitivity_at_epoch(t)
        assert np.all(np.isfinite(matrix)), (
            f"Combined state-transition+sensitivity matrix at t={t} has non-finite entries: {matrix}"
        )

/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <cmath>
#include <iostream>
#include <map>
#include <memory>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>

#include "tudat/astro/aerodynamics/comaModel.h"
#include "tudat/astro/aerodynamics/comaWindModel.h"
#include "tudat/astro/aerodynamics/exponentialAtmosphere.h"
#include "tudat/io/basicInputOutput.h"
#include "tudat/io/comaModelInputOutput.h"
#include "tudat/math/basic/mathematicalConstants.h"
#include "tudat/simulation/environment_setup/createAtmosphereModel.h"
#include "tudat/simulation/environment_setup/createBodiesFactory.h"
#include "tudat/simulation/environment_setup/defaultBodies.h"
#include "tudat/simulation/estimation_setup/createEstimatableParametersFactory.h"
#include "tudat/simulation/estimation_setup/orbitDeterminationManager.h"
#include "tudat/simulation/estimation_setup/simulatePseudoObservations.h"
#include "tudat/simulation/estimation_setup/simulateObservations.h"
#include "tudat/simulation/propagation_setup/singleArcDynamicsSimulator.h"

namespace tudat
{
namespace unit_tests
{

using namespace tudat::aerodynamics;
using namespace tudat::basic_astrodynamics;
using namespace tudat::estimatable_parameters;
using namespace tudat::observation_models;
using namespace tudat::orbit_determination;
using namespace tudat::propagators;
using namespace tudat::simulation_setup;

BOOST_AUTO_TEST_SUITE( test_estimation_coma_model_drag )

// Closest mirror of the failing Rosetta estimation cases (Cases 2-6) at minimal scale.
// All five failing cases share: ComaModel atmosphere + a wind model + drag + Cd estimated.
// The variational equations propagation inside OrbitDeterminationManager construction produces
// NaN entries that downstream cause:
//
//     RuntimeError: Error: Lagrange interpolator cannot identify zero entry.
//
// from lagrangeInterpolator.h:131. This test reproduces the configuration end-to-end using the
// same coma test datasets that unitTestComaWindModel.cpp uses (tests/data/coma/...).
BOOST_AUTO_TEST_CASE( test_VariationalEquationsWithComaAndWind )
{
    spice_interface::loadStandardSpiceKernels( );

    // Test data setup, matching unitTestComaWindModel.cpp:649-666.
    const boost::filesystem::path comaDataDir =
            boost::filesystem::path( tudat::paths::getTudatTestDataPath( ) ) / "coma";
    const boost::filesystem::path windDataDir = comaDataDir / "wind";

    const std::vector< std::string > xWindFiles = { ( windDataDir / "polynomial" / "input_poly_wind_x.txt" ).string( ) };
    const std::vector< std::string > yWindFiles = { ( windDataDir / "polynomial" / "input_poly_wind_y.txt" ).string( ) };
    const std::vector< std::string > zWindFiles = { ( windDataDir / "polynomial" / "input_poly_wind_z.txt" ).string( ) };
    const std::vector< std::string > densityFiles = {
            ( comaDataDir / "density" / "polynomial" / "input_poly_coef_test_file.txt" ).string( )
    };

    if( !boost::filesystem::exists( xWindFiles[ 0 ] ) || !boost::filesystem::exists( densityFiles[ 0 ] ) )
    {
        BOOST_TEST_MESSAGE( "Coma test data files missing — skipping test." );
        return;
    }

    // Datasets for ComaModel + ComaWindModel — radii in the validity range (4-10 km) and two
    // solar-longitude samples.
    const std::vector< double > radii_m = { 4000.0, 10000.0 };
    const std::vector< double > solLongitudes_deg = { 0.0, 30.0 };
    const int maxDegree = 10;
    const int maxOrder = 10;

    ComaModelFileProcessor densityProcessor( densityFiles );
    const ComaStokesDataset densityStokes = densityProcessor.createSHDataset( radii_m, solLongitudes_deg, maxDegree, maxOrder );

    ComaWindModelFileProcessor windProcessor( xWindFiles, yWindFiles, zWindFiles );
    const ComaWindDatasetCollection windDatasets =
            windProcessor.createSHDatasets( radii_m, solLongitudes_deg, maxDegree, maxOrder );

    // Solar longitude convention from the user's project memory: Sun at +X in inertial frame,
    // comet with identity rotation. This is the convention used in the Rosetta scripts.
    auto sunStateFunction = []( ) -> Eigen::Vector6d {
        Eigen::Vector6d state = Eigen::Vector6d::Zero( );
        state.segment( 0, 3 ) = Eigen::Vector3d( 1.5e11, 0.0, 0.0 );
        return state;
    };
    auto cometStateFunction = []( ) -> Eigen::Vector6d { return Eigen::Vector6d::Zero( ); };
    auto cometRotationFunction = []( ) -> Eigen::Matrix3d { return Eigen::Matrix3d::Identity( ); };

    const double molecularWeight = 18.01528e-3;  // kg/mol (water)

    auto comaModel = std::make_shared< ComaModel >( densityStokes,
                                                    molecularWeight,
                                                    sunStateFunction,
                                                    cometStateFunction,
                                                    cometRotationFunction,
                                                    maxDegree,
                                                    maxOrder );

    auto comaWindModel = std::make_shared< ComaWindModel >( windDatasets.getXStokesDataset( ),
                                                            windDatasets.getYStokesDataset( ),
                                                            windDatasets.getZStokesDataset( ),
                                                            comaModel,
                                                            sunStateFunction,
                                                            cometStateFunction,
                                                            cometRotationFunction,
                                                            maxDegree,
                                                            maxOrder,
                                                            reference_frames::vertical_frame,
                                                            true,  // includeCorotation
                                                            true );  // useRadius
    comaModel->setWindModel( comaWindModel );

    // Body setup: Sun (constant ephemeris at +X), Comet (constant ephemeris at origin,
    // identity rotation, tiny gravity, ComaModel atmosphere), Spacecraft (orbit at ~5 km).
    // Initial epoch must fall inside the test coma data file's time validity window
    // (2015-07-21 00:00:00 -> 2015-08-21 in J2000 seconds, per the header of
    // input_poly_coef_test_file.txt).
    const double initialTime = 490795200.0;  // 2015-07-22 00:00:00 J2000
    const double finalTime = initialTime + 7200.0;  // 2 hours — enough steps to expose NaN drift

    BodyListSettings bodySettings;
    bodySettings.addSettings( "Sun" );
    bodySettings.at( "Sun" )->ephemerisSettings = std::make_shared< ConstantEphemerisSettings >(
            ( Eigen::Vector6d( ) << 1.5e11, 0.0, 0.0, 0.0, 0.0, 0.0 ).finished( ) );
    bodySettings.at( "Sun" )->gravityFieldSettings = centralGravitySettings( 1.32712440018e20 );

    bodySettings.addSettings( "Comet" );
    bodySettings.at( "Comet" )->ephemerisSettings =
            std::make_shared< ConstantEphemerisSettings >( Eigen::Vector6d::Zero( ) );
    bodySettings.at( "Comet" )->gravityFieldSettings = centralGravitySettings( 670.0 );
    bodySettings.at( "Comet" )->shapeModelSettings = std::make_shared< SphericalBodyShapeSettings >( 2000.0 );
    bodySettings.at( "Comet" )->rotationModelSettings =
            std::make_shared< SimpleRotationModelSettings >( "ECLIPJ2000", "ECLIPJ2000_FIXED", Eigen::Quaterniond::Identity( ), initialTime, 0.0 );

    bodySettings.addSettings( "Spacecraft" );
    std::map< double, Eigen::Vector6d > emptyMap;
    bodySettings.at( "Spacecraft" )->ephemerisSettings =
            std::make_shared< TabulatedEphemerisSettings >( emptyMap, "Comet", "ECLIPJ2000" );

    SystemOfBodies bodies = createSystemOfBodies( bodySettings );

    // Wire the directly-constructed ComaModel into the Comet body.
    bodies.at( "Comet" )->setAtmosphereModel( comaModel );

    // Spacecraft physical properties.
    bodies.at( "Spacecraft" )->setConstantBodyMass( 100.0 );
    const Eigen::Vector3d forceCoefficients( 2.0, 0.0, 0.0 );
    std::shared_ptr< AerodynamicCoefficientSettings > aerodynamicCoefficientSettings =
            std::make_shared< ConstantAerodynamicCoefficientSettings >(
                    10.0, forceCoefficients, aerodynamics::negative_aerodynamic_frame_coefficients );
    bodies.at( "Spacecraft" )
            ->setAerodynamicCoefficientInterface(
                    createAerodynamicCoefficientInterface( aerodynamicCoefficientSettings, "Spacecraft", bodies ) );

    // Initial state: circular-ish orbit at 5 km radius. Gravity is tiny so velocity is small;
    // wind dominates airspeed (just like the Rosetta case).
    Eigen::Vector6d initialState;
    initialState << 5000.0, 0.0, 0.0,
                    0.0, 0.4, 0.0;  // ~0.4 m/s tangential

    // Acceleration setup.
    SelectedAccelerationMap accelerationMap;
    std::map< std::string, std::vector< std::shared_ptr< AccelerationSettings > > > accelerationsOfSpacecraft;
    accelerationsOfSpacecraft[ "Comet" ].push_back( std::make_shared< AccelerationSettings >( aerodynamic ) );
    accelerationsOfSpacecraft[ "Comet" ].push_back( std::make_shared< AccelerationSettings >( point_mass_gravity ) );
    accelerationMap[ "Spacecraft" ] = accelerationsOfSpacecraft;

    std::vector< std::string > bodiesToEstimate = { "Spacecraft" };
    std::vector< std::string > centralBodies = { "Comet" };
    AccelerationMap accelerationModelMap = createAccelerationModelsMap( bodies, accelerationMap, bodiesToEstimate, centralBodies );

    std::shared_ptr< TranslationalStatePropagatorSettings< double > > propagatorSettings =
            std::make_shared< TranslationalStatePropagatorSettings< double > >(
                    centralBodies,
                    accelerationModelMap,
                    bodiesToEstimate,
                    initialState,
                    initialTime,
                    numerical_integrators::rungeKuttaFixedStepSettings( 30.0, numerical_integrators::rungeKuttaFehlberg78 ),
                    std::make_shared< PropagationTimeTerminationSettings >( finalTime ),
                    cowell );

    // Parameters: initial state + Cd + global coma log-density correction.
    // Density correction estimation is the specific ingredient that flips the failing Rosetta
    // estimation cases from passing (Cases 1-5) to failing (Cases 6-7). The custom partial is
    // routed via ComaModel::getDensityCorrectionAccelerationPartial.
    std::vector< std::shared_ptr< EstimatableParameterSettings > > parameterNames =
            getInitialStateParameterSettings< double, double >( propagatorSettings, bodies );
    parameterNames.push_back( estimatable_parameters::constantDragCoefficient( "Spacecraft" ) );

    comaModel->setDensityCorrectionHarmonicDegree( 0 );
    comaModel->setDensityCorrectionParameterVector(
            Eigen::VectorXd::Zero( comaModel->getDensityCorrectionParameterSize( ) ) );

    auto densityParameterSettings = estimatable_parameters::customParameterSettings(
            "global_coma_log_density_correction",
            comaModel->getDensityCorrectionParameterSize( ),
            [ comaModel ]( ) { return comaModel->getDensityCorrectionParameterVector( ); },
            [ comaModel ]( const Eigen::VectorXd& v ) { comaModel->setDensityCorrectionParameterVector( v ); } );
    densityParameterSettings->customPartialSettings_.push_back(
            estimatable_parameters::analyticalAccelerationPartialSettings(
                    [ comaModel ]( const double t, const Eigen::Vector3d& accel ) {
                        return comaModel->getDensityCorrectionAccelerationPartial( t, accel );
                    },
                    "Spacecraft",
                    "Comet",
                    basic_astrodynamics::aerodynamic ) );
    parameterNames.push_back( densityParameterSettings );

    std::shared_ptr< estimatable_parameters::EstimatableParameterSet< double > > parametersToEstimate =
            createParametersToEstimate< double, double >( parameterNames, bodies, propagatorSettings );

    std::pair< std::vector< std::shared_ptr< observation_models::ObservationModelSettings > >,
               std::shared_ptr< observation_models::ObservationCollection< double > > >
            observationCollectionAndModelSettings =
                    simulatePseudoObservations( bodies, bodiesToEstimate, centralBodies, initialTime, finalTime, 60.0 );
    const std::vector< std::shared_ptr< observation_models::ObservationModelSettings > > observationModelSettingsList =
            observationCollectionAndModelSettings.first;

    // === The actual bug surface ===
    std::shared_ptr< OrbitDeterminationManager< double > > orbitDeterminationManager;
    BOOST_REQUIRE_NO_THROW(
            orbitDeterminationManager = std::make_shared< OrbitDeterminationManager< double > >(
                    bodies, parametersToEstimate, observationModelSettingsList, propagatorSettings ) );
    BOOST_REQUIRE( orbitDeterminationManager != nullptr );

    auto stateTransitionInterface = orbitDeterminationManager->getStateTransitionAndSensitivityMatrixInterface( );
    BOOST_REQUIRE( stateTransitionInterface != nullptr );

    // Probe the state-transition + sensitivity interface at a few epochs — catches the case
    // where the upstream catch block silently swallowed the Lagrange exception and left null
    // interpolators that produce NaN when queried.
    const std::vector< double > probeTimes = { initialTime + 60.0, initialTime + 300.0, initialTime + 540.0 };
    for( const double t: probeTimes )
    {
        Eigen::MatrixXd combined;
        BOOST_REQUIRE_NO_THROW(
                combined = stateTransitionInterface->getCombinedStateTransitionAndSensitivityMatrix( t ) );
        for( int row = 0; row < combined.rows( ); ++row )
        {
            for( int col = 0; col < combined.cols( ); ++col )
            {
                BOOST_CHECK_MESSAGE( std::isfinite( combined( row, col ) ),
                                     "combined state transition + sensitivity matrix has non-finite entry at t="
                                             << t << " (" << row << "," << col << ") = " << combined( row, col ) );
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END( )

}  // namespace unit_tests
}  // namespace tudat

/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 *
 */

#define BOOST_TEST_MAIN

#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <boost/test/included/unit_test.hpp>

#include <Eigen/Cholesky>
#include <Eigen/SparseCore>

#include "tudat/astro/basic_astro/orbitalElementConversions.h"
#include "tudat/simulation/environment_setup/createBodiesFactory.h"
#include "tudat/simulation/environment_setup/createGroundStations.h"
#include "tudat/simulation/estimation_setup/createEstimatableParametersFactory.h"
#include "tudat/simulation/estimation_setup/orbitDeterminationManager.h"
#include "tudat/simulation/estimation_setup/simulateObservations.h"
#include "tudat/simulation/estimation_setup/varianceComponentEstimation.h"
#include "tudat/simulation/propagation_setup/accelerationSettings.h"
#include "tudat/simulation/propagation_setup/propagationSettings.h"

namespace tudat
{
namespace unit_tests
{

BOOST_AUTO_TEST_SUITE( test_variance_component_estimation )

using namespace tudat::estimatable_parameters;
using namespace tudat::numerical_integrators;
using namespace tudat::observation_models;
using namespace tudat::simulation_setup;
using namespace tudat::propagators;
using namespace tudat::basic_astrodynamics;

//! Linear problem y = A x + noise with 400 rows in group 0 and 100 rows in group 1. Group 0 noise can be correlated in pairs.
struct LinearProblem {
    Eigen::MatrixXd designMatrix;
    Eigen::VectorXd observations;
    std::vector< std::vector< int > > groupRowIndices;
    Eigen::Matrix2d pairCorrelation;

    LinearProblem( const double correlation, const std::vector< double >& sigmas ): designMatrix( 500, 6 ), observations( 500 )
    {
        std::mt19937 generator( 42 );
        std::normal_distribution< double > normal( 0.0, 1.0 );
        for( int i = 0; i < 500; i++ )
        {
            for( int j = 0; j < 6; j++ )
            {
                designMatrix( i, j ) = normal( generator );
            }
        }
        observations = designMatrix * Eigen::VectorXd::LinSpaced( 6, 1.0, 2.0 );

        pairCorrelation << 1.0, correlation, correlation, 1.0;
        const Eigen::Matrix2d choleskyFactor = pairCorrelation.llt( ).matrixL( );
        for( int pair = 0; pair < 200; pair++ )
        {
            observations.segment( 2 * pair, 2 ) +=
                    sigmas.at( 0 ) * choleskyFactor * Eigen::Vector2d( normal( generator ), normal( generator ) );
        }
        for( int row = 400; row < 500; row++ )
        {
            observations( row ) += sigmas.at( 1 ) * normal( generator );
        }
        groupRowIndices = { std::vector< int >( 400 ), std::vector< int >( 100 ) };
        std::iota( groupRowIndices.at( 0 ).begin( ), groupRowIndices.at( 0 ).end( ), 0 );
        std::iota( groupRowIndices.at( 1 ).begin( ), groupRowIndices.at( 1 ).end( ), 400 );
    }

    //! Weight matrix with the given group sigmas, using the pair correlation in group 0.
    Eigen::SparseMatrix< double > getWeights( const double sigma0, const double sigma1 ) const
    {
        const Eigen::Matrix2d pairWeights = pairCorrelation.inverse( ) / ( sigma0 * sigma0 );
        std::vector< Eigen::Triplet< double > > entries;
        for( int pair = 0; pair < 200; pair++ )
        {
            for( int i = 0; i < 2; i++ )
            {
                for( int j = 0; j < 2; j++ )
                {
                    entries.emplace_back( 2 * pair + i, 2 * pair + j, pairWeights( i, j ) );
                }
            }
        }
        for( int row = 400; row < 500; row++ )
        {
            entries.emplace_back( row, row, 1.0 / ( sigma1 * sigma1 ) );
        }
        Eigen::SparseMatrix< double > weights( 500, 500 );
        weights.setFromTriplets( entries.begin( ), entries.end( ) );
        return weights;
    }

    //! Weighted least-squares solution. Returns the post-fit residuals and sets the normal matrix.
    Eigen::VectorXd solve( const Eigen::SparseMatrix< double >& weights, const Eigen::MatrixXd& prior, Eigen::MatrixXd& normalMatrix ) const
    {
        normalMatrix = designMatrix.transpose( ) * ( weights * designMatrix ) + prior;
        return observations - designMatrix * normalMatrix.ldlt( ).solve( designMatrix.transpose( ) * ( weights * observations ) );
    }
};

//! Group redundancies and weighted square sums must match the hat matrix I - W A N^-1 A^T, with a prior. Scaling the
//! parameters must not change the variance factors.
BOOST_AUTO_TEST_CASE( testRedundancy )
{
    const LinearProblem problem( 0.0, { 0.5, 20.0 } );
    const Eigen::SparseMatrix< double > weights = problem.getWeights( 2.0, 3.0 );
    const Eigen::MatrixXd prior = Eigen::VectorXd::LinSpaced( 6, 1.0, 50.0 ).asDiagonal( );
    Eigen::MatrixXd normalMatrix;
    const Eigen::VectorXd residuals = problem.solve( weights, prior, normalMatrix );
    const VarianceComponentEstimate estimate =
            computeHelmertVarianceComponents( problem.designMatrix, weights, normalMatrix, residuals, problem.groupRowIndices );

    const Eigen::VectorXd redundancyNumbers = Eigen::VectorXd::Ones( 500 ) -
            ( weights * problem.designMatrix * normalMatrix.inverse( ) * problem.designMatrix.transpose( ) ).diagonal( );
    for( int k = 0; k < 2; k++ )
    {
        double expectedRedundancy = 0.0, expectedSquareSum = 0.0;
        for( int row : problem.groupRowIndices.at( k ) )
        {
            expectedRedundancy += redundancyNumbers( row );
            expectedSquareSum += weights.coeff( row, row ) * residuals( row ) * residuals( row );
        }
        BOOST_CHECK_CLOSE_FRACTION( estimate.groupResults.at( k ).redundancy, expectedRedundancy, 1.0E-10 );
        BOOST_CHECK_CLOSE_FRACTION( estimate.groupResults.at( k ).weightedSquareSum, expectedSquareSum, 1.0E-12 );
    }

    const Eigen::VectorXd scaling = Eigen::VectorXd::LinSpaced( 6, 1.0E-4, 1.0E4 );
    const VarianceComponentEstimate scaledEstimate =
            computeHelmertVarianceComponents( problem.designMatrix * scaling.asDiagonal( ),
                                              weights,
                                              scaling.asDiagonal( ) * normalMatrix * scaling.asDiagonal( ),
                                              residuals,
                                              problem.groupRowIndices );
    for( int k = 0; k < 2; k++ )
    {
        BOOST_CHECK_CLOSE_FRACTION(
                scaledEstimate.groupResults.at( k ).varianceFactor, estimate.groupResults.at( k ).varianceFactor, 1.0E-9 );
    }
}

//! Starting 10x off in sigma, Helmert's method and the simplified estimator must reach the same weights, close to the true
//! noise. Variants: no prior, a prior, rows in no group, and noise correlated in pairs with matching weights.
BOOST_AUTO_TEST_CASE( testSigmaRecovery )
{
    const std::vector< double > trueSigmas = { 0.5, 20.0 };
    for( int variant = 0; variant < 4; variant++ )
    {
        LinearProblem problem( variant == 3 ? 0.8 : 0.0, trueSigmas );
        const Eigen::MatrixXd prior = ( variant == 1 ? 10.0 : 0.0 ) * Eigen::MatrixXd::Identity( 6, 6 );
        if( variant == 2 )
        {
            // The last 50 rows keep their true weight.
            problem.groupRowIndices.at( 1 ).resize( 50 );
        }

        std::vector< Eigen::Vector2d > finalSigmas;
        for( int useSimplified = 0; useSimplified < 2; useSimplified++ )
        {
            Eigen::Vector2d sigmas( 5.0, 2.0 );
            for( int iteration = 0; iteration < 200; iteration++ )
            {
                Eigen::SparseMatrix< double > weights = problem.getWeights( sigmas( 0 ), sigmas( 1 ) );
                for( int row = 450; variant == 2 && row < 500; row++ )
                {
                    weights.coeffRef( row, row ) = 1.0 / ( trueSigmas.at( 1 ) * trueSigmas.at( 1 ) );
                }
                Eigen::MatrixXd normalMatrix;
                const Eigen::VectorXd residuals = problem.solve( weights, prior, normalMatrix );
                const VarianceComponentEstimate estimate =
                        computeHelmertVarianceComponents( problem.designMatrix, weights, normalMatrix, residuals, problem.groupRowIndices );

                Eigen::Vector2d factors;
                for( int k = 0; k < 2; k++ )
                {
                    factors( k ) = useSimplified ? estimate.groupResults.at( k ).simplifiedVarianceFactor
                                                 : estimate.groupResults.at( k ).varianceFactor;
                }
                sigmas = sigmas.cwiseProduct( factors.cwiseSqrt( ) );
                if( ( factors.array( ) - 1.0 ).abs( ).maxCoeff( ) < 1.0E-12 )
                {
                    break;
                }
            }
            finalSigmas.push_back( sigmas );
        }

        for( int k = 0; k < 2; k++ )
        {
            BOOST_CHECK_CLOSE_FRACTION( finalSigmas.at( 0 )( k ), finalSigmas.at( 1 )( k ), 1.0E-8 );
            BOOST_CHECK_CLOSE_FRACTION( finalSigmas.at( 0 )( k ), trueSigmas.at( k ), 0.15 );
        }
    }
}

//! Orbit estimation from range (two stations) and position observations. Starting 10x off, the estimator must recover the
//! range noise of each arc and the correlated position noise. Run for one arc and for two arcs with their own range noise.
void checkOrbitSigmaRecovery( const bool useMultiArc )
{
    const double gravitationalParameter = 3.986004418E14;
    const double arcDuration = 4.0 * 3600.0;
    const int numberOfArcs = useMultiArc ? 2 : 1;
    std::vector< double > arcStartTimes, arcEndTimes;
    for( int arc = 0; arc < numberOfArcs; arc++ )
    {
        arcStartTimes.push_back( arc * arcDuration );
        arcEndTimes.push_back( ( arc + 1 ) * arcDuration );
    }

    // Earth with a fixed position and orientation, and a vehicle whose orbit is estimated.
    BodyListSettings bodySettings( "SSB", "J2000" );
    bodySettings.addSettings( "Earth" );
    bodySettings.at( "Earth" )->ephemerisSettings = constantEphemerisSettings( Eigen::Vector6d::Zero( ), "SSB", "J2000" );
    bodySettings.at( "Earth" )->gravityFieldSettings = centralGravitySettings( gravitationalParameter );
    bodySettings.at( "Earth" )->shapeModelSettings = sphericalBodyShapeSettings( 6.371E6 );
    bodySettings.at( "Earth" )->rotationModelSettings =
            constantRotationModelSettings( "J2000", "IAU_Earth", Eigen::Quaterniond::Identity( ) );
    for( int i = 0; i < 2; i++ )
    {
        bodySettings.at( "Earth" )->groundStationSettings.push_back( groundStationSettings( "Station" + std::to_string( i ),
                                                                                            Eigen::Vector3d( 0.0, 0.3 * i, 1.0 + 2.0 * i ),
                                                                                            coordinate_conversions::geodetic_position ) );
    }
    bodySettings.addSettings( "Vehicle" );
    bodySettings.at( "Vehicle" )->ephemerisSettings =
            std::make_shared< TabulatedEphemerisSettings >( std::map< double, Eigen::Vector6d >( ), "Earth", "J2000" );
    bodySettings.at( "Vehicle" )->ephemerisSettings->resetMakeMultiArcEphemeris( useMultiArc );
    SystemOfBodies bodies = createSystemOfBodies( bodySettings );

    SelectedAccelerationMap accelerationMap;
    accelerationMap[ "Vehicle" ][ "Earth" ].push_back( std::make_shared< AccelerationSettings >( point_mass_gravity ) );
    const std::vector< std::string > bodiesToIntegrate = { "Vehicle" };
    const std::vector< std::string > centralBodies = { "Earth" };
    AccelerationMap accelerationModelMap = createAccelerationModelsMap( bodies, accelerationMap, bodiesToIntegrate, centralBodies );

    Eigen::Vector6d keplerianElements;
    keplerianElements << 7200.0E3, 0.05, 1.2, 0.5, 0.3, 0.0;
    const Eigen::Vector6d initialState =
            orbital_element_conversions::convertKeplerianToCartesianElements( keplerianElements, gravitationalParameter );
    std::vector< std::shared_ptr< SingleArcPropagatorSettings< double, double > > > arcPropagatorSettings;
    for( int arc = 0; arc < numberOfArcs; arc++ )
    {
        arcPropagatorSettings.push_back( std::make_shared< TranslationalStatePropagatorSettings< double, double > >(
                centralBodies,
                accelerationModelMap,
                bodiesToIntegrate,
                initialState * ( 1.0 + 0.01 * arc ),
                arcStartTimes.at( arc ),
                rungeKutta4Settings< double >( 10.0 ),
                propagationTimeTerminationSettings( arcEndTimes.at( arc ) ) ) );
    }
    std::shared_ptr< PropagatorSettings< double > > propagatorSettings;
    std::vector< std::shared_ptr< EstimatableParameterSettings > > parameterNames;
    if( useMultiArc )
    {
        auto multiArcSettings = std::make_shared< MultiArcPropagatorSettings< double, double > >( arcPropagatorSettings );
        parameterNames = getInitialMultiArcParameterSettings< double, double >( multiArcSettings, bodies, arcStartTimes );
        propagatorSettings = multiArcSettings;
    }
    else
    {
        parameterNames = getInitialStateParameterSettings< double, double >( arcPropagatorSettings.at( 0 ), bodies );
        propagatorSettings = arcPropagatorSettings.at( 0 );
    }
    std::shared_ptr< EstimatableParameterSet< double > > parametersToEstimate =
            createParametersToEstimate< double >( parameterNames, bodies );

    std::vector< LinkEnds > rangeLinkEnds( 2 );
    for( int i = 0; i < 2; i++ )
    {
        rangeLinkEnds.at( i )[ transmitter ] = LinkEndId( "Earth", "Station" + std::to_string( i ) );
        rangeLinkEnds.at( i )[ receiver ] = LinkEndId( "Vehicle", "" );
    }
    LinkEnds positionLinkEnds;
    positionLinkEnds[ observed_body ] = LinkEndId( "Vehicle", "" );
    const std::vector< std::shared_ptr< ObservationModelSettings > > observationSettingsList = {
        std::make_shared< ObservationModelSettings >( one_way_range, rangeLinkEnds.at( 0 ) ),
        std::make_shared< ObservationModelSettings >( one_way_range, rangeLinkEnds.at( 1 ) ),
        std::make_shared< ObservationModelSettings >( position_observable, positionLinkEnds )
    };
    OrbitDeterminationManager< double, double > orbitDeterminationManager(
            bodies, parametersToEstimate, observationSettingsList, propagatorSettings );
    const Eigen::VectorXd trueParameters = parametersToEstimate->getFullParameterValues< double >( );

    // Range noise of 1 m and 3 m per arc. Position noise of 50 m, correlated between the components.
    const std::vector< double > trueRangeSigmas = { 1.0, 3.0 };
    const double truePositionSigma = 50.0;
    Eigen::Matrix3d positionCorrelation;
    positionCorrelation << 1.0, 0.5, 0.0, 0.5, 1.0, 0.3, 0.0, 0.3, 1.0;
    const Eigen::Matrix3d positionCholesky = positionCorrelation.llt( ).matrixL( );
    auto generator = std::make_shared< std::mt19937 >( 1234 );
    auto createNoiseFunction = [ generator ]( const Eigen::MatrixXd& noiseFactor ) {
        return std::function< Eigen::VectorXd( const double ) >( [ generator, noiseFactor ]( const double ) {
            std::normal_distribution< double > normal( 0.0, 1.0 );
            Eigen::VectorXd noise( noiseFactor.cols( ) );
            for( int i = 0; i < noise.size( ); i++ )
            {
                noise( i ) = normal( *generator );
            }
            return Eigen::VectorXd( noiseFactor * noise );
        } );
    };

    std::vector< std::shared_ptr< ObservationSimulationSettings< double > > > simulationSettings;
    for( int arc = 0; arc < numberOfArcs; arc++ )
    {
        std::vector< double > rangeTimes, positionTimes;
        for( double time = arcStartTimes.at( arc ) + 60.0; time < arcEndTimes.at( arc ) - 60.0; time += 30.0 )
        {
            rangeTimes.push_back( time );
            if( rangeTimes.size( ) % 20 == 1 )
            {
                positionTimes.push_back( time );
            }
        }
        for( const LinkEnds& linkEnds : rangeLinkEnds )
        {
            simulationSettings.push_back(
                    std::make_shared< TabulatedObservationSimulationSettings< double > >( one_way_range, linkEnds, rangeTimes, receiver ) );
            simulationSettings.back( )->setObservationNoiseFunction(
                    createNoiseFunction( Eigen::MatrixXd::Constant( 1, 1, trueRangeSigmas.at( arc ) ) ) );
        }
        simulationSettings.push_back( std::make_shared< TabulatedObservationSimulationSettings< double > >(
                position_observable, positionLinkEnds, positionTimes, observed_body ) );
        simulationSettings.back( )->setObservationNoiseFunction( createNoiseFunction( truePositionSigma * positionCholesky ) );
    }
    std::shared_ptr< ObservationDataset< double, double > > observations = simulateObservationDataset< double, double >(
            simulationSettings, orbitDeterminationManager.getObservationSimulators( ), bodies );

    // One group per observable type and arc. Start with range sigma 10 m and position sigma 5 m.
    using Condition = ObservationSelectionCondition< double, double >;
    std::vector< Condition > groups;
    std::vector< double > trueSigmas, initialSigmas;
    for( int arc = 0; arc < numberOfArcs; arc++ )
    {
        const Condition arcCondition = Condition::timeBounds( arcStartTimes.at( arc ), arcEndTimes.at( arc ) );
        groups.push_back( Condition::observableType( one_way_range ) && arcCondition );
        observations->setConstantSingleObservationScalarWeight( groups.back( ), 1.0 / 100.0 );
        trueSigmas.push_back( trueRangeSigmas.at( arc ) );
        initialSigmas.push_back( 10.0 );

        groups.push_back( Condition::observableType( position_observable ) && arcCondition );
        observations->setConstantSingleObservationMatrixWeight( groups.back( ), positionCorrelation.inverse( ) / 25.0 );
        trueSigmas.push_back( truePositionSigma );
        initialSigmas.push_back( 5.0 );
    }

    Eigen::VectorXd perturbedParameters = trueParameters;
    for( int arc = 0; arc < numberOfArcs; arc++ )
    {
        perturbedParameters.segment( 6 * arc, 3 ) += Eigen::Vector3d::Constant( 100.0 );
        perturbedParameters.segment( 6 * arc + 3, 3 ) += Eigen::Vector3d::Constant( 0.1 );
    }
    parametersToEstimate->resetParameterValues( perturbedParameters );

    auto estimationInput = std::make_shared< EstimationInput< double, double > >(
            observations, Eigen::MatrixXd::Zero( 0, 0 ), estimationConvergenceChecker( 6 ) );
    estimationInput->defineEstimationSettings( true, true, true, false, false, false );
    auto output = estimateParametersWithVarianceComponents(
            orbitDeterminationManager, estimationInput, varianceComponentSettings< double, double >( groups, 0.01, 15 ) );

    BOOST_CHECK( output->isConverged );
    for( unsigned int k = 0; k < groups.size( ); k++ )
    {
        const double estimatedSigma = initialSigmas.at( k ) * std::sqrt( output->totalVarianceFactorHistory.back( )( k ) );
        BOOST_CHECK_CLOSE_FRACTION( estimatedSigma, trueSigmas.at( k ), k % 2 == 0 ? 0.08 : 0.3 );
    }
}

BOOST_AUTO_TEST_CASE( testOrbitSigmaRecovery )
{
    checkOrbitSigmaRecovery( false );
    checkOrbitSigmaRecovery( true );
}

BOOST_AUTO_TEST_SUITE_END( )

}  // namespace unit_tests

}  // namespace tudat

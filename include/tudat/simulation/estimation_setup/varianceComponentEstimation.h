/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 *
 *    References
 *      Sahin, M., Cross, P. A., and Sellers, P. C. (1992). Variance component estimation applied to satellite
 *          laser ranging. Bulletin Geodesique, 66(3):284-295. doi:10.1007/BF02033189
 *      Foerstner, W. (1979). Ein Verfahren zur Schaetzung von Varianz- und Kovarianzkomponenten.
 *          Allgemeine Vermessungs-Nachrichten, 86:446-453.
 */

#ifndef TUDAT_VARIANCECOMPONENTESTIMATION_H
#define TUDAT_VARIANCECOMPONENTESTIMATION_H

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "tudat/astro/orbit_determination/podInputOutputTypes.h"
#include "tudat/simulation/estimation_setup/observationCondition.h"
#include "tudat/simulation/estimation_setup/observationDataset.h"
#include "tudat/simulation/estimation_setup/orbitDeterminationManager.h"

namespace tudat
{

namespace simulation_setup
{

//! Methods to estimate the variance factor of each observation group.
enum class VarianceComponentEstimatorType {
    //! Helmert's equations, which couple the groups (Sahin et al., 1992, Eq. 19). They can give factors that are not
    //! positive when the starting weights are far off.
    helmert_estimator,
    //! Weighted square sum of each group divided by its redundancy (Foerstner, 1979). Its factors are always positive.
    simplified_helmert_estimator
};

//! Variance component results for one group of observations.
struct VarianceComponentGroupResult {
    //! Number of observations in the group with a non-zero weight.
    int numberOfObservations = 0;

    //! Number of observations minus the share used up by the estimated parameters.
    double redundancy = 0.0;

    //! Weighted sum of squared residuals.
    double weightedSquareSum = 0.0;

    //! Estimated variance factor. The group weights are divided by this value.
    double varianceFactor = 1.0;

    //! Root mean square of the residuals.
    double rmsResidual = 0.0;

    //! Whether the group kept a factor of 1 because its redundancy is below 1.
    bool isFrozen = false;
};

//! Variance component results for all groups, from one least-squares solution.
struct VarianceComponentEstimate {
    //! Results per group, in the order of the groups.
    std::vector< VarianceComponentGroupResult > groupResults;
};

//! Function to estimate the variance factor of each observation group with Helmert's method.
/*!
 * Estimates one variance factor per group from one least-squares solution, with Helmert's equations (Sahin et al., 1992,
 * Eq. 19) or the simplified estimator. Parts of the normal matrix that belong to no group, such as a priori information,
 * keep a factor of 1. Weight correlations are only allowed within a group. The design matrix and normal matrix may be
 * normalized, as long as both use the same normalization.
 * \param designMatrix Design matrix, one row per observation
 * \param weightsMatrix Observation weight matrix
 * \param normalMatrix Full normal matrix (inverse covariance) of the solution
 * \param residuals Post-fit residuals
 * \param groupRowIndices Rows of each group
 * \param estimatorType Method used to compute the variance factors
 * \return Variance factor and related values per group
 */
VarianceComponentEstimate computeHelmertVarianceComponents(
        const Eigen::MatrixXd& designMatrix,
        const Eigen::SparseMatrix< double >& weightsMatrix,
        const Eigen::MatrixXd& normalMatrix,
        const Eigen::VectorXd& residuals,
        const std::vector< std::vector< int > >& groupRowIndices,
        const VarianceComponentEstimatorType estimatorType = VarianceComponentEstimatorType::simplified_helmert_estimator );

//! Settings for an estimation in which the weight of each observation group is estimated.
template< typename ObservationScalarType = double, typename TimeType = double >
struct VarianceComponentSettings {
    VarianceComponentSettings(
            const std::vector< observation_models::ObservationSelectionCondition< ObservationScalarType, TimeType > >& groups,
            const double tolerance,
            const int maximumNumberOfIterations,
            const VarianceComponentEstimatorType estimatorType,
            const int leastSquaresIterationsPerUpdate ):
        groups_( groups ), tolerance_( tolerance ), maximumNumberOfIterations_( maximumNumberOfIterations ),
        estimatorType_( estimatorType ), leastSquaresIterationsPerUpdate_( leastSquaresIterationsPerUpdate )
    {
        if( groups_.empty( ) || maximumNumberOfIterations_ < 1 || leastSquaresIterationsPerUpdate_ < 0 )
        {
            throw std::runtime_error(
                    "Error when creating variance component settings: at least one group and one iteration are required, and the "
                    "number of least-squares iterations per update must not be negative." );
        }
    }

    //! One selection condition per group.
    std::vector< observation_models::ObservationSelectionCondition< ObservationScalarType, TimeType > > groups_;

    //! The iteration stops when all variance factors differ from 1 by less than this value.
    double tolerance_;

    //! Maximum number of least-squares estimations.
    int maximumNumberOfIterations_;

    //! Method used to compute the variance factors.
    VarianceComponentEstimatorType estimatorType_;

    //! Number of least-squares iterations between two weight updates. With 0, each estimation runs to convergence.
    int leastSquaresIterationsPerUpdate_;
};

//! Function to create settings for an estimation in which the weight of each observation group is estimated.
template< typename ObservationScalarType = double, typename TimeType = double >
std::shared_ptr< VarianceComponentSettings< ObservationScalarType, TimeType > > varianceComponentSettings(
        const std::vector< observation_models::ObservationSelectionCondition< ObservationScalarType, TimeType > >& groups,
        const double tolerance = 0.01,
        const int maximumNumberOfIterations = 10,
        const VarianceComponentEstimatorType estimatorType = VarianceComponentEstimatorType::simplified_helmert_estimator,
        const int leastSquaresIterationsPerUpdate = 3 )
{
    return std::make_shared< VarianceComponentSettings< ObservationScalarType, TimeType > >(
            groups, tolerance, maximumNumberOfIterations, estimatorType, leastSquaresIterationsPerUpdate );
}

//! Output of an estimation in which the weight of each observation group is estimated.
template< typename ObservationScalarType = double, typename TimeType = double >
struct VarianceComponentEstimationOutput {
    //! Output of the last least-squares estimation.
    std::shared_ptr< EstimationOutput< ObservationScalarType, TimeType > > estimationOutput;

    //! Variance component results of each iteration.
    std::vector< VarianceComponentEstimate > iterationResults;

    //! Total scaling of the group variances used in each iteration (product of all earlier variance factors).
    std::vector< Eigen::VectorXd > totalVarianceFactorHistory;

    //! Best iteration of each least-squares estimation.
    std::vector< int > leastSquaresBestIterations;

    //! Whether all variance factors converged to 1 within the tolerance.
    bool isConverged = false;
};

//! Function to get the estimation rows of each group.
/*!
 * \param rowObservationIds Observation id of each estimation row
 * \param observationDataset Observations of the estimation
 * \param groups One selection condition per group
 * \return Rows of each group
 */
template< typename ObservationScalarType, typename TimeType >
std::vector< std::vector< int > > getVarianceComponentGroupRows(
        const std::vector< unsigned int >& rowObservationIds,
        const std::shared_ptr< observation_models::ObservationDataset< ObservationScalarType, TimeType > > observationDataset,
        const std::vector< observation_models::ObservationSelectionCondition< ObservationScalarType, TimeType > >& groups )
{
    std::vector< std::vector< int > > groupRows;
    std::vector< bool > isRowInGroup( rowObservationIds.size( ), false );
    for( const auto& group : groups )
    {
        const std::vector< unsigned int > selectedIds = observationDataset->getObservationIdsMatchingCondition( group );
        const std::unordered_set< unsigned int > selected( selectedIds.begin( ), selectedIds.end( ) );
        std::vector< int > rows;
        bool isOverlapping = false;
        for( unsigned int row = 0; row < rowObservationIds.size( ); row++ )
        {
            if( selected.count( rowObservationIds.at( row ) ) )
            {
                isOverlapping = isOverlapping || isRowInGroup.at( row );
                isRowInGroup.at( row ) = true;
                rows.push_back( row );
            }
        }
        if( rows.empty( ) || isOverlapping )
        {
            throw std::runtime_error(
                    "Error when estimating variance components: each group must contain active observations and groups must not "
                    "overlap." );
        }
        groupRows.push_back( rows );
    }
    return groupRows;
}

//! Function to estimate the variance factor of each observation group from an estimation output.
/*!
 * The rows are those that were active in the best iteration, so observations rejected as outliers after it are still
 * included.
 * \param estimationOutput Output of the estimation, with the design matrix and the iteration history saved
 * \param observationDataset Observations, with the same weights as in the estimation
 * \param groups One selection condition per group
 * \param estimatorType Method used to compute the variance factors
 * \return Variance factor and related values per group
 */
template< typename ObservationScalarType, typename TimeType >
VarianceComponentEstimate computeHelmertVarianceComponents(
        const std::shared_ptr< EstimationOutput< ObservationScalarType, TimeType > > estimationOutput,
        const std::shared_ptr< observation_models::ObservationDataset< ObservationScalarType, TimeType > > observationDataset,
        const std::vector< observation_models::ObservationSelectionCondition< ObservationScalarType, TimeType > >& groups,
        const VarianceComponentEstimatorType estimatorType = VarianceComponentEstimatorType::simplified_helmert_estimator )
{
    const observation_models::ObservationVectorData< ObservationScalarType, TimeType > allRows =
            observationDataset->createOrderedObservationVectorData( true );
    const int bestIteration = estimationOutput->bestIteration_;
    if( !estimationOutput->designMatrixSaved_ ||
        estimationOutput->getNormalizedInverseCovarianceMatrix( ).rows( ) != estimationOutput->getNormalizedDesignMatrix( ).cols( ) ||
        static_cast< int >( estimationOutput->activeFlagsPerIteration_.size( ) ) <= bestIteration ||
        estimationOutput->activeFlagsPerIteration_.at( bestIteration ).rows( ) != allRows.getWeightVector( ).rows( ) )
    {
        throw std::runtime_error(
                "Error when computing variance components: the estimation output must contain the design matrix and the iteration "
                "history, have no linear parameter constraints and match the observation dataset." );
    }

    // Observation ids and weights of the rows that were active in the best iteration.
    const auto& isRowActive = estimationOutput->activeFlagsPerIteration_.at( bestIteration );
    std::vector< unsigned int > rowObservationIds;
    std::vector< double > rowWeights;
    for( int row = 0; row < isRowActive.rows( ); row++ )
    {
        if( isRowActive( row ) )
        {
            rowObservationIds.push_back( allRows.getObservationIds( ).at( row ) );
            rowWeights.push_back( allRows.getWeightVector( )( row ) );
        }
    }
    if( static_cast< int >( rowWeights.size( ) ) != estimationOutput->weightsMatrixDiagonal_.rows( ) ||
        Eigen::Map< const Eigen::VectorXd >( rowWeights.data( ), rowWeights.size( ) ) != estimationOutput->weightsMatrixDiagonal_ )
    {
        throw std::runtime_error(
                "Error when computing variance components: the observation weights differ from those used in the estimation." );
    }

    return computeHelmertVarianceComponents( estimationOutput->getNormalizedDesignMatrix( ),
                                             estimationOutput->getWeightsMatrix( ),
                                             estimationOutput->getNormalizedInverseCovarianceMatrix( ),
                                             estimationOutput->residuals_,
                                             getVarianceComponentGroupRows( rowObservationIds, observationDataset, groups ),
                                             estimatorType );
}

//! Function to estimate parameters, while estimating the weight of each observation group with Helmert's method.
/*!
 * Follows Sahin et al. (1992, Fig. 1): estimate the parameters, estimate a variance factor per group, divide the group
 * weights by it, and repeat until all factors are close to 1. To save time, the weights are updated after a few
 * least-squares iterations, and only the final estimation runs to convergence. Outlier rejection runs inside each
 * estimation. The weights in the observation dataset are changed and end up as the estimated weights.
 * \param orbitDeterminationManager Object used to perform the estimation
 * \param estimationInput Input to the estimation, with the starting weights
 * \param settings Groups and iteration settings
 * \return Output of the last estimation and the variance factors of each iteration
 */
template< typename ObservationScalarType, typename TimeType >
std::shared_ptr< VarianceComponentEstimationOutput< ObservationScalarType, TimeType > > estimateParametersWithVarianceComponents(
        OrbitDeterminationManager< ObservationScalarType, TimeType >& orbitDeterminationManager,
        const std::shared_ptr< EstimationInput< ObservationScalarType, TimeType > > estimationInput,
        const std::shared_ptr< VarianceComponentSettings< ObservationScalarType, TimeType > > settings )
{
    using ParameterVectorType = Eigen::Matrix< ObservationScalarType, Eigen::Dynamic, 1 >;

    auto parametersToEstimate = orbitDeterminationManager.getParametersToEstimate( );
    Eigen::MatrixXd constraintMultiplier;
    Eigen::VectorXd constraintRightHandSide;
    parametersToEstimate->getConstraints( constraintMultiplier, constraintRightHandSide );
    // With an a priori constraint on the total parameter deviation, every estimation must start from the initial values, so
    // that the a priori values stay the same. A few iterations from the initial values are not enough to update the weights.
    const bool startFromPreviousResult = !estimationInput->getApplyAprioriParameterDeviation( );
    if( !estimationInput->getSaveDesignMatrix( ) || !estimationInput->getSaveResidualsAndParametersFromEachIteration( ) ||
        constraintMultiplier.rows( ) > 0 || ( !startFromPreviousResult && settings->leastSquaresIterationsPerUpdate_ > 0 ) )
    {
        throw std::runtime_error(
                "Error when estimating variance components: the design matrix and the iteration history must be saved, linear "
                "parameter constraints are not supported, and with an a priori constraint on the total parameter deviation the "
                "number of least-squares iterations per update must be 0." );
    }

    auto observationDataset = estimationInput->getObservationDataset( );
    getVarianceComponentGroupRows(
            observationDataset->createOrderedObservationVectorData( false ).getObservationIds( ), observationDataset, settings->groups_ );
    const ParameterVectorType initialParameterValues = parametersToEstimate->template getFullParameterValues< ObservationScalarType >( );

    // Copy of the input that stops after a few least-squares iterations. It shares the observations with the input.
    auto shortEstimationInput = std::make_shared< EstimationInput< ObservationScalarType, TimeType > >( *estimationInput );
    shortEstimationInput->setConvergenceChecker(
            std::make_shared< EstimationConvergenceChecker >( settings->leastSquaresIterationsPerUpdate_ ) );
    bool isFullEstimation = ( settings->leastSquaresIterationsPerUpdate_ == 0 );

    auto output = std::make_shared< VarianceComponentEstimationOutput< ObservationScalarType, TimeType > >( );
    Eigen::VectorXd totalVarianceFactors = Eigen::VectorXd::Ones( settings->groups_.size( ) );
    for( int iteration = 0; iteration < settings->maximumNumberOfIterations_; iteration++ )
    {
        // After an estimation, the parameters hold the last iterate, which is not always the best one.
        const ParameterVectorType startValues = ( startFromPreviousResult && output->estimationOutput != nullptr )
                ? output->estimationOutput->parameterEstimate_
                : initialParameterValues;
        if( estimationInput->getReintegrateEquationsOnFirstIteration( ) )
        {
            parametersToEstimate->template resetParameterValues< ObservationScalarType >( startValues );
        }
        else
        {
            orbitDeterminationManager.resetParameterEstimate( startValues, estimationInput->getReintegrateVariationalEquations( ) );
        }

        output->totalVarianceFactorHistory.push_back( totalVarianceFactors );
        output->estimationOutput =
                orbitDeterminationManager.estimateParameters( isFullEstimation ? estimationInput : shortEstimationInput );
        output->leastSquaresBestIterations.push_back( output->estimationOutput->bestIteration_ );
        output->iterationResults.push_back( computeHelmertVarianceComponents(
                output->estimationOutput, observationDataset, settings->groups_, settings->estimatorType_ ) );
        const VarianceComponentEstimate& estimate = output->iterationResults.back( );

        // Once the factors are close to 1, the next estimation runs to convergence. The result is converged when its factors
        // are still close to 1.
        bool areFactorsConverged = true;
        for( unsigned int k = 0; k < estimate.groupResults.size( ); k++ )
        {
            const VarianceComponentGroupResult& groupResult = estimate.groupResults.at( k );
            areFactorsConverged = areFactorsConverged && std::fabs( groupResult.varianceFactor - 1.0 ) < settings->tolerance_;
            if( estimationInput->getPrintOutput( ) )
            {
                std::cout << "Variance component iteration " << iteration << ", group " << k << ": variance factor "
                          << groupResult.varianceFactor << ", rms residual " << groupResult.rmsResidual << std::endl;
            }
        }

        output->isConverged = areFactorsConverged && isFullEstimation;
        isFullEstimation = areFactorsConverged || settings->leastSquaresIterationsPerUpdate_ == 0;

        // Stop without changing the weights, so that they match the returned estimation output.
        if( output->isConverged || iteration == settings->maximumNumberOfIterations_ - 1 )
        {
            break;
        }
        for( unsigned int k = 0; k < estimate.groupResults.size( ); k++ )
        {
            observationDataset->scaleWeights( settings->groups_.at( k ), 1.0 / estimate.groupResults.at( k ).varianceFactor );
            totalVarianceFactors( k ) *= estimate.groupResults.at( k ).varianceFactor;
        }
    }

    if( !output->isConverged )
    {
        std::cerr << "Warning when estimating variance components: the variance factors did not converge within "
                  << settings->maximumNumberOfIterations_ << " iterations." << std::endl;
    }
    return output;
}

}  // namespace simulation_setup

}  // namespace tudat

#endif  // TUDAT_VARIANCECOMPONENTESTIMATION_H

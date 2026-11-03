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

#include <Eigen/QR>

#include "tudat/simulation/estimation_setup/varianceComponentEstimation.h"

namespace tudat
{

namespace simulation_setup
{

VarianceComponentEstimate computeHelmertVarianceComponents( const Eigen::SparseMatrix< double >& designMatrix,
                                                            const Eigen::SparseMatrix< double >& weightsMatrix,
                                                            const Eigen::MatrixXd& covarianceMatrix,
                                                            const Eigen::VectorXd& residuals,
                                                            const std::vector< std::vector< int > >& groupRowIndices,
                                                            const VarianceComponentEstimatorType estimatorType )
{
    const int numberOfObservations = designMatrix.rows( );
    const int numberOfGroups = groupRowIndices.size( );

    // Weight correlations between groups (or between a group and ungrouped observations) would need their own component.
    std::vector< int > groupOfRow( numberOfObservations, -1 );
    for( int k = 0; k < numberOfGroups; k++ )
    {
        for( int row : groupRowIndices.at( k ) )
        {
            groupOfRow.at( row ) = k;
        }
    }
    for( int column = 0; column < weightsMatrix.outerSize( ); column++ )
    {
        for( Eigen::SparseMatrix< double >::InnerIterator it( weightsMatrix, column ); it; ++it )
        {
            if( it.value( ) != 0.0 && groupOfRow.at( it.row( ) ) != groupOfRow.at( column ) )
            {
                throw std::runtime_error(
                        "Error when computing variance components: correlated observation weights must be in the same group." );
            }
        }
    }

    // Per group: N_k = A_k^T W_k A_k, the redundancy n_k - tr(P N_k) and c_k = v_k^T W_k v_k, with P the covariance. The
    // trace only needs the entries of P where N_k is not zero.
    VarianceComponentEstimate estimate;
    estimate.groupResults.resize( numberOfGroups );
    std::vector< Eigen::SparseMatrix< double > > groupNormalMatrices( numberOfGroups );
    for( int k = 0; k < numberOfGroups; k++ )
    {
        const std::vector< int >& rows = groupRowIndices.at( k );
        std::vector< Eigen::Triplet< double > > selectionEntries;
        for( unsigned int i = 0; i < rows.size( ); i++ )
        {
            selectionEntries.emplace_back( i, rows.at( i ), 1.0 );
        }
        Eigen::SparseMatrix< double > rowSelection( rows.size( ), numberOfObservations );
        rowSelection.setFromTriplets( selectionEntries.begin( ), selectionEntries.end( ) );

        const Eigen::SparseMatrix< double > groupDesignMatrix = rowSelection * designMatrix;
        const Eigen::SparseMatrix< double > groupWeights = rowSelection * weightsMatrix * rowSelection.transpose( );
        const Eigen::VectorXd groupResiduals = rowSelection * residuals;
        groupNormalMatrices.at( k ) = groupDesignMatrix.transpose( ) * groupWeights * groupDesignMatrix;

        VarianceComponentGroupResult& groupResult = estimate.groupResults.at( k );
        groupResult.numberOfObservations = ( Eigen::VectorXd( groupWeights.diagonal( ) ).array( ) > 0.0 ).count( );
        groupResult.weightedSquareSum = groupResiduals.dot( groupWeights * groupResiduals );
        groupResult.rmsResidual = std::sqrt( groupResiduals.squaredNorm( ) / rows.size( ) );
        groupResult.redundancy = groupResult.numberOfObservations - groupNormalMatrices.at( k ).cwiseProduct( covarianceMatrix ).sum( );
        groupResult.isFrozen = groupResult.redundancy < 1.0;
        if( !groupResult.isFrozen )
        {
            groupResult.varianceFactor = groupResult.weightedSquareSum / groupResult.redundancy;
        }
    }
    if( estimatorType == VarianceComponentEstimatorType::simplified_helmert_estimator )
    {
        return estimate;
    }

    // Helmert's equations, for the groups that are not frozen. The rest of the normal matrix (a priori information,
    // ungrouped observations and frozen groups) has a factor of 1 and moves to the right-hand side. With M_k = P N_k,
    // the coupling terms are tr(M_k M_l), and the fixed part gives tr(M_k (I - sum_l M_l)).
    std::vector< int > activeGroups;
    for( int k = 0; k < numberOfGroups; k++ )
    {
        if( !estimate.groupResults.at( k ).isFrozen )
        {
            activeGroups.push_back( k );
        }
    }
    const int numberOfActiveGroups = activeGroups.size( );
    std::vector< Eigen::MatrixXd > covarianceTimesGroupNormalMatrices;
    for( int k : activeGroups )
    {
        covarianceTimesGroupNormalMatrices.push_back( covarianceMatrix * groupNormalMatrices.at( k ) );
    }
    Eigen::MatrixXd helmertMatrix( numberOfActiveGroups, numberOfActiveGroups );
    Eigen::VectorXd rightHandSide( numberOfActiveGroups );
    for( int i = 0; i < numberOfActiveGroups; i++ )
    {
        const Eigen::MatrixXd& firstProduct = covarianceTimesGroupNormalMatrices.at( i );
        for( int j = 0; j < numberOfActiveGroups; j++ )
        {
            // tr(M_i M_j), without forming the product.
            helmertMatrix( i, j ) = firstProduct.cwiseProduct( covarianceTimesGroupNormalMatrices.at( j ).transpose( ) ).sum( );
        }
        const VarianceComponentGroupResult& groupResult = estimate.groupResults.at( activeGroups.at( i ) );
        rightHandSide( i ) = groupResult.weightedSquareSum - firstProduct.trace( ) + helmertMatrix.row( i ).sum( );
        helmertMatrix( i, i ) += groupResult.numberOfObservations - 2.0 * firstProduct.trace( );
    }
    const Eigen::VectorXd helmertVarianceFactors = helmertMatrix.colPivHouseholderQr( ).solve( rightHandSide );

    // Helmert's method does not keep the factors positive, for instance when the starting weights are far off.
    if( !( helmertVarianceFactors.array( ) > 0.0 ).all( ) || !helmertVarianceFactors.allFinite( ) )
    {
        throw std::runtime_error(
                "Error when computing variance components: Helmert's equations gave a variance factor that is not positive. The "
                "simplified Helmert estimator always gives positive factors." );
    }
    for( int i = 0; i < numberOfActiveGroups; i++ )
    {
        estimate.groupResults.at( activeGroups.at( i ) ).varianceFactor = helmertVarianceFactors( i );
    }
    return estimate;
}

}  // namespace simulation_setup

}  // namespace tudat

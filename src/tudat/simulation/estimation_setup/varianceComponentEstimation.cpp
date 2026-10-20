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

#include <Eigen/Cholesky>
#include <Eigen/QR>

#include "tudat/simulation/estimation_setup/varianceComponentEstimation.h"

namespace tudat
{

namespace simulation_setup
{

VarianceComponentEstimate computeHelmertVarianceComponents( const Eigen::MatrixXd& designMatrix,
                                                            const Eigen::SparseMatrix< double >& weightsMatrix,
                                                            const Eigen::MatrixXd& normalMatrix,
                                                            const Eigen::VectorXd& residuals,
                                                            const std::vector< std::vector< int > >& groupRowIndices )
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

    // Per group: N^-1 N_k with N_k = A_k^T W_k A_k, and c_k = v_k^T W_k v_k.
    const Eigen::LDLT< Eigen::MatrixXd > normalMatrixFactorization( normalMatrix );
    std::vector< int > localRowIndex( numberOfObservations, -1 );
    VarianceComponentEstimate estimate;
    estimate.groupResults.resize( numberOfGroups );
    std::vector< Eigen::MatrixXd > solvedGroupNormalMatrices( numberOfGroups );
    Eigen::MatrixXd fixedNormalMatrix = normalMatrix;
    for( int k = 0; k < numberOfGroups; k++ )
    {
        const std::vector< int >& rows = groupRowIndices.at( k );
        Eigen::MatrixXd groupDesignMatrix( rows.size( ), designMatrix.cols( ) );
        Eigen::VectorXd groupResiduals( rows.size( ) );
        std::vector< Eigen::Triplet< double > > groupWeightEntries;
        for( unsigned int i = 0; i < rows.size( ); i++ )
        {
            localRowIndex.at( rows.at( i ) ) = i;
            groupDesignMatrix.row( i ) = designMatrix.row( rows.at( i ) );
            groupResiduals( i ) = residuals( rows.at( i ) );
        }
        for( unsigned int i = 0; i < rows.size( ); i++ )
        {
            for( Eigen::SparseMatrix< double >::InnerIterator it( weightsMatrix, rows.at( i ) ); it; ++it )
            {
                if( it.value( ) != 0.0 )
                {
                    groupWeightEntries.emplace_back( localRowIndex.at( it.row( ) ), i, it.value( ) );
                }
            }
        }
        Eigen::SparseMatrix< double > groupWeights( rows.size( ), rows.size( ) );
        groupWeights.setFromTriplets( groupWeightEntries.begin( ), groupWeightEntries.end( ) );

        const Eigen::MatrixXd groupNormalMatrix = groupDesignMatrix.transpose( ) * ( groupWeights * groupDesignMatrix );
        solvedGroupNormalMatrices.at( k ) = normalMatrixFactorization.solve( groupNormalMatrix );

        VarianceComponentGroupResult& groupResult = estimate.groupResults.at( k );
        groupResult.numberOfObservations = ( groupWeights.diagonal( ).array( ) > 0.0 ).count( );
        groupResult.weightedSquareSum = groupResiduals.dot( groupWeights * groupResiduals );
        groupResult.rmsResidual = std::sqrt( groupResiduals.squaredNorm( ) / rows.size( ) );
        groupResult.redundancy = groupResult.numberOfObservations - solvedGroupNormalMatrices.at( k ).trace( );
        groupResult.isFrozen = groupResult.redundancy < 1.0;
        groupResult.simplifiedVarianceFactor = groupResult.isFrozen ? 1.0 : groupResult.weightedSquareSum / groupResult.redundancy;
        if( !groupResult.isFrozen )
        {
            fixedNormalMatrix -= groupNormalMatrix;
        }
    }

    // Helmert's equations, for the groups that are not frozen. The rest of the normal matrix (a priori information,
    // ungrouped observations and frozen groups) has a factor of 1 and moves to the right-hand side.
    std::vector< int > activeGroups;
    for( int k = 0; k < numberOfGroups; k++ )
    {
        if( !estimate.groupResults.at( k ).isFrozen )
        {
            activeGroups.push_back( k );
        }
    }
    const Eigen::MatrixXd solvedFixedNormalMatrix = normalMatrixFactorization.solve( fixedNormalMatrix );
    const int numberOfActiveGroups = activeGroups.size( );
    Eigen::MatrixXd helmertMatrix( numberOfActiveGroups, numberOfActiveGroups );
    Eigen::VectorXd rightHandSide( numberOfActiveGroups );
    for( int i = 0; i < numberOfActiveGroups; i++ )
    {
        const Eigen::MatrixXd& solvedGroupNormalMatrix = solvedGroupNormalMatrices.at( activeGroups.at( i ) );
        for( int j = 0; j < numberOfActiveGroups; j++ )
        {
            // tr(N^-1 N_k N^-1 N_l), without forming the product.
            helmertMatrix( i, j ) =
                    solvedGroupNormalMatrix.cwiseProduct( solvedGroupNormalMatrices.at( activeGroups.at( j ) ).transpose( ) ).sum( );
        }
        const VarianceComponentGroupResult& groupResult = estimate.groupResults.at( activeGroups.at( i ) );
        helmertMatrix( i, i ) += groupResult.numberOfObservations - 2.0 * solvedGroupNormalMatrix.trace( );
        rightHandSide( i ) =
                groupResult.weightedSquareSum - solvedGroupNormalMatrix.cwiseProduct( solvedFixedNormalMatrix.transpose( ) ).sum( );
    }
    const Eigen::VectorXd helmertVarianceFactors = helmertMatrix.colPivHouseholderQr( ).solve( rightHandSide );

    // Helmert's method does not keep the factors positive. Then the simplified estimator of Foerstner (1979) is used, which
    // converges to the same factors.
    estimate.simplifiedEstimatorUsed = !( helmertVarianceFactors.array( ) > 0.0 ).all( ) || !helmertVarianceFactors.allFinite( );
    for( int i = 0; i < numberOfActiveGroups; i++ )
    {
        VarianceComponentGroupResult& groupResult = estimate.groupResults.at( activeGroups.at( i ) );
        groupResult.varianceFactor = estimate.simplifiedEstimatorUsed ? groupResult.simplifiedVarianceFactor : helmertVarianceFactors( i );
    }
    return estimate;
}

}  // namespace simulation_setup

}  // namespace tudat

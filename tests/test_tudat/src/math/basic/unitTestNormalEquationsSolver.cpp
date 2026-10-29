/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 *
 */

#define BOOST_TEST_MAIN

#include <boost/test/included/unit_test.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "tudat/math/basic/normalEquationsSolver.h"

namespace tudat
{

namespace unit_tests
{

using namespace linear_algebra;

BOOST_AUTO_TEST_SUITE( test_normal_equations_solver )

//! Test that all solvers give the same solution and inverse as a full-pivoting LU decomposition
BOOST_AUTO_TEST_CASE( testSolverAgreement )
{
    // Well-conditioned normal matrix from a random design matrix
    std::srand( 1 );
    const Eigen::MatrixXd designMatrix = Eigen::MatrixXd::Random( 50, 10 );
    const Eigen::MatrixXd normalMatrix = designMatrix.transpose( ) * designMatrix;
    const Eigen::VectorXd rightHandSide = Eigen::VectorXd::Random( 10 );

    const Eigen::VectorXd expectedSolution = normalMatrix.fullPivLu( ).solve( rightHandSide );
    const Eigen::MatrixXd expectedInverse = normalMatrix.fullPivLu( ).inverse( );

    for( const auto& solverSettings : { jacobiSvdSolver( ), bdcSvdSolver( ), choleskySolver( ) } )
    {
        auto solver = createNormalEquationsSolver< Eigen::MatrixXd >( solverSettings );
        solver->factorize( normalMatrix, TUDAT_NAN );
        BOOST_CHECK( Eigen::VectorXd( solver->solve( rightHandSide ) ).isApprox( expectedSolution, 1.0E-12 ) );
        BOOST_CHECK( solver->inverse( ).isApprox( expectedInverse, 1.0E-12 ) );
    }

    const Eigen::SparseMatrix< double > sparseNormalMatrix = normalMatrix.sparseView( );
    for( const auto& solverSettings : { sparseQrSolver( ), sparseCholeskySolver( ) } )
    {
        auto solver = createNormalEquationsSolver< Eigen::SparseMatrix< double > >( solverSettings );
        solver->factorize( sparseNormalMatrix, TUDAT_NAN );
        BOOST_CHECK( Eigen::VectorXd( solver->solve( rightHandSide ) ).isApprox( expectedSolution, 1.0E-12 ) );
        BOOST_CHECK( solver->inverse( ).isApprox( expectedInverse, 1.0E-12 ) );
    }
}

//! Test that a truncated SVD gives the minimum-norm solution and the pseudo-inverse of a rank-deficient system
BOOST_AUTO_TEST_CASE( testTruncatedSvd )
{
    // Normal matrix built from known eigenvalues, two of which are (almost) zero
    std::srand( 2 );
    const Eigen::MatrixXd eigenvectors = Eigen::MatrixXd::Random( 8, 8 ).householderQr( ).householderQ( );
    Eigen::VectorXd eigenvalues( 8 );
    eigenvalues << 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 1.0E-12, 1.0E-13;
    const Eigen::MatrixXd normalMatrix = eigenvectors * eigenvalues.asDiagonal( ) * eigenvectors.transpose( );
    const Eigen::VectorXd rightHandSide = Eigen::VectorXd::Random( 8 );

    // Minimum-norm solution: invert the six large eigenvalues and drop the two small ones
    Eigen::VectorXd truncatedInverseEigenvalues = Eigen::VectorXd::Zero( 8 );
    truncatedInverseEigenvalues.head( 6 ) = eigenvalues.head( 6 ).cwiseInverse( );
    const Eigen::MatrixXd expectedPseudoInverse = eigenvectors * truncatedInverseEigenvalues.asDiagonal( ) * eigenvectors.transpose( );
    const Eigen::VectorXd expectedSolution = expectedPseudoInverse * rightHandSide;

    for( const auto& solverSettings : { jacobiSvdSolver( 1.0E-8 ), bdcSvdSolver( 1.0E-8 ) } )
    {
        auto solver = createNormalEquationsSolver< Eigen::MatrixXd >( solverSettings );
        solver->factorize( normalMatrix, TUDAT_NAN );
        BOOST_CHECK( Eigen::VectorXd( solver->solve( rightHandSide ) ).isApprox( expectedSolution, 1.0E-12 ) );
        BOOST_CHECK( solver->inverse( ).isApprox( expectedPseudoInverse, 1.0E-12 ) );
    }
}

BOOST_AUTO_TEST_SUITE_END( )

}  // namespace unit_tests

}  // namespace tudat

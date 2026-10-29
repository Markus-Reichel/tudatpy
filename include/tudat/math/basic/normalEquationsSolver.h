/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_NORMALEQUATIONSSOLVER_H
#define TUDAT_NORMALEQUATIONSSOLVER_H

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>
#include <Eigen/SVD>

#include "tudat/math/basic/mathematicalConstants.h"

namespace tudat
{

namespace linear_algebra
{

//! Types of decomposition that can be used to solve the normal equations of a least-squares problem
enum class NormalEquationsSolverType { jacobi_svd, bdc_svd, cholesky, sparse_qr, sparse_cholesky };

//! Function to get the name of a normal equations solver type
inline std::string getNormalEquationsSolverName( const NormalEquationsSolverType solverType )
{
    switch( solverType )
    {
        case NormalEquationsSolverType::jacobi_svd:
            return "Jacobi SVD";
        case NormalEquationsSolverType::bdc_svd:
            return "BDC SVD";
        case NormalEquationsSolverType::cholesky:
            return "Cholesky";
        case NormalEquationsSolverType::sparse_qr:
            return "sparse QR";
        case NormalEquationsSolverType::sparse_cholesky:
            return "sparse Cholesky";
        default:
            throw std::runtime_error( "Error when getting normal equations solver name, solver type not recognized." );
    }
}

//! Class defining which decomposition is used to solve the normal equations
class NormalEquationsSolverSettings
{
public:
    //! Constructor
    /*!
     * Constructor
     * \param solverType Type of decomposition used to solve the normal equations
     * \param relativeSingularValueCutoff Singular values smaller than this value times the largest singular value are
     * treated as zero (truncated SVD). Only used by the SVD solvers. NaN means no truncation.
     */
    NormalEquationsSolverSettings( const NormalEquationsSolverType solverType, const double relativeSingularValueCutoff = TUDAT_NAN ):
        solverType_( solverType ), relativeSingularValueCutoff_( relativeSingularValueCutoff )
    {
        const bool isSvd = ( solverType == NormalEquationsSolverType::jacobi_svd || solverType == NormalEquationsSolverType::bdc_svd );
        const bool hasCutoff = !std::isnan( relativeSingularValueCutoff );
        if( hasCutoff && ( !isSvd || !( relativeSingularValueCutoff > 0.0 && relativeSingularValueCutoff < 1.0 ) ) )
        {
            throw std::runtime_error(
                    "Error when creating normal equations solver settings, a singular value cutoff must be between 0 and 1, and can "
                    "only be used with an SVD solver." );
        }
    }

    //! Function to get the type of decomposition used to solve the normal equations
    NormalEquationsSolverType getSolverType( ) const
    {
        return solverType_;
    }

    //! Function to get the relative singular value cutoff (NaN if no truncation is used)
    double getRelativeSingularValueCutoff( ) const
    {
        return relativeSingularValueCutoff_;
    }

protected:
    //! Type of decomposition used to solve the normal equations
    NormalEquationsSolverType solverType_;

    //! Relative singular value cutoff of the truncated SVD (NaN if no truncation is used)
    double relativeSingularValueCutoff_;
};

//! Function to create settings for a Jacobi SVD solver, optionally truncated
inline std::shared_ptr< NormalEquationsSolverSettings > jacobiSvdSolver( const double relativeSingularValueCutoff = TUDAT_NAN )
{
    return std::make_shared< NormalEquationsSolverSettings >( NormalEquationsSolverType::jacobi_svd, relativeSingularValueCutoff );
}

//! Function to create settings for a divide-and-conquer (BDC) SVD solver, optionally truncated
inline std::shared_ptr< NormalEquationsSolverSettings > bdcSvdSolver( const double relativeSingularValueCutoff = TUDAT_NAN )
{
    return std::make_shared< NormalEquationsSolverSettings >( NormalEquationsSolverType::bdc_svd, relativeSingularValueCutoff );
}

//! Function to create settings for a Cholesky solver (dense design matrix)
inline std::shared_ptr< NormalEquationsSolverSettings > choleskySolver( )
{
    return std::make_shared< NormalEquationsSolverSettings >( NormalEquationsSolverType::cholesky );
}

//! Function to create settings for a sparse QR solver (sparse design matrix)
inline std::shared_ptr< NormalEquationsSolverSettings > sparseQrSolver( )
{
    return std::make_shared< NormalEquationsSolverSettings >( NormalEquationsSolverType::sparse_qr );
}

//! Function to create settings for a sparse Cholesky solver (sparse design matrix)
inline std::shared_ptr< NormalEquationsSolverSettings > sparseCholeskySolver( )
{
    return std::make_shared< NormalEquationsSolverSettings >( NormalEquationsSolverType::sparse_cholesky );
}

//! Function to get condition number of matrix from SVD decomposition
template< typename Derived >
typename Eigen::SVDBase< Derived >::RealScalar getConditionNumberOfDecomposedMatrix(
        const Eigen::SVDBase< Derived >& singularValueDecomposition )
{
    const auto& singularValues = singularValueDecomposition.singularValues( );
    return singularValues( 0 ) / singularValues( singularValues.rows( ) - 1 );
}

//! Base class for a solver of the normal equations, with a normal matrix of type M (dense or sparse)
/*!
 * The normal matrix is factorized once, after which the factorization is reused to solve for the parameter correction
 * and to compute the covariance.
 */
template< typename M >
class NormalEquationsSolver
{
public:
    //! Destructor
    virtual ~NormalEquationsSolver( ) = default;

    //! Function to factorize the normal matrix
    /*!
     * Function to factorize the normal matrix, printing a warning if its condition number is too large
     * \param normalMatrix Normal matrix (inverse of the covariance matrix) to factorize
     * \param limitConditionNumberForWarning Condition number above which a warning is printed (NaN for no check)
     */
    virtual void factorize( const M& normalMatrix, const double limitConditionNumberForWarning ) = 0;

    //! Function to solve the factorized normal equations for one or more right-hand sides (as columns)
    virtual Eigen::MatrixXd solve( const Eigen::MatrixXd& rightHandSide ) const = 0;

    //! Function to compute the inverse of the factorized normal matrix (the pseudo-inverse for a truncated SVD)
    virtual Eigen::MatrixXd inverse( ) const
    {
        return solve( Eigen::MatrixXd::Identity( size_, size_ ) );
    }

protected:
    //! Number of rows (and columns) of the factorized normal matrix
    Eigen::Index size_ = 0;
};

//! Solver of the normal equations that uses an Eigen decomposition of type Decomposition
template< typename M, typename Decomposition >
class EigenNormalEquationsSolver : public NormalEquationsSolver< M >
{
public:
    //! Constructor
    /*!
     * Constructor
     * \param solverName Name of the decomposition, used in error messages
     * \param relativeSingularValueCutoff Relative singular value cutoff of a truncated SVD (NaN for no truncation)
     */
    EigenNormalEquationsSolver( const std::string& solverName, const double relativeSingularValueCutoff ):
        solverName_( solverName ), relativeSingularValueCutoff_( relativeSingularValueCutoff )
    {}

    //! Function to factorize the normal matrix
    void factorize( const M& normalMatrix, const double limitConditionNumberForWarning ) override
    {
        this->size_ = normalMatrix.rows( );
        constexpr bool isSvd = std::is_base_of_v< Eigen::SVDBase< Decomposition >, Decomposition >;
        if constexpr( isSvd )
        {
            if( !std::isnan( relativeSingularValueCutoff_ ) )
            {
                decomposition_.setThreshold( relativeSingularValueCutoff_ );
            }
        }
        if constexpr( isDefaultJacobiSvd )
        {
            if( std::isnan( relativeSingularValueCutoff_ ) )
            {
                normalMatrix_ = normalMatrix;
            }
        }

        decomposition_.compute( normalMatrix );
        if( decomposition_.info( ) != Eigen::Success )
        {
            throw std::runtime_error( "Error when solving normal equations, the " + solverName_ + " decomposition failed." );
        }

        // Only the SVD and the dense Cholesky decomposition provide a cheap condition number (the latter as an estimate)
        double conditionNumber = TUDAT_NAN;
        if constexpr( isSvd )
        {
            conditionNumber = getConditionNumberOfDecomposedMatrix( decomposition_ );
        }
        else if constexpr( std::is_same_v< Decomposition, Eigen::LLT< Eigen::MatrixXd > > )
        {
            conditionNumber = 1.0 / decomposition_.rcond( );
        }
        if( conditionNumber > limitConditionNumberForWarning )
        {
            std::cerr << "Warning when performing least squares, condition number is " << conditionNumber << std::endl;
        }
    }

    //! Function to solve the factorized normal equations for one or more right-hand sides (as columns)
    Eigen::MatrixXd solve( const Eigen::MatrixXd& rightHandSide ) const override
    {
        return decomposition_.solve( rightHandSide );
    }

    //! Function to compute the inverse of the factorized normal matrix
    /*!
     * Function to compute the inverse of the factorized normal matrix. An SVD solver inverts all singular values, unless a
     * cutoff is set. A small singular value then gives a large variance, instead of being dropped by the default threshold
     * of the SVD solve. With a cutoff, the singular values below it are dropped (pseudo-inverse). The default Jacobi SVD
     * without cutoff uses an LU inverse of the normal matrix, which keeps the covariance of existing estimations unchanged.
     * \return Inverse of the factorized normal matrix
     */
    Eigen::MatrixXd inverse( ) const override
    {
        if constexpr( isDefaultJacobiSvd )
        {
            if( std::isnan( relativeSingularValueCutoff_ ) )
            {
                return normalMatrix_.inverse( );
            }
        }
        if constexpr( std::is_base_of_v< Eigen::SVDBase< Decomposition >, Decomposition > )
        {
            Eigen::VectorXd inverseSingularValues = decomposition_.singularValues( ).cwiseInverse( );
            if( !std::isnan( relativeSingularValueCutoff_ ) )
            {
                const double singularValueLimit = relativeSingularValueCutoff_ * decomposition_.singularValues( )( 0 );
                inverseSingularValues =
                        ( decomposition_.singularValues( ).array( ) > singularValueLimit ).select( inverseSingularValues, 0.0 );
            }
            return decomposition_.matrixV( ) * inverseSingularValues.asDiagonal( ) * decomposition_.matrixU( ).transpose( );
        }
        else
        {
            return NormalEquationsSolver< M >::inverse( );
        }
    }

private:
    //! Whether this is the (default) dense Jacobi SVD
    static constexpr bool isDefaultJacobiSvd =
            std::is_same_v< Decomposition, Eigen::JacobiSVD< Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV > >;

    //! Normal matrix, stored only for the LU inverse of the default Jacobi SVD without cutoff
    Eigen::MatrixXd normalMatrix_;

    //! Name of the decomposition, used in error messages
    std::string solverName_;

    //! Relative singular value cutoff of a truncated SVD (NaN for no truncation)
    double relativeSingularValueCutoff_;

    //! Eigen decomposition of the normal matrix
    Decomposition decomposition_;
};

//! Function to create a solver of the normal equations, with a normal matrix of type M (dense or sparse)
/*!
 * Function to create a solver of the normal equations. Without settings, the default for the matrix storage is used
 * (Jacobi SVD for dense, sparse QR for sparse matrices).
 * \param solverSettings Settings of the solver (nullptr for the default)
 * \param hasLinearConstraints Whether the normal equations include linear constraints, which make the normal matrix
 * indefinite so that a Cholesky decomposition cannot be used
 * \return Solver of the normal equations
 */
template< typename M >
std::shared_ptr< NormalEquationsSolver< M > > createNormalEquationsSolver(
        const std::shared_ptr< NormalEquationsSolverSettings >& solverSettings,
        const bool hasLinearConstraints = false )
{
    constexpr bool isSparse = std::is_same_v< M, Eigen::SparseMatrix< double > >;
    const NormalEquationsSolverType solverType = ( solverSettings != nullptr ) ? solverSettings->getSolverType( )
            : isSparse                                                         ? NormalEquationsSolverType::sparse_qr
                                                                               : NormalEquationsSolverType::jacobi_svd;
    const double cutoff = ( solverSettings != nullptr ) ? solverSettings->getRelativeSingularValueCutoff( ) : TUDAT_NAN;
    const std::string solverName = getNormalEquationsSolverName( solverType );

    if( hasLinearConstraints &&
        ( solverType == NormalEquationsSolverType::cholesky || solverType == NormalEquationsSolverType::sparse_cholesky ) )
    {
        throw std::runtime_error( "Error when creating normal equations solver, the " + solverName +
                                  " solver cannot be used with linear constraints, use an SVD or sparse QR solver." );
    }

    if constexpr( isSparse )
    {
        switch( solverType )
        {
            case NormalEquationsSolverType::sparse_qr:
                return std::make_shared< EigenNormalEquationsSolver< M, Eigen::SparseQR< M, Eigen::COLAMDOrdering< int > > > >( solverName,
                                                                                                                                cutoff );
            case NormalEquationsSolverType::sparse_cholesky:
                return std::make_shared< EigenNormalEquationsSolver< M, Eigen::SimplicialLDLT< M > > >( solverName, cutoff );
            default:
                throw std::runtime_error( "Error when creating normal equations solver, the " + solverName +
                                          " solver requires a dense design matrix." );
        }
    }
    else
    {
        switch( solverType )
        {
            case NormalEquationsSolverType::jacobi_svd:
                return std::make_shared<
                        EigenNormalEquationsSolver< M, Eigen::JacobiSVD< M, Eigen::ComputeThinU | Eigen::ComputeThinV > > >( solverName,
                                                                                                                             cutoff );
            case NormalEquationsSolverType::bdc_svd:
                return std::make_shared< EigenNormalEquationsSolver< M, Eigen::BDCSVD< M, Eigen::ComputeThinU | Eigen::ComputeThinV > > >(
                        solverName, cutoff );
            case NormalEquationsSolverType::cholesky:
                return std::make_shared< EigenNormalEquationsSolver< M, Eigen::LLT< M > > >( solverName, cutoff );
            default:
                throw std::runtime_error( "Error when creating normal equations solver, the " + solverName +
                                          " solver requires a sparse design matrix." );
        }
    }
}

}  // namespace linear_algebra

}  // namespace tudat

#endif  // TUDAT_NORMALEQUATIONSSOLVER_H

/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include "expose_variance_component_estimation.h"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "scalarTypes.h"
#include "tudat/simulation/estimation_setup/varianceComponentEstimation.h"

namespace py = pybind11;
namespace tss = tudat::simulation_setup;
namespace tom = tudat::observation_models;

namespace tudatpy
{
namespace estimation
{
namespace estimation_analysis
{

void expose_variance_component_estimation( py::module& m )
{
    using Settings = tss::VarianceComponentSettings< STATE_SCALAR_TYPE, TIME_TYPE >;
    using Output = tss::VarianceComponentEstimationOutput< STATE_SCALAR_TYPE, TIME_TYPE >;
    using Condition = tom::ObservationSelectionCondition< STATE_SCALAR_TYPE, TIME_TYPE >;

    py::enum_< tss::VarianceComponentEstimatorType >( m, "VarianceComponentEstimatorType", R"doc(

         Methods to estimate the variance factor of each group of observations.

      )doc" )
            .value( "helmert_estimator",
                    tss::VarianceComponentEstimatorType::helmert_estimator,
                    "Helmert's equations, which couple the groups (Sahin et al., 1992, Eq. 19). They can give factors that are "
                    "not positive when the starting weights are far off." )
            .value( "simplified_helmert_estimator",
                    tss::VarianceComponentEstimatorType::simplified_helmert_estimator,
                    "Weighted square sum of each group divided by its redundancy (Foerstner, 1979). Its factors are always positive." )
            .export_values( );

    py::class_< Settings, std::shared_ptr< Settings > >( m, "VarianceComponentSettings", R"doc(

         Settings for estimating the weight of each group of observations, created with :func:`variance_component_settings`.

      )doc" );

    m.def( "variance_component_settings",
           &tss::varianceComponentSettings< STATE_SCALAR_TYPE, TIME_TYPE >,
           py::arg( "groups" ),
           py::arg( "tolerance" ) = 0.01,
           py::arg( "maximum_iterations" ) = 10,
           py::arg( "estimator_type" ) = tss::VarianceComponentEstimatorType::simplified_helmert_estimator,
           py::arg( "least_squares_iterations_per_update" ) = 3,
           R"doc(

         Function to create settings for estimating the weight of each group of observations with Helmert's method
         (`Sahin et al., 1992 <https://doi.org/10.1007/BF02033189>`_).

         Parameters
         ----------
         groups : list[ObservationSelectionCondition]
             One condition per group, for instance an observable type combined with the time bounds of an arc. Groups must
             not overlap. Weight correlations are only allowed within a group.
         tolerance : float, default = 0.01
             The iteration stops when all variance factors differ from 1 by less than this value.
         maximum_iterations : int, default = 10
             Maximum number of least-squares estimations.
         estimator_type : :class:`VarianceComponentEstimatorType`, default = ``simplified_helmert_estimator``
             Method used to compute the variance factors.
         least_squares_iterations_per_update : int, default = 3
             Number of least-squares iterations between two weight updates. Only the final estimation runs to convergence.
             With 0, each estimation runs to convergence.

         Returns
         -------
         :class:`VarianceComponentSettings`
             Settings for :meth:`Estimator.perform_estimation_with_variance_components`.

      )doc" );

    py::class_< tss::VarianceComponentGroupResult >( m, "VarianceComponentGroupResult", R"doc(

         Variance component results for one group of observations.

      )doc" )
            .def_readonly( "number_of_observations",
                           &tss::VarianceComponentGroupResult::numberOfObservations,
                           "Number of observations with a non-zero weight." )
            .def_readonly( "redundancy",
                           &tss::VarianceComponentGroupResult::redundancy,
                           "Number of observations minus the share used up by the estimated parameters." )
            .def_readonly(
                    "weighted_square_sum", &tss::VarianceComponentGroupResult::weightedSquareSum, "Weighted sum of squared residuals." )
            .def_readonly( "variance_factor",
                           &tss::VarianceComponentGroupResult::varianceFactor,
                           "Estimated variance factor. The group weights are divided by this value." )
            .def_readonly( "rms_residual", &tss::VarianceComponentGroupResult::rmsResidual, "Root mean square of the residuals." )
            .def_readonly( "is_frozen",
                           &tss::VarianceComponentGroupResult::isFrozen,
                           "Whether the group kept a factor of 1 because its redundancy is below 1." );

    py::class_< tss::VarianceComponentEstimate >( m, "VarianceComponentEstimate", R"doc(

         Variance component results for all groups, from one least-squares solution.

      )doc" )
            .def_readonly(
                    "group_results", &tss::VarianceComponentEstimate::groupResults, "Results per group, in the order of the groups." );

    py::class_< Output, std::shared_ptr< Output > >( m, "VarianceComponentEstimationOutput", R"doc(

         Output of :meth:`Estimator.perform_estimation_with_variance_components`.

      )doc" )
            .def_readonly( "estimation_output", &Output::estimationOutput, "Output of the last least-squares estimation." )
            .def_readonly( "iteration_results", &Output::iterationResults, "Variance component results of each iteration." )
            .def_readonly( "total_variance_factor_history",
                           &Output::totalVarianceFactorHistory,
                           "Total scaling of the group variances used in each iteration." )
            .def_readonly( "least_squares_best_iterations",
                           &Output::leastSquaresBestIterations,
                           "Best iteration of each least-squares estimation." )
            .def_readonly( "is_converged", &Output::isConverged, "Whether all variance factors converged to 1." );

    m.def(
            "compute_variance_components",
            []( const std::shared_ptr< tss::EstimationOutput< STATE_SCALAR_TYPE, TIME_TYPE > > estimationOutput,
                const std::shared_ptr< tom::ObservationDataset< STATE_SCALAR_TYPE, TIME_TYPE > > observationDataset,
                const std::vector< Condition >& groups,
                const tss::VarianceComponentEstimatorType estimatorType ) {
                return tss::computeHelmertVarianceComponents< STATE_SCALAR_TYPE, TIME_TYPE >(
                        estimationOutput, observationDataset, groups, estimatorType );
            },
            py::arg( "estimation_output" ),
            py::arg( "observation_dataset" ),
            py::arg( "groups" ),
            py::arg( "estimator_type" ) = tss::VarianceComponentEstimatorType::simplified_helmert_estimator,
            R"doc(

         Function to estimate the variance factor of each group of observations from one estimation, without changing any weights.

         Parameters
         ----------
         estimation_output : :class:`EstimationOutput`
             Output of the estimation, with the design matrix and the iteration history saved.
         observation_dataset : :class:`~tudatpy.estimation.observations.ObservationDataset`
             Observations, with the same weights as in the estimation.
         groups : list[ObservationSelectionCondition]
             One condition per group.
         estimator_type : :class:`VarianceComponentEstimatorType`, default = ``simplified_helmert_estimator``
             Method used to compute the variance factors.

         Returns
         -------
         :class:`VarianceComponentEstimate`
             Variance factor and related values per group.

      )doc" );
}

}  // namespace estimation_analysis
}  // namespace estimation
}  // namespace tudatpy

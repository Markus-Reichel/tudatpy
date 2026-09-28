/*    Copyright (c) 2010-2026, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */
#if TUDATPY_ENABLE_DETAILED_PYBIND11_ERRORS
#define PYBIND11_DETAILED_ERROR_MESSAGES
#endif
#include "expose_observations_wrapper_bindings.h"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "scalarTypes.h"

#include "tudat/io/readSumLmkFiles.h"
#include "tudat/simulation/estimation_setup/processSumLmkFiles.h"

namespace tom = tudat::observation_models;
namespace tss = tudat::simulation_setup;
namespace tio = tudat::input_output::sum_lmk;

namespace tudatpy
{
namespace estimation
{
namespace observations_setup
{
namespace observations_wrapper
{

void expose_observations_wrapper_sum_lmk_bindings( py::module& m )
{
    py::class_< tio::SumImageData >( m, "SumImageData", R"doc(
        Parsed contents of a single SPC SUM file (image metadata + landmark/limb pixel rows). Opaque
        handle passed back into ``create_sum_lmk_observation_collection`` to convert without re-reading.
        )doc" )
            .def_readonly( "image_id", &tio::SumImageData::imageId_ )
            .def_readonly( "utc_epoch_string", &tio::SumImageData::utcEpochString_ )
            .def_readonly( "source_file", &tio::SumImageData::sourceFile_ );

    py::class_< tio::LmkLandmarkData >( m, "LmkLandmarkData", R"doc(
        Parsed contents of a single SPC LMK landmark file. Opaque handle passed back into
        ``create_sum_lmk_observation_collection`` to convert without re-reading.
        )doc" )
            .def_readonly( "landmark_id", &tio::LmkLandmarkData::landmarkId_ )
            .def_readonly( "body_fixed_position",
                           &tio::LmkLandmarkData::bodyFixedPosition_,
                           R"doc(Landmark position in the target body-fixed frame, in m (LMK VLM record).)doc" )
            .def_readonly( "landmark_position_sigma",
                           &tio::LmkLandmarkData::landmarkPositionSigma_,
                           R"doc(Per-component 1-sigma uncertainty on the body-fixed position, in m (LMK SIGMA_LMK record).)doc" )
            .def_readonly( "local_x_axis",
                           &tio::LmkLandmarkData::localXAxis_,
                           R"doc(Landmark local frame X axis, as a unit vector in the body-fixed frame (LMK UX record).)doc" )
            .def_readonly( "local_y_axis",
                           &tio::LmkLandmarkData::localYAxis_,
                           R"doc(Landmark local frame Y axis, as a unit vector in the body-fixed frame (LMK UY record).)doc" )
            .def_readonly(
                    "local_z_axis",
                    &tio::LmkLandmarkData::localZAxis_,
                    R"doc(Landmark local frame Z axis - the maplet plane normal - as a unit vector in the body-fixed frame (LMK UZ record).)doc" )
            .def_readonly( "source_file", &tio::LmkLandmarkData::sourceFile_ );

    py::class_< tom::SumLmkObservationConversionSettings >( m, "SumLmkObservationConversionSettings", R"doc(
        Settings for converting SPC-style SUM/LMK optical-landmark files to Tudat pixel-coordinate observations.
        )doc" )
            .def( py::init< const std::string&, const std::string& >( ), py::arg( "target_body_name" ), py::arg( "receiver_body_name" ) )
            .def_readwrite( "target_body_name", &tom::SumLmkObservationConversionSettings::targetBodyName_ )
            .def_readwrite( "receiver_body_name", &tom::SumLmkObservationConversionSettings::receiverBodyName_ )
            .def_readwrite( "body_fixed_camera_position", &tom::SumLmkObservationConversionSettings::bodyFixedCameraPosition_ )
            .def_readwrite( "validate_spacecraft_object_geometry",
                            &tom::SumLmkObservationConversionSettings::validateSpacecraftObjectGeometry_ )
            .def_readwrite( "skip_observations_with_missing_landmarks",
                            &tom::SumLmkObservationConversionSettings::skipObservationsWithMissingLandmarks_ );

    py::class_< tom::SumLmkObservationConversionResult< STATE_SCALAR_TYPE, TIME_TYPE > >( m, "SumLmkObservationConversionResult", R"doc(
        Result of a SUM/LMK observation conversion: the observation collection, matching observation
        model settings, per-image pointing a-priori inverse covariance entries, and image-to-camera map.
        )doc" )
            .def_readonly( "observation_collection",
                           &tom::SumLmkObservationConversionResult< STATE_SCALAR_TYPE, TIME_TYPE >::observationCollection_,
                           R"doc(Pixel-coordinate observation collection (observed values), keyed by (image, landmark) link ends.)doc" )
            .def_readonly( "observation_model_settings",
                           &tom::SumLmkObservationConversionResult< STATE_SCALAR_TYPE, TIME_TYPE >::observationModelSettings_,
                           R"doc(Observation model settings matching the collection (one pixel_coordinates setting per link end).)doc" )
            .def_readonly(
                    "inverse_apriori_covariance_diagonal_entries",
                    &tom::SumLmkObservationConversionResult< STATE_SCALAR_TYPE, TIME_TYPE >::inverseAprioriCovarianceDiagonalEntries_,
                    R"doc(Per-image camera_pointing_correction inverse a-priori covariance diagonal entries derived from SIGMA_PTG.)doc" )
            .def_readonly( "receiver_body_name",
                           &tom::SumLmkObservationConversionResult< STATE_SCALAR_TYPE, TIME_TYPE >::receiverBodyName_,
                           R"doc(Name of the body carrying the per-image cameras, i.e. the body the pointing parameters belong to.)doc" )
            .def_readonly( "image_id_to_camera_name",
                           &tom::SumLmkObservationConversionResult< STATE_SCALAR_TYPE, TIME_TYPE >::imageIdToCameraName_,
                           R"doc(Map from SUM image ID to the registered Camera_<imageId> reference-point name.)doc" )
            .def_readonly( "target_body_name",
                           &tom::SumLmkObservationConversionResult< STATE_SCALAR_TYPE, TIME_TYPE >::targetBodyName_,
                           R"doc(Name of the body carrying the landmarks, i.e. the body the landmark position parameters belong to.)doc" )
            .def_readonly( "landmarks",
                           &tom::SumLmkObservationConversionResult< STATE_SCALAR_TYPE, TIME_TYPE >::landmarks_,
                           R"doc(The landmarks actually referenced by the converted images, keyed by landmark ID.)doc" );

    m.def( "create_sum_lmk_observation_collection",
           py::overload_cast< const std::vector< std::string >&,
                              const std::vector< std::string >&,
                              const tss::SystemOfBodies&,
                              const tom::SumLmkObservationConversionSettings& >(
                   &tom::createSumLmkObservationCollection< STATE_SCALAR_TYPE, TIME_TYPE > ),
           py::arg( "sum_files" ),
           py::arg( "lmk_files" ),
           py::arg( "bodies" ),
           py::arg( "conversion_settings" ),
           R"doc(
        Create a pixel-coordinate observation collection from SPC-style SUM/LMK files.

        Reads the given SUM and LMK files, registers one ``Camera_<imageId>`` per image on the receiver
        body and each landmark as a body-fixed ground station on the target body (both added to ``bodies``
        in place), and returns the converted observations together with their matching observation model
        settings and the per-image SIGMA_PTG pointing a-priori. Each (image, landmark) pair becomes one
        observation; all landmarks of an image share the same camera (and thus the same pointing parameter).
        )doc" );

    m.def( "read_sum_files",
           &tio::readSumFiles,
           py::arg( "sum_files" ),
           R"doc(
        Parse SPC SUM files into opaque ``SumImageData`` handles without touching any environment.
        The result can be passed to ``create_sum_lmk_observation_collection`` (parsed-data overload)
        for repeated conversions against different body systems without re-reading the files.
        )doc" );

    m.def( "read_lmk_files",
           &tio::readLmkFiles,
           py::arg( "lmk_files" ),
           R"doc(
        Parse SPC LMK landmark files into a ``{landmark_id: LmkLandmarkData}`` map without touching any
        environment, for reuse across repeated SUM/LMK conversions.
        )doc" );

    m.def( "create_sum_lmk_observation_collection",
           py::overload_cast< const std::vector< tio::SumImageData >&,
                              const std::map< std::string, tio::LmkLandmarkData >&,
                              const tss::SystemOfBodies&,
                              const tom::SumLmkObservationConversionSettings& >(
                   &tom::createSumLmkObservationCollection< STATE_SCALAR_TYPE, TIME_TYPE > ),
           py::arg( "sum_images" ),
           py::arg( "landmarks" ),
           py::arg( "bodies" ),
           py::arg( "conversion_settings" ),
           R"doc(
        Create a pixel-coordinate observation collection from already-parsed SUM/LMK data (as returned by
        ``read_sum_files`` / ``read_lmk_files``). Identical to the file-path overload but skips disk I/O,
        so the same parsed data can be converted against multiple body systems. Registers one
        ``Camera_<imageId>`` per image on the receiver body and each landmark as a body-fixed ground
        station on the target body (both added to ``bodies`` in place).
        )doc" );

    m.def( "create_sum_lmk_observation_model_settings",
           &tom::createSumLmkObservationModelSettings< STATE_SCALAR_TYPE, TIME_TYPE >,
           py::arg( "observation_collection" ),
           py::arg( "light_time_corrections" ) = std::vector< std::shared_ptr< tom::LightTimeCorrectionSettings > >( ),
           R"doc(
        Build pixel-coordinate observation model settings matching every (image, landmark) link end in a
        SUM/LMK observation collection (light-time geometric single-leg on, stellar aberration off).
        )doc" );

    m.def( "create_sum_lmk_pointing_parameter_settings",
           &tom::createSumLmkPointingParameterSettings< STATE_SCALAR_TYPE, TIME_TYPE >,
           py::arg( "conversion_result" ),
           py::arg( "image_ids" ) = std::vector< std::string >( ),
           R"doc(
        Create the ``camera_pointing_correction`` parameter settings for the per-image cameras registered by a
        SUM/LMK conversion: one 3-vector pointing parameter per image, on the conversion's receiver body. The
        settings are ordered by camera name, so the resulting parameter vector has a reproducible layout. Pass
        ``image_ids`` to restrict the settings to a subset of the images; an unknown image ID raises.
        )doc" );

    m.def( "create_sum_lmk_landmark_parameter_settings",
           &tom::createSumLmkLandmarkParameterSettings< STATE_SCALAR_TYPE, TIME_TYPE >,
           py::arg( "conversion_result" ),
           py::arg( "observation_collection" ) = std::shared_ptr< tom::ObservationCollection< STATE_SCALAR_TYPE, TIME_TYPE > >( ),
           py::arg( "landmark_ids" ) = std::vector< std::string >( ),
           R"doc(
        Create the ``ground_station_position`` parameter settings for the landmarks of a SUM/LMK conversion:
        one 3-vector body-fixed position per landmark, on the conversion's target body. Landmarks are
        registered as body-fixed ground stations by the conversion, so this is an ordinary global (not
        arc-wise) parameter: a landmark observed from several arcs becomes one shared parameter, tying those
        arcs together.

        Only landmarks that the estimated observations actually observe get a parameter. By default that is
        judged from the conversion's own collection; pass ``observation_collection`` to judge it from a
        filtered collection instead, which is what you want when images or observations were removed before
        the estimation. The settings are ordered by landmark ID, so the parameter vector has a reproducible
        layout. Pass ``landmark_ids`` to restrict the settings further; an ID that is unknown, or whose
        observations have all been filtered away, raises.
        )doc" );

    m.def( "get_sum_lmk_landmark_local_frame_uncertainties",
           &tom::getSumLmkLandmarkLocalFrameUncertainties< STATE_SCALAR_TYPE, TIME_TYPE, STATE_SCALAR_TYPE >,
           py::arg( "conversion_result" ),
           py::arg( "parameters_to_estimate" ),
           py::arg( "covariance" ),
           R"doc(
        Express the formal uncertainties of the estimated landmark positions in each landmark's own local
        frame, i.e. along its LMK UX/UY/UZ axes, with UZ the maplet plane normal. Returns a map from landmark
        ID to a 3-vector of sigmas.

        Landmark positions are estimated in body-fixed coordinates (the frame SIGMA_LMK is also given in), so
        this is a rotation of the covariance block, ``sqrt(diag(U^T P U))``, rather than a different
        estimation setup. Landmarks that are not part of the estimated parameter set are omitted.
        )doc" );

    m.def( "create_sum_lmk_inverse_apriori_covariance",
           &tom::createSumLmkInverseAprioriCovariance< STATE_SCALAR_TYPE, TIME_TYPE, STATE_SCALAR_TYPE >,
           py::arg( "conversion_result" ),
           py::arg( "parameters_to_estimate" ),
           py::arg( "base_inverse_apriori_covariance" ) = Eigen::MatrixXd::Zero( 0, 0 ),
           R"doc(
        Assemble an inverse a-priori covariance matrix for an ``EstimationInput`` from the per-image SIGMA_PTG
        pointing a-priori carried by a SUM/LMK conversion result. Each entry is written onto the diagonal of its
        parameter's block in the full parameter vector; entries for parameters that are not being estimated are
        skipped, since a conversion produces an a-priori for every image while only a subset may be estimated.
        Pass ``base_inverse_apriori_covariance`` to add these entries on top of an existing a-priori (for example
        one already constraining the initial state).
        )doc" );

    m.def( "compute_sum_lmk_residuals",
           &tom::computeSumLmkResiduals< STATE_SCALAR_TYPE, TIME_TYPE >,
           py::arg( "observation_collection" ),
           py::arg( "bodies" ),
           py::arg( "light_time_corrections" ) = std::vector< std::shared_ptr< tom::LightTimeCorrectionSettings > >( ),
           R"doc(
        Compute observed-minus-computed (O-C) pixel residuals for a SUM/LMK observation collection given a
        fixed environment (e.g. a SPICE spacecraft trajectory). Residuals are stored in the collection
        (accessible via ``get_concatenated_residuals`` / ``get_rms_residuals``) and returned concatenated.
        )doc" );
}

}  // namespace observations_wrapper
}  // namespace observations_setup
}  // namespace estimation
}  // namespace tudatpy

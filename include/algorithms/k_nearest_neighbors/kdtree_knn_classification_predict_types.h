/* file: kdtree_knn_classification_predict_types.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of the K-Nearest Neighbors (kNN) algorithm interface
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_PREDICT_TYPES_H__
#define __KDTREE_KNN_CLASSIFICATION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/data_serialize.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{

/**
 * \brief Contains classes of the KD-tree based kNN algorithm
 */
namespace kdtree_knn_classification
{

/**
 * @defgroup kdtree_knn_classification_prediction Prediction
 * \copydoc daal::algorithms::kdtree_knn_classification::prediction
 * @ingroup kdtree_knn_classification
 * @{
 */
/**
 * \brief Contains a class for making KD-tree based kNN model-based prediction
 */
namespace prediction
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__METHOD"></a>
 * \brief Available methods for making KD-tree based kNN model-based prediction
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making KD-tree based kNN model-based prediction
 */
class DAAL_EXPORT Input : public classifier::prediction::Input
{
    typedef classifier::prediction::Input super;
public:
    /** Default constructor */
    Input();

    using super::get;
    using super::set;

    /**
     * Returns the input Model object in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    kdtree_knn_classification::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Sets the input Model object in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id      Identifier of the input object
     * \param[in] value   Input Model object
     */
    void set(classifier::prediction::ModelInputId id, const kdtree_knn_classification::interface1::ModelPtr & value);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

} // namespace interface1

using interface1::Input;

} // namespace prediction
/** @} */
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif

/* file: bf_knn_classification_model.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of the class defining the K-Nearest Neighbors (kNN) classification model
//--
*/

#ifndef __BF_KNN_CLASSIFICATION_MODEL_H__
#define __BF_KNN_CLASSIFICATION_MODEL_H__

#include "algorithms/classifier/classifier_model.h"
#include "data_management/data/aos_numeric_table.h"
#include "data_management/data/soa_numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/engines/mcg59/mcg59.h"

namespace daal
{
namespace algorithms
{

/**
 * @defgroup bf_knn_classification k-Nearest Neighbors
 * \copydoc daal::algorithms::bf_knn_classification
 * @ingroup classification
 * @{
 */

/**
 * \brief Contains classes for BF kNN algorithm
 */
namespace bf_knn_classification
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__BF_KNN_CLASSIFICATION__DATAUSEINMODEL"></a>
 * \brief The option to enable/disable an usage of the input dataset in kNN model
 */
enum DataUseInModel
{
    doNotUse = 0, /*!< The input data and labels will not be the component of the trained kNN model */
    doUse    = 1  /*!< The input data and labels will be the component of the trained kNN model */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__BF_KNN_CLASSIFICATION__PARAMETER"></a>
 * \brief BF kNN algorithm parameters
 *
 * \snippet k_nearest_neighbors/bf_knn_classification_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::classifier::Parameter
{
    /**
     *  Parameter constructor
     *  \param[in] nClasses             Number of classes
     *  \param[in] nNeighbors           Number of neighbors
     *  \param[in] dataUse              The option to enable/disable an usage of the input dataset in kNN model
     */
    Parameter(size_t nClasses = 2, size_t nNeighbors = 1, DataUseInModel dataUse = doNotUse)
        : daal::algorithms::classifier::Parameter(nClasses),
          k(nNeighbors),
          dataUseInModel(dataUse),
          engine(engines::mcg59::Batch<>::create())
    {}

    /**
     *  Parameter copy constructor
     *  \param[in] other             Object to copy
     */
    Parameter(const Parameter& other)
        : daal::algorithms::classifier::Parameter(other.nClasses),
          k(other.k),
          dataUseInModel(other.dataUseInModel),
          engine(other.engine->clone())
    {}

    /**
     * Checks a parameter of the BF kNN algorithm
     */
    services::Status check() const DAAL_C11_OVERRIDE;

    size_t k;                      /*!< Number of neighbors */
    DataUseInModel dataUseInModel; /*!< The option to enable/disable an usage of the input dataset in kNN model */
    engines::EnginePtr engine;     /*!< Engine for random choosing elements from training dataset */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__MODEL"></a>
 * \brief %Base class for models trained with the BF kNN algorithm
 *
 * \tparam modelFPType  Data type to store BF kNN model data, double or float
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::classifier::Model
{
public:
    DECLARE_MODEL_IFACE(Model, classifier::Model);
    /**
     * Constructs the model trained with the BF kNN algorithm
     * \param[in] nFeatures Number of features in the dataset
     */
    Model(size_t nFeatures = 0);

    virtual ~Model();

    class ModelImpl;

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    const ModelImpl *impl() const { return _impl; }

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    ModelImpl *impl() { return _impl; }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

protected:
    Model(size_t nFeatures, services::Status &st);

    services::Status serializeImpl(data_management::InputDataArchive   *arch) DAAL_C11_OVERRIDE;

    services::Status deserializeImpl(const data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE;

private:
    ModelImpl *_impl;  /*!< Model implementation */
};
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1

using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;

} // namespace bf_knn_classification

/** @} */
} // namespace algorithms
} // namespace daal

#endif

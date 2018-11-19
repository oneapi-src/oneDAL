/* file: neural_networks_prediction_result.h */
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
//  Implementation of neural network algorithm interface.
//--
*/

#ifndef __NEURAL_NETWORKS_PREDICTION_RESULT_H__
#define __NEURAL_NETWORKS_PREDICTION_RESULT_H__

#include "algorithms/algorithm.h"

#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "services/collection.h"
#include "neural_networks_prediction_input.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for prediction and prediction using neural network
 */
namespace neural_networks
{
namespace prediction
{
/**
 * @ingroup neural_networks_prediction
 * @{
 */
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__RESULTID"></a>
 * Available identifiers of results obtained in the prediction stage of the neural network algorithm
 */
enum ResultId
{
    prediction,        /*!< Prediction results */
    lastResultId = prediction
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__RESULTID"></a>
 * Available identifiers of results obtained in the prediction stage of the neural network algorithm
 */
enum ResultCollectionId
{
    predictionCollection = lastResultId + 1,       /*!< Prediction results */
    lastResultCollectionId = predictionCollection
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__RESULT"></a>
 * \brief Provides methods to access result obtained with the compute() method of the neural networks prediction algorithm
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);

    Result();

    /**
     * Returns the result of the neural networks model based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(ResultId id) const;

    /**
     * Returns the result of the neural networks model based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(ResultCollectionId id) const;

    /**
     * Returns the result of the neural networks model based prediction
     * \param[in] id    Identifier of the result
     * \param[in] key   Index of the tensor with partial results in the key-value data collection
     * \return          Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(ResultCollectionId id, size_t key) const;

    /**
     * Sets the result of neural networks model based prediction
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultId id, const data_management::TensorPtr &value);

    /**
     * Sets the result of neural networks model based prediction
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultCollectionId id, const data_management::KeyValueDataCollectionPtr &value);

    /**
     * Add the value to the key-value data collection of partial results
     * \param[in] id    Identifier of the result
     * \param[in] key   Key to use to retrieve data
     * \param[in] value Result
     */
    void add(ResultCollectionId id, size_t key, const data_management::TensorPtr &value);

    /**
     * Registers user-allocated memory to store partial results of the neural networks model based prediction
     * \param[in] input Pointer to an object containing %input data
     * \param[in] method Computation method for the algorithm
     * \param[in] parameter %Parameter of the neural networks prediction
     *
     * \return Status of computations
     */
    template<typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks result of the neural networks algorithm
     * \param[in] input   %Input object of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1
using interface1::Result;
using interface1::ResultPtr;

/** @} */
}
}
}
} // namespace daal
#endif

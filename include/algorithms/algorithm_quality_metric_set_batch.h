/* file: algorithm_quality_metric_set_batch.h */
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
//  Interface for the quality metric set in the batch processing mode.
//--
*/

#ifndef __ALGORITHM_QUALITY_METRIC_SET_BATCH_H__
#define __ALGORITHM_QUALITY_METRIC_SET_BATCH_H__

#include "algorithms/algorithm_quality_metric_set_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes to compute a quality metric set
 */
namespace quality_metric_set
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @addtogroup base_algorithms
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC_SET__BATCH"></a>
 * \brief Provides methods to compute a quality metric set of an algorithm in the batch processing mode.
 */
class Batch
{
public:
    InputAlgorithmsCollection                inputAlgorithms;      /*!< Collection of quality metrics algorithms */

    Batch(bool useDefaultMetrics = true) : _useDefaultMetrics(useDefaultMetrics)
    {}

    virtual ~Batch() {}
    /**
     * Returns the structure that contains a computed quality metric set
     * \return Structure that contains a computed quality metric set
     */
    ResultCollectionPtr getResultCollection() { return _resultCollection; }

    /**
     * Returns the collection of input objects for the quality metrics algorithm
     * \return The collection of input objects for the quality metrics algorithm
     */
    InputDataCollectionPtr getInputDataCollection() { return _inputData; }

    /**
     * Computes results for a quality metric set in the batch processing mode.
     *
     * \return Status of computations
     */
    services::Status compute()
    {
        this->_status = computeNoThrow();
        return services::throwIfPossible(this->_status);
    }

    /**
     * Computes results for a quality metric set in the batch processing mode.
     *
     * \return Status of computations
     */
    services::Status computeNoThrow()
    {
        DAAL_CHECK(inputAlgorithms.size(), services::ErrorEmptyInputAlgorithmsCollection)
        services::Status s;
        for (size_t i = 0; i < inputAlgorithms.size(); i++)
        {
            size_t key = inputAlgorithms.getKeyByIndex((int)i);
            inputAlgorithms[key]->setInput(_inputData->getInput(key).get());
            s = inputAlgorithms[key]->computeNoThrow();
            if (!s) break;
            _resultCollection->add(key, inputAlgorithms[key]->getResult());
        }
        return s;
    }

    /**
     * Returns errors that happened during the computations
     * \return Errors that happened during the computations
     */
    services::SharedPtr<services::ErrorCollection> getErrors()
    {
        return _status.getCollection();
    }

protected:
    virtual void initializeQualityMetrics() = 0;

    bool _useDefaultMetrics;
    InputDataCollectionPtr _inputData;            /*!< Collection of input objects of quality metrics algorithms */
    ResultCollectionPtr    _resultCollection;
    services::Status _status;
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace daal::algorithms::quality_metric_set
} // namespace daal::algorithms
} // namespace daal
#endif

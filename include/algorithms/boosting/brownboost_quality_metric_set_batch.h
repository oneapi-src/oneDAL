/* file: brownboost_quality_metric_set_batch.h */
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
//  Interface for the BrownBoost quality metric set.
//--
*/

#ifndef __BROWNBOOST_QUALITY_METRIC_SET_BATCH_H__
#define __BROWNBOOST_QUALITY_METRIC_SET_BATCH_H__

#include "algorithms/algorithm_quality_metric_set_batch.h"
#include "algorithms/classifier/binary_confusion_matrix_batch.h"
#include "algorithms/boosting/brownboost_quality_metric_set_types.h"

namespace daal
{
namespace algorithms
{
namespace brownboost
{
/**
 * \brief Contains classes for checking the quality of the model trained with the BrownBoost algorithm
 */
namespace quality_metric_set
{

namespace interface1
{
/**
 * @defgroup brownboost_quality_metric_set_batch Batch
 * @ingroup brownboost_quality_metric_set
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BROWNBOOST__QUALITY_METRIC_SET__BATCH"></a>
 * \brief Class that represents a set of quality metrics to check the model trained with the BrownBoost algorithm
 *
 * \par Enumerations
 *      - \ref QualityMetricId  Quality metrics provided by the library
 *
 * \par References
 *      - \ref algorithms::quality_metric_set::interface1::InputAlgorithmsCollection "algorithms::quality_metric_set::InputAlgorithmsCollection" class
 */
class Batch : public algorithms::quality_metric_set::Batch
{
public:
    /**
     * Constructs a quality metric set for the model trained with the BrownBoost algorithm
     * \param[in] useDefaultMetrics     Flag. If true, a quality metric set is initialized with the quality metrics provided by the library
     */
    Batch(bool useDefaultMetrics = true) : algorithms::quality_metric_set::Batch(useDefaultMetrics)
    {
        _inputData = InputDataCollectionPtr(new InputDataCollection());
        if (_useDefaultMetrics)
        {
            initializeQualityMetrics();
        }
        _resultCollection = ResultCollectionPtr(new ResultCollection());
    }

    virtual ~Batch() {}

    /**
     * Returns the structure that contains a computed quality metric set
     * \return Structure that contains a computed quality metric set
     */
    ResultCollectionPtr getResultCollection()
    {
        return services::staticPointerCast<ResultCollection,
                                           algorithms::quality_metric_set::ResultCollection>(_resultCollection);
    }

    /**
     * Returns the collection of input objects of the quality metrics algorithm
     * \return Collection of input objects of the quality metrics algorithm
     */
    InputDataCollectionPtr getInputDataCollection()
    {
        return services::staticPointerCast<InputDataCollection,
                                           algorithms::quality_metric_set::InputDataCollection>(_inputData);
    }

protected:
    virtual void initializeQualityMetrics()
    {
        inputAlgorithms[confusionMatrix] = services::SharedPtr<classifier::quality_metric::binary_confusion_matrix::Batch<> >(
                new classifier::quality_metric::binary_confusion_matrix::Batch<>());
        _inputData->add(confusionMatrix, algorithms::InputPtr(
                new classifier::quality_metric::binary_confusion_matrix::Input));
    }
};
/** @} */
} // namespace interface1
using interface1::Batch;

}
}
}
}
#endif

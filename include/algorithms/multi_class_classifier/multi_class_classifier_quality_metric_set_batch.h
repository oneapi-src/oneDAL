/* file: multi_class_classifier_quality_metric_set_batch.h */
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
//  Interface for the multi-class classifier quality metric set.
//--
*/

#ifndef __MULTI_CLASS_CLASSIFIER_QUALITY_METRIC_SET_BATCH_H__
#define __MULTI_CLASS_CLASSIFIER_QUALITY_METRIC_SET_BATCH_H__

#include "algorithms/algorithm_quality_metric_set_batch.h"
#include "algorithms/classifier/multiclass_confusion_matrix_batch.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_quality_metric_set_types.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
/**
 * \brief Contains classes for checking the quality of the model trained with the multi-class classifier algorithm
 */
namespace quality_metric_set
{

namespace interface1
{
/**
 * @defgroup multi_class_classifier_quality_metric_set_batch Batch
 * @ingroup multi_class_classifier_quality_metric_set
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__QUALITY_METRIC_SET__BATCH"></a>
 * \brief Class containing a set of quality metrics to check the model trained with the multi-class classifier algorithm
 *
 * \par Enumerations
 *      - \ref QualityMetricId  Identifiers of quality metrics provided by the library
 *
 * \par References
 *      - \ref algorithms::quality_metric_set::interface1::InputAlgorithmsCollection "algorithms::quality_metric_set::InputAlgorithmsCollection" class
 */
class Batch : public algorithms::quality_metric_set::Batch
{
public:
    Parameter parameter;    /*!< Parameters of the algorithm */
    /**
     * Constructs a quality metric set for the model trained with the multi-class classifier algorithm
     * \param[in] nClasses          Number of classes
     * \param[in] useDefaultMetrics Flag. If true, a quality metric set is initialized with the quality metrics provided by the library
     */
    Batch(size_t nClasses = 2, bool useDefaultMetrics = true) :
        algorithms::quality_metric_set::Batch(useDefaultMetrics),
        parameter(nClasses)
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
        inputAlgorithms[confusionMatrix] = services::SharedPtr<classifier::quality_metric::multiclass_confusion_matrix::Batch<> >(
                                               new classifier::quality_metric::multiclass_confusion_matrix::Batch<>(parameter.nClasses));
        _inputData->add(confusionMatrix, algorithms::InputPtr(
                          new classifier::quality_metric::multiclass_confusion_matrix::Input));
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

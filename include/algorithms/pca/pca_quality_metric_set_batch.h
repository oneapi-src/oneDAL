/* file: pca_quality_metric_set_batch.h */
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
//  Interface for the pca quality metric batch.
//--
*/

#ifndef __PCA_QUALITY_METRIC_SET_BATCH_H__
#define __PCA_QUALITY_METRIC_SET_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/algorithm_quality_metric_set_batch.h"
#include "algorithms/pca/pca_quality_metric_set_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
/**
 * \brief Contains classes to check the quality of the pca algorithm
 */
namespace quality_metric_set
{

namespace interface1
{
/**
 * @defgroup pca_quality_metric_set_batch Batch
 * @ingroup pca_quality_metric_set
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC_SET__BATCH"></a>
 * \brief Class that represents a quality metric set of the pca algorithm
 *
 * \par Enumerations
 *      - \ref QualityMetricId  Identifiers of quality metrics provided by the library
 *
 * \par References
 *      - \ref algorithms::quality_metric_set::interface1::InputAlgorithmsCollection "algorithms::quality_metric_set::InputAlgorithmsCollection" class
 */
class DAAL_EXPORT Batch : public algorithms::quality_metric_set::Batch
{
public:
    Parameter parameter;    /*!< Parameters of the algorithm */

    /**
     * Constructs a quality metric set for the pca algorithm
     * \param[in] nComponents           Number of principal components
     * \param[in] nFeatures             Number of features of source dataset
     * \param[in] useDefaultMetrics     Flag. If true, a quality metric set is initialized with the quality metrics provided by the library
     */
    Batch(size_t nComponents = 0, size_t nFeatures = 0, bool useDefaultMetrics = true):
        algorithms::quality_metric_set::Batch(useDefaultMetrics), parameter(nComponents, nFeatures)
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
    virtual void initializeQualityMetrics();
};
/** @} */
} // namespace interface1
using interface1::Batch;

}
}
}
}
#endif

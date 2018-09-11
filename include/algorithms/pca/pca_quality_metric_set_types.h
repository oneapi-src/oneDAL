/* file: pca_quality_metric_set_types.h */
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
//  Interface for the pca algorithm quality metrics
//--
*/

#ifndef __PCA_QUALITY_METRIC_SET_TYPES_H__
#define __PCA_QUALITY_METRIC_SET_TYPES_H__

#include "services/daal_shared_ptr.h"
#include "algorithms/algorithm.h"
#include "algorithms/algorithm_quality_metric_set_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
/**
 * @defgroup pca_quality_metric_set Quality Metrics
 * \copydoc daal::algorithms::pca::quality_metric_set
 * @ingroup pca
 * @{
 */
namespace quality_metric_set
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__QUALITY_METRIC_SET__QUALITYMETRICID"></a>
 * Available identifiers of the quality metrics of pca algorithm
 */
enum QualityMetricId
{
    explainedVariancesMetrics,     /*!< Explained and noise variances metrics*/
    lastQualityMetricId = explainedVariancesMetrics
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
* <a name="DAAL-STRUCT-ALGORITHMS__PCA__QUALITY_METRIC_SET__PARAMETER"></a>
* \brief Parameters for the quality metrics set compute() method
*
* \snippet pca/pca_quality_metric_set_types.h Parameter source code
*/
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nComponents = 0, size_t nFeatures = 0);

    virtual ~Parameter() {}

    size_t nComponents;              /*!< Number of principal components to compute metrics for */
    size_t nFeatures;                /*!< Number of features in dataset used as input in PCA */

    /**
    * Checks the correctness of the parameter
    *
     * \return Status of computations
     */
    virtual services::Status check() const;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC_SET__RESULTCOLLECTION"></a>
 * \brief Class that implements functionality of the collection of result objects of the quality metrics algorithm
 *        specialized for using with the pca algorithm
 */
class DAAL_EXPORT ResultCollection: public algorithms::quality_metric_set::ResultCollection
{
public:
    ResultCollection() {}
    virtual ~ResultCollection() {}

    /**
     * Returns the result of the quality metrics algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    algorithms::ResultPtr getResult(QualityMetricId id) const;
};
typedef services::SharedPtr<ResultCollection> ResultCollectionPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC_SET__INPUTDATACOLLECTION"></a>
 * \brief Class that implements functionality of the collection of input objects of the quality metrics algorithm
 *        specialized for using with the pca algorithm
 */
class DAAL_EXPORT InputDataCollection: public algorithms::quality_metric_set::InputDataCollection
{
public:
    InputDataCollection() {}
    virtual ~InputDataCollection() {}

    /**
     * Returns the input object of the quality metrics algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    algorithms::InputPtr getInput(QualityMetricId id) const;
};
typedef services::SharedPtr<InputDataCollection> InputDataCollectionPtr;

}
using interface1::Parameter;
using interface1::ResultCollection;
using interface1::InputDataCollection;
using interface1::ResultCollectionPtr;
using interface1::InputDataCollectionPtr;

}
/** @} */
}
}
}

#endif // __PCA_QUALITY_METRIC_SET_TYPES_H__

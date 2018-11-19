/* file: algorithm_quality_metric_batch.h */
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
//  Interface for the quality metrics in the batch processing mode.
//--
*/

#ifndef __ALGORITHM_QUALITY_METRIC_BATCH_H__
#define __ALGORITHM_QUALITY_METRIC_BATCH_H__

#include "algorithms/analysis.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes to compute quality metrics
 */
namespace quality_metric
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
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC__BATCH"></a>
 * \brief Provides methods to compute quality metrics of an algorithm in the batch processing mode.
 *        Quality metric is a numerical characteristic or a set of connected numerical characteristics
 *        that represents the qualitative aspect of a computed statistical estimate, model,
 *        or decision-making result.
 */
class DAAL_EXPORT Batch : public Analysis<batch>
{
public:
    Batch() : Analysis<batch>() {}
    virtual ~Batch() {}

    /**
     * Sets an input object
     * \param[in] input Pointer to the input object
     */
    virtual void setInput(const algorithms::Input *input) = 0;

    /**
     * Returns the structure that contains computed quality metrics
     * \return Structure that contains computed quality metrics
     */
     algorithms::ResultPtr getResult() const
     {
        return getResultImpl();
     }

protected:
     virtual algorithms::ResultPtr getResultImpl() const = 0;
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace quality_metric
} // namespace algorithms
} // namespace daal
#endif

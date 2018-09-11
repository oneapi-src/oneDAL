/* file: em_gmm_covariance_storage_id.h */
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
//  the EM for GMM CovarianceStorageId class.
//--
*/

#ifndef __EM_GMM_COVARIANCE_STORAGE_ID_H__
#define __EM_GMM_COVARIANCE_STORAGE_ID_H__

namespace daal
{
namespace algorithms
{
namespace em_gmm
{

/**
 * @ingroup em_gmm
 * @{
 */
/**
 * <a name="DAAL-ENUM-ALGORITHMS__EM_GMM__COVARIANCESTORAGEID"></a>
 * Available identifiers of covariance types in the EM for GMM algorithm
 */
enum CovarianceStorageId
{
    full,
    diagonal
};

/** @} */

} // namespace em_gmm
} // namespace algorithm
} // namespace daal
#endif

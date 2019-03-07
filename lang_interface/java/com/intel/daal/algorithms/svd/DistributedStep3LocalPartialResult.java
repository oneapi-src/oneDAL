/* file: DistributedStep3LocalPartialResult.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

/**
 * @ingroup svd_distributed
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDSTEP3LOCALPARTIALRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the SVD algorithm in the third step in the
 * distributed processing mode
 */
public class DistributedStep3LocalPartialResult extends com.intel.daal.algorithms.PartialResult {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep3LocalPartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */

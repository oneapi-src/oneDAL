/* file: DistributedStep1LocalInput.java */
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

/**
 * @ingroup kmeans_distributed
 * @{
 */
package com.intel.daal.algorithms.kmeans;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTEDSTEP1LOCALINPUT"></a>
 * @brief Input objects for the K-Means algorithm.
 *        Represents input objects for the algorithm on local nodes.
 */
public final class DistributedStep1LocalInput extends com.intel.daal.algorithms.kmeans.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep1LocalInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */

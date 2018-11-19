/* file: InitDistributedStep1LocalInput.java */
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
 * @ingroup kmeans_init_distributed
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP1LOCALINPUT"></a>
 * @brief Input objects for computing initial clusters for the K-Means algorithm.
 *        The class represents input objects for computing initial clusters for the algorithm on local nodes.
 */
public final class InitDistributedStep1LocalInput extends com.intel.daal.algorithms.kmeans.init.InitInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep1LocalInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */

/* file: DistributedStep2MasterInput.java */
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
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input objects for the K-Means algorithm in the second step of the distributed processing mode.
 *        Represents input objects for the algorithm on the master node.
 */
public final class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep2MasterInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Adds a partial result computed on local nodes to the input for the K-Means algorithm
     * in the second step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param pres          Partial results of the algorithm obtained in the first step
     *                      of the distributed processing mode
     */

    public void add(DistributedStep2MasterInputId id, PartialResult pres) {
        cAddInput(cObject, id.getValue(), pres.getCObject());
    }

    private native void cAddInput(long algAddr, int id, long presAddr);
}
/** @} */

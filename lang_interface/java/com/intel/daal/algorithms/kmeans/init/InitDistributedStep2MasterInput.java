/* file: InitDistributedStep2MasterInput.java */
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
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input objects for computing initial clusters for the K-Means algorithm.
 *        The class represents input objects for computing initial clusters for the algorithm on the master node.
 */
public final class InitDistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep2MasterInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Adds partial results computed on local nodes to the input for computing initial clusters for the K-Means algorithm
     * in the second step in the distributed processing mode
     * @param id            Identifier of the input object
     * @param pres          Partial results of the K-Means initialization algorithm obtained in the
     *                      first step of the distributed processing mode
     */

    public void add(InitDistributedStep2MasterInputId id, InitPartialResult pres) {
        cAddInput(cObject, id.getValue(), pres.getCObject());
    }

    private native void cAddInput(long inputAddr, int id, long presAddr);
}
/** @} */

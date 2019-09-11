/* file: DistributedStep2MasterInput.java */
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
 * @ingroup pca_distributed
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.PartialResult;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input objects for the second step of the PCA algorithm
 *        in the distributed processing mode.
 */
public final class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    public Method                               method;  /*!< Computation method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep2MasterInput(DaalContext context, long cObject, Method method) {
        super(context, cObject);
        this.method = method;
    }

    /**
     * Adds partial result to the input of the PCA algorithm on the second step in the distributed processing mode
     * @param id            Identifier of the input object
     * @param pres          Partial result obtained on the first step of the PCA algorithm in the distributed processing mode
     */
    public void add(MasterInputId id, PartialResult pres) {
        cAddInput(cObject, id.getValue(), pres.getCObject(), method.getValue());
    }

    private native void cAddInput(long algAddr, int id, long presAddr, int method);
}
/** @} */

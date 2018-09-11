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
 * @ingroup implicit_als_training_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.implicit_als.PartialModel;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDSTEP1LOCALINPUT"></a>
 * @brief %Input objects for the implicit ALS training algorithm in the first step of the distributed processing mode
 */

public final class DistributedStep1LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep1LocalInput(DaalContext context, long cAlgorithm, Precision prec, TrainingMethod method) {
        super(context);
        this.cObject = cGetInput(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Sets an input object for the implicit ALS training algorithm in the first step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(PartialModelInputId id, PartialModel val) {
        if (id != PartialModelInputId.partialModel) {
            throw new IllegalArgumentException("Incorrect id");
        }
        cSetPartialModel(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input object for the implicit ALS training algorithm in the first step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public PartialModel get(PartialModelInputId id) {
        if (id != PartialModelInputId.partialModel) {
            throw new IllegalArgumentException("Incorrect id"); // error processing
        }
        return new PartialModel(getContext(), cGetPartialModel(this.cObject, id.getValue()));
    }

    private native long cGetInput(long cAlgorithm, int prec, int method);

    private native void cSetPartialModel(long cObject, int id, long partialModelAddr);
    private native long cGetPartialModel(long cObject, int id);
}
/** @} */

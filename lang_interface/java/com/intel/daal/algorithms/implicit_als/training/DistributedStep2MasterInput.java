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
 * @ingroup implicit_als_training_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief %Input objects for the implicit ALS training algorithm in the second step of the distributed processing mode
 */
public final class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep2MasterInput(DaalContext context, long cAlgorithm, Precision prec, TrainingMethod method) {
        super(context);
        this.cObject = cGetInput(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Adds an input object for the implicit ALS training algorithm in the second step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param partialResult Value of the input object
     */
    public void add(MasterInputId id, DistributedPartialResultStep1 partialResult) {
        if (id != MasterInputId.inputOfStep2FromStep1) {
            throw new IllegalArgumentException("Incorrect id");
        }
        cAddDataCollection(this.cObject, id.getValue(), partialResult.getCObject());
    }

    /**
     * Returns an input object for the implicit ALS training algorithm in the second step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        Input object that corresponds to the given identifier
     */
    public DataCollection get(MasterInputId id) {
        if (id != MasterInputId.inputOfStep2FromStep1) {
            throw new IllegalArgumentException("Incorrect id");
        }
        return new DataCollection(getContext(), cGetDataCollection(this.cObject, id.getValue()));
    }

    private native long cGetInput(long cAlgorithm, int prec, int method);

    private native void cAddDataCollection(long cInput, int id, long partialResultAddr);
    private native long cGetDataCollection(long cInput, int id);
}
/** @} */

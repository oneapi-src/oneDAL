/* file: InitInput.java */
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
 * @defgroup implicit_als_init Initialization
 * @brief Contains classes for the implicit ALS initialization algorithm
 * @ingroup implicit_als_training
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.utils.*;
import java.io.Serializable;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITINPUT"></a>
 * @brief Initializes input objects for the implicit ALS initialization algorithm
 */
public class InitInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input for the implicit ALS initialization algorithm in the distributed processing mode
     * @param context Context to manage the constructed object
     */
    public InitInput(DaalContext context) {
        super(context);
    }

    public InitInput(DaalContext context, long cInput) {
        super(context);
        this.cObject = cInput;
    }

    public InitInput(DaalContext context, long cAlgorithm, Precision prec, InitMethod method, ComputeMode cmode) {
        super(context);
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue());
    }

    /**
     * Sets an input object for the implicit ALS initialization algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(InitInputId id, Serializable val) {
        if (id != InitInputId.data) {
            throw new IllegalArgumentException("Incorrect InitInputId");
        }
        cSetInput(cObject, id.getValue(), ((NumericTable) val).getCObject());
    }

    /**
     * Returns an input object for the implicit ALS initialization algorithm
     * @param id    Identifier of the input object
     * @return      %Input object that corresponds to the given identifier
     */
    public NumericTable get(InitInputId id) {
        if (id != InitInputId.data) {
            throw new IllegalArgumentException("Incorrect InitInputId");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
    }

    protected long cObject;

    private native long cInit(long algAddr, int prec, int method, int cmode);

    private native void cSetInput(long cInput, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);

}
/** @} */

/* file: InitInput.java */
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
 * @brief Contains classes for initializing the EM for GMM algorithm
 */
/**
 * @ingroup em_gmm_init
 * @{
 */
package com.intel.daal.algorithms.em_gmm.init;

import com.intel.daal.utils.*;
import java.io.Serializable;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__INITINPUT"></a>
 * @brief  %Input objects for the default initialization of the EM for GMM algorithm
 */
public class InitInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
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
     * Sets the input object for the default initialization of the EM for GMM algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public int set(InitInputId id, Serializable val) {
        if (id == InitInputId.data) {
            return cSetInput(cObject, id.getValue(), ((NumericTable) val).getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object for the default initialization of the EM for GMM algorithm
     * @param id Identifier of the input object
     * @return   %Input object that corresponds to the given identifier
     */
    public NumericTable get(InitInputId id) {
        if (id == InitInputId.data) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    protected long cObject;

    private native long cInit(long algAddr, int prec, int method, int cmode);

    private native int cSetInput(long cInput, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);

}
/** @} */

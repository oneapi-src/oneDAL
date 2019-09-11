/* file: Input.java */
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
 * @ingroup cosine_distance
 * @{
 */
package com.intel.daal.algorithms.cosdistance;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COSDISTANCE__INPUT"></a>
 * \brief %Input objects for the cosine distance algorithm
 */
public final class Input extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Input(DaalContext context, long cObject) {
        super(context, cObject);
    }

    public Input(DaalContext context, long cAlgorithm, Precision prec, Method method) {
        super(context);
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
    * Sets the input object of the cosine distance algorithm
    * @param id    %Identifier of the input object
     * @param val  Value to set
    */
    public void set(InputId id, NumericTable val) {
        if (id != InputId.data) {
            throw new IllegalArgumentException("id unsupported");
        }

        NumericTable nt = val;
        long ntAddr = nt.getCObject();
        cSetInput(this.cObject, id.getValue(), ntAddr);
    }

    /**
    * Gets the input object for the cosine distance algorithm
    * @param id    Identifier of the input object
     * @return     %Input object that corresponds to the given identifier
    */
    public NumericTable get(InputId id) {
        if (id != InputId.data) {
            throw new IllegalArgumentException("id unsupported");
        }

        return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(this.cObject, id.getValue()));
    }

    private native long cInit(long algAddr, int prec, int method);

    private native void cSetInput(long inputAddr, int id, long ntAddr);

    private native long cGetInput(long inputAddr, int id);
}
/** @} */

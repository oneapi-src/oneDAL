/* file: DistributedStep3LocalInput.java */
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
 * @ingroup gbt_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDSTEP3LOCALINPUT"></a>
 * @brief %Input objects for the model-based training algorithm in the third step of the distributed processing mode
 */

public final class DistributedStep3LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep3LocalInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets an input object for the model-based training in the third step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step3LocalNumericTableInputId id, NumericTable val) {
        if (id == Step3LocalNumericTableInputId.step3BinnedData || id == Step3LocalNumericTableInputId.step3InputTreeStructure ||
            id == Step3LocalNumericTableInputId.step3InputTreeOrder || id == Step3LocalNumericTableInputId.step3OptCoeffs ||
            id == Step3LocalNumericTableInputId.step3BinSizes) {
            cSetNumericTable(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the model-based training in the third step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public NumericTable get(Step3LocalNumericTableInputId id) {
        if (id == Step3LocalNumericTableInputId.step3BinnedData || id == Step3LocalNumericTableInputId.step3InputTreeStructure ||
            id == Step3LocalNumericTableInputId.step3InputTreeOrder || id == Step3LocalNumericTableInputId.step3OptCoeffs ||
            id == Step3LocalNumericTableInputId.step3BinSizes) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets an input object for the the model-based training in the third step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step3LocalCollectionInputId id, DataCollection val) {
        if (id == Step3LocalCollectionInputId.step3ParentHistograms) {
            cSetDataCollection(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Adds an input object for the the model-based training in the third step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param val           Value of the input object
     */
    public void add(Step3LocalCollectionInputId id, NumericTable val) {
        if (id == Step3LocalCollectionInputId.step3ParentHistograms) {
            cAddNumericTable(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the the model-based training in the third step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public DataCollection get(Step3LocalCollectionInputId id) {
        if (id == Step3LocalCollectionInputId.step3ParentHistograms) {
            return (DataCollection)Factory.instance().createObject(getContext(), cGetDataCollection(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetNumericTable(long cObject, int id, long ntAddr);
    private native long cGetNumericTable(long cObject, int id);

    private native void cSetDataCollection(long cObject, int id, long dcAddr);
    private native void cAddNumericTable(long cObject, int id, long ntAddr);
    private native long cGetDataCollection(long cObject, int id);
}
/** @} */

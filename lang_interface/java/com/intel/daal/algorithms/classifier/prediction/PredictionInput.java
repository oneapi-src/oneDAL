/* file: PredictionInput.java */
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
 * @ingroup prediction
 * @{
 */
package com.intel.daal.algorithms.classifier.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.classifier.Model;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__PREDICTIONINPUT"></a>
 * @brief  %Input objects for the classification algorithm
 */
public class PredictionInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input of the classifier model-based prediction algorithm
     * @param context   Context to manage the input of the classifier model-based prediction algorithm
     */
    public PredictionInput(DaalContext context) {
        super(context);
    }

    public PredictionInput(DaalContext context, long cAlgorithm) {
        super(context);
        this.cObject = cInit(cAlgorithm);
    }

    public PredictionInput(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context);
        this.cObject = cInit(cAlgorithm);
    }

    /**
     * Sets the NumericTable input object for the classifier model-based prediction algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(NumericTableInputId id, NumericTable val) {
        if (id != NumericTableInputId.data) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInputTable(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the NumericTable input object for the classifier model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(NumericTableInputId id) {
        if (id == NumericTableInputId.data) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the Model input object for the classifier model-based prediction algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(ModelInputId id, Model val) {
        if (id != ModelInputId.model) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInputModel(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the Model input object for the classifier model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Model get(ModelInputId id) {
        if (id == ModelInputId.model) {
            return new Model(getContext(), cGetInputModel(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInputTable(long inputAddr, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);

    private native void cSetInputModel(long inputAddr, int id, long modelAddr);

    protected native long cGetInputModel(long inputAddr, int id);

    private native long cInit(long inputAddr);
}
/** @} */

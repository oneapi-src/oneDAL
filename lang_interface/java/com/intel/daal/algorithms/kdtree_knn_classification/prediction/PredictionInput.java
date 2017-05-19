/* file: PredictionInput.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/**
 * @defgroup kdtree_knn_classification_prediction Prediction
 * @brief Contains a class for making KD-tree based kNN model-based prediction
 * @ingroup kdtree_knn_classification
 * @{
 */
package com.intel.daal.algorithms.kdtree_knn_classification.prediction;

import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.algorithms.kdtree_knn_classification.Model;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__PREDICTIONINPUT"></a>
 * @brief  %Input objects for the k nearest neighbors algorithm
 */
public class PredictionInput extends com.intel.daal.algorithms.classifier.prediction.PredictionInput {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public PredictionInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the NumericTable input object for the k nearest neighbors model-based prediction algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(NumericTableInputId id, NumericTable val) {
        if (id != NumericTableInputId.data) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInput(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the NumericTable input object for the k nearest neighbors model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(NumericTableInputId id) {
        if (id == NumericTableInputId.data) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the Model input object for the k nearest neighbors model-based prediction algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(ModelInputId id, Model val) {
        if (id != ModelInputId.model) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInput(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the Model input object for the k nearest neighbors model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Model get(ModelInputId id) {
        if (id == ModelInputId.model) {
            return new Model(getContext(), cGetInput(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long inputAddr, int id, long ntAddr);
    private native long cGetInput(long cInput, int id);
}
/** @} */

/* file: TransformBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @defgroup pca_transform Pca-Transform
 * @brief Contains classes for PCA transformation algorithms
 * @ingroup pca
 * @{
 */
/**
 * @defgroup pca_transform_batch Batch
 * @ingroup pca_transform
 * @{
 */
/**
 * @brief Contains classes for computing PCA transformation solvers
 */
package com.intel.daal.algorithms.pca.transform;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.ComputeMode;

/**
 * <a name="DAAL-CLASS-ALGORITHMS-ALGORITHMS__PCA__TRANSFORM__BATCH"></a>
 * \brief Computes PCA transformation in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-PCA-TRANSFORM-ALGORITHM">PCA transformation algorithm description and usage models</a> -->
 *
 * \par References
 *      - @ref InputId class
 *      - @ref ResultId class
 *
 */
public class TransformBatch extends AnalysisBatch {
    public TransformInput      input;     /*!< %Input data */
    public TransformMethod     method;    /*!< Computation method for the algorithm */
    private Precision prec;      /*!< Precision of computations */
    public TransformParameter  parameter; /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * <a name="DAAL-CLASS-ALGORITHMS-ALGORITHMS__PCA__TRANSFORM__BATCH__BATCH"></a>
     * Constructs the PCA transformation algorithm
     *
     * @param context     Context to manage the PCA transformation algorithm
     * @param cls         Data type to use in intermediate computations for PCA transformation, Double.class or Float.class
     * @param nComponents Number of components
     * @param method      PCA transformation computation method, @ref Method
     */
    public TransformBatch(DaalContext context, Class<? extends Number> cls, TransformMethod method, long nComponents) {
        super(context);

        this.method = method;

        if (method != TransformMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue(), nComponents);
        input = new TransformInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TransformParameter(context, cGetParameter(cObject, prec.getValue(), method.getValue()));
        parameter.setNumberOfComponents(nComponents);
    }

    /**
     * <a name="DAAL-CLASS-ALGORITHMS-ALGORITHMS__PCA__TRANSFORM__BATCH__BATCH2"></a>
     * Constructs the PCA transformation algorithm
     *
     * @param context     Context to manage the PCA transformation algorithm
     * @param cls         Data type to use in intermediate computations for PCA transformation, Double.class or Float.class
     * @param method      PCA transformation computation method, @ref Method
     */
    public TransformBatch(DaalContext context, Class<? extends Number> cls, TransformMethod method) {
        this(context, cls, method, 0);
    }

    /**
    * Constructs algorithm that computes normalization by copying input objects and parameters
    * of another algorithm
    * @param context      Context to manage the normalization algorithms
    * @param other        An algorithm to be used as the source to initialize the input objects
    *                     and parameters of the algorithm
    */
    public TransformBatch(DaalContext context, TransformBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new TransformInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TransformParameter(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes PCA transformation
     * @return  PCA transformation results
    */
    @Override
    public TransformResult compute() {
        super.compute();
        TransformResult result = new TransformResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the PCA transformation algorithm
     * @param result    Structure to store the result of the PCA transformation algorithm
     */
    public void setResult(TransformResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated algorithm that computes normalization
     * with a copy of input objects and parameters of this algorithm
     * @param context      Context to manage the normalization algorithms
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TransformBatch clone(DaalContext context) {
        return new TransformBatch(context, this);
    }

    private native long cInit(int prec, int method, long nComponents);
    private native long cGetParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */

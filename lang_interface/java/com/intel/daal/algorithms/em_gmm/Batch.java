/* file: Batch.java */
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
 * @defgroup em_gmm_compute Computation
 * @brief Contains classes for the EM for GMM algorithm
 * @ingroup em_gmm
 * @{
 */
/**
 * @defgroup em_gmm_batch Batch
 * @ingroup em_gmm_compute
 * @{
 */
package com.intel.daal.algorithms.em_gmm;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__BATCH"></a>
 * \brief Runs the EM for GMM algorithm in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-EM_GMM-ALGORITHM">EM for GMM algorithm description and usage models</a> -->
 *
 * \par References
 *      - @ref InputId class
 *      - @ref ResultId class
 *
 */
public class Batch extends AnalysisBatch {
    public Input          input;     /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public Method     method; /*!< Computation method for the algorithm */
    private Result    result;      /*!< %Result of the algorithm */
    private Precision precision; /*!< Precision of intermediate computations */
    private long      nComponents; /*!< Number of components in the Gaussian mixture model */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the EM for GMM algorithm by copying input objects and parameters
     * of another EM for GMM algorithm
     * @param context      Context to manage the EM for GMM algorithm
     * @param other        An algorithm to be used as the source to initialize the input objects
     *                     and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        method = other.method;
        precision = other.precision;
        nComponents = other.nComponents;

        this.cObject = cClone(other.cObject, precision.getValue(), method.getValue());

        input = new Input(getContext(), cObject, precision, method, ComputeMode.batch);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, precision.getValue(), method.getValue()));
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__EM_GMM__BATCH__BATCH"></a>
     * Constructs the EM for GMM algorithm
     *
     * @param context      Context to manage the EM for GMM algorithm
     * @param cls          Data type to use in intermediate computations for the EM for GMM algorithm, Double.class or Float.class
     * @param method       EM for GMM computation method, @ref Method
     * @param nComponents  Number of components in the Gaussian mixture model
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method, long nComponents) {
        super(context);
        this.method = method;
        this.nComponents = nComponents;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            precision = Precision.doublePrecision;
        } else {
            precision = Precision.singlePrecision;
        }

        this.cObject = cInit(precision.getValue(), this.method.getValue(), nComponents);

        input = new Input(getContext(), cObject, precision, method, ComputeMode.batch);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, precision.getValue(), method.getValue()));
    }

    /**
    * Runs the EM for GMM algorithm
    * @return Results of the EM for GMM algorithm
    */
    @Override
    public Result compute() {
        super.compute();
        result = new Result(getContext(), cGetResult(cObject, precision.getValue(), method.getValue()));
        return result;
    }

    /**
    * Registers user-allocated memory for storing results of the EM for GMM algorithm
    * @param result    Structure for storing the results EM for GMM algorithm
    */
    public void setResult(Result result) {
        cSetResult(cObject, precision.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated EM for GMM algorithm with a copy of input objects
     * of this EM for GMM algorithm
     * @param context      Context to manage the EM for GMM algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int precision, int method, long nComponents);

    private native long cGetResult(long cObject, int prec, int method);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native long cInitParameter(long algAddr, int precision, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */

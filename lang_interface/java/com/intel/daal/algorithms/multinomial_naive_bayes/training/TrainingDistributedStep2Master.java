/* file: TrainingDistributedStep2Master.java */
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
 * @ingroup multinomial_naive_bayes_training_distributed
 * @{
 */
package com.intel.daal.algorithms.multinomial_naive_bayes.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.multinomial_naive_bayes.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__TRAININGDISTRIBUTEDSTEP2MASTER"></a>
 * @brief Algorithm class for training naive Bayes model on the second step in the distributed processing mode
 * <!-- \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a> -->
 *
 * @par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 *      - com.intel.daal.algorithms.multinomial_naive_bayes.training.TrainingDistributedInput class
 */
public class TrainingDistributedStep2Master extends com.intel.daal.algorithms.TrainingDistributed {
    public  Parameter                parameter; /*!< Parameters of the algorithm */
    public  TrainingDistributedInput input;     /*!< %Input objects for the algorithm */
    public  TrainingMethod           method;    /*!< %Training method for the algorithm */
    private Precision                prec;      /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs multinomial naive Bayes training algorithm by copying input objects and parameters
     * of another multinomial naive Bayes training algorithm
     * @param context   Context to manage the multinomial naive Bayes training
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingDistributedStep2Master(DaalContext context, TrainingDistributedStep2Master other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());

        input = new TrainingDistributedInput(getContext(), cGetInput(this.cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs multinomial naive Bayes training algorithm
     * @param context   Context to manage the multinomial naive Bayes training
     * @param cls       Data type to use in intermediate computations of the multinomial naive Bayes training on the second step in the distributed
     *                  processing mode,
     *                  Double.class or Float.class
     * @param method    Multinomial naive Bayes training method on the second step in the distributed processing mode, @ref TrainingMethod
     * @param nClasses  Number of classes
     */
    public TrainingDistributedStep2Master(DaalContext context, Class<? extends Number> cls, TrainingMethod method,
            long nClasses) {
        super(context);

        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != TrainingMethod.defaultDense &&
            this.method != TrainingMethod.fastCSR) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue(), nClasses);

        input = new TrainingDistributedInput(getContext(), cGetInput(this.cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes naive Bayes training results on the second step in the distributed processing mode
     * \return Naive Bayes training results on the second step in the distributed processing mode
     */
    @Override
    public TrainingPartialResult compute() {
        super.compute();
        TrainingPartialResult presult = new TrainingPartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        return presult;
    }

    /**
     * Computes naive Bayes training results on the second step in the distributed processing mode
     * \return Naive Bayes training results on the second step in the distributed processing mode
     */
    @Override
    public TrainingResult finalizeCompute() {
        super.finalizeCompute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store naive Bayes training results
     * @param result    Structure to store naive Bayes training results
     */
    public void setResult(TrainingResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Registers user-allocated memory to store naive Bayes training partial results
     * @param result    Structure to store naive Bayes training partial results
     */
    public void setPartialResult(TrainingPartialResult result) {
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated multinomial naive Bayes training algorithm
     * with a copy of input objects and parameters of this multinomial naive Bayes training algorithm
     * @param context   Context to manage the multinomial naive Bayes training
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingDistributedStep2Master clone(DaalContext context) {
        return new TrainingDistributedStep2Master(context, this);
    }

    private native long cInit(int prec, int method, long nClasses);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native void cSetResult(long algAddr, int prec, int method, long cObject);

    private native long cGetPartialResult(long algAddr, int prec, int method);

    private native void cSetPartialResult(long algAddr, int prec, int method, long cObject);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */

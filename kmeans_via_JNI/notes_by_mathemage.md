# Notes on kmeans via JNI

## Settings on MacBook Pro 12.1
Intel DAAL supports the IA-32 and Intel(R) 64 architectures. For a detailed explanation of these architecture names, read the [Intel Architecture Platform Terminology for Development Tools](https://software.intel.com/en-us/articles/intel-architecture-platform-terminology-for-development-tools) article.

### Operating Systems
* Ubuntu* 17.04 (zesty)

### C/C++ Compilers
* gcc version 6.3.0 20170406 (Ubuntu 6.3.0-12ubuntu2)

### Java* Compilers:

    $ java -version
    openjdk version "1.8.0_131"
    OpenJDK Runtime Environment (build 1.8.0_131-8u131-b11-0ubuntu1.17.04.1-b11)
    OpenJDK 64-Bit Server VM (build 25.131-b11, mixed mode)

## Installation

### Installing from the Sources

#### Installation Steps
1. Clone the sources from GitHub* as follows:

        $ git clone --recursive https://github.com/01org/daal.git
        Cloning into 'daal'...
        remote: Counting objects: 20865, done.
        remote: Total 20865 (delta 0), reused 0 (delta 0), pack-reused 20865
        Receiving objects: 100% (20865/20865), 448.05 MiB | 1.68 MiB/s, done.
        Resolving deltas: 100% (16684/16684), done.
        Checking out files: 100% (5367/5367), done.

2. Set an environment variable for one of the supported Java* compilers; for example:

        $ echo $PATH
        /home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin

        $ cat >>~/.bashrc
        # Java JDK -> $PATH
        JAVA_JDK=/usr/lib/jvm/java-8-openjdk-amd64/bin
        export JAVA_JDK
        PATH=$JAVA_JDK/bin:$PATH
        export PATH

        $ source ~/.bashrc

        $ echo $PATH
        /usr/lib/jvm/java-8-openjdk-amd64/bin/bin:/home/mathemage/torch/install/bin:jdk-install-dir/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin

3. Build Intel DAAL via the command-line interface with the following commands (on Linux using GNU Compiler Collection):

        make daal PLAT=lnx32e COMPILER=gnu
            
 * JNI compilation issue:
 
        ========= Building threading =========
        ========= Building java =========
        g++ -D__int64="long long" -D__int32="int" -m64 -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -fPIC -std=c++11   -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED @./__work_gnu/lnx32e/jni_tmpdir/inc_j_folders.txt  -c  -o__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o lang_service/java/com/intel/daal/algorithms/parameter.cpp && printf '\n%s\n' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps += makefile build/cmplr.gnu.mk build/common.mk build/deps.mk makefile.ver' '$(__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps):' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o: $(__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps)' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.trigger = g++ -D__int64="long long" -D__int32="int" -m64 -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -fPIC -std=c++11   -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED @./__work_gnu/lnx32e/jni_tmpdir/inc_j_folders.txt  -c  -o__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o lang_service/java/com/intel/daal/algorithms/parameter.cpp' >> __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d.tmp && mv -f __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d.tmp __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d
        lang_service/java/com/intel/daal/algorithms/parameter.cpp:18:17: fatal error: jni.h: No such file or directory
         #include <jni.h>
                         ^
        compilation terminated.
        makefile:576: recipe for target '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o' failed
        make: *** [__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o] Error 1
    
    Stack Overflow suggests this [solution](https://stackoverflow.com/questions/14529720/how-to-make-jni-h-be-found):

        make -I/usr/lib/jvm/jdk*/include
    
    Hence, I added following line to [makefile](../makefile):
    
        $(THR.objs): INCLUDES += $(addprefix -I, /usr/lib/jvm/java-8-openjdk-amd64/include/)

 * `libdaal_core.a` compilation issue:
 
        ========= Building core =========
        printf "create __work_gnu/lnx32e/daal/lib/libdaal_core.a\naddlib externals/mklfpk/lnx/lib/intel64/libdaal_vmlipp_core.a\n addlib __work_gnu/lnx32e/kernel/libdaal_core_link.a\n\n\n\nsave\n" | ar -M && printf '\n%s\n' '__work_gnu/lnx32e/daal/lib/libdaal_core.a.mkdeps += makefile build/cmplr.gnu.mk build/common.mk build/deps.mk makefile.ver' '$(__work_gnu/lnx32e/daal/lib/libdaal_core.a.mkdeps):' '__work_gnu/lnx32e/daal/lib/libdaal_core.a: $(__work_gnu/lnx32e/daal/lib/libdaal_core.a.mkdeps)' '__work_gnu/lnx32e/daal/lib/libdaal_core.a.trigger = printf "create __work_gnu/lnx32e/daal/lib/libdaal_core.a\naddlib externals/mklfpk/lnx/lib/intel64/libdaal_vmlipp_core.a\n addlib __work_gnu/lnx32e/kernel/libdaal_core_link.a\n\n\n\nsave\n" | ar -M' >> __work_gnu/lnx32e/daal/lib/libdaal_core.a.d.tmp && mv -f __work_gnu/lnx32e/daal/lib/libdaal_core.a.d.tmp __work_gnu/lnx32e/daal/lib/libdaal_core.a.d
        ar: __work_gnu/lnx32e/kernel/libdaal_core_link.a: No such file or directory
        makefile:356: recipe for target '__work_gnu/lnx32e/daal/lib/libdaal_core.a' failed
        make: *** [__work_gnu/lnx32e/daal/lib/libdaal_core.a] Error 9

    I make-cleaned and reinstall again. This time, I got a different issue (see below).
    
 * JNI issue again:
 
        $ make _daal_jj PLAT=lnx32e COMPILER=gnu
        ========= Building java =========
        g++ -D__int64="long long" -D__int32="int" -m64 -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -fPIC -std=c++11   -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED @./__work_gnu/lnx32e/jni_tmpdir/inc_j_folders.txt  -c  -o__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o lang_service/java/com/intel/daal/algorithms/parameter.cpp && printf '\n%s\n' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps += makefile build/cmplr.gnu.mk build/common.mk build/deps.mk makefile.ver' '$(__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps):' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o: $(__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps)' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.trigger = g++ -D__int64="long long" -D__int32="int" -m64 -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -fPIC -std=c++11   -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED @./__work_gnu/lnx32e/jni_tmpdir/inc_j_folders.txt  -c  -o__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o lang_service/java/com/intel/daal/algorithms/parameter.cpp' >> __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d.tmp && mv -f __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d.tmp __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d
        lang_service/java/com/intel/daal/algorithms/parameter.cpp:18:17: fatal error: jni.h: No such file or directory
         #include <jni.h>
                         ^
        compilation terminated.
        makefile:577: recipe for target '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o' failed
        make: *** [__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o] Error 1
 
    The [solution](https://github.com/01org/daal/issues/4) is taken from the issue #4 of DAAL's GitHub, which finally compiles:
    
        $ make daal PLAT=lnx32e COMPILER=gnu CPATH=/usr/lib/jvm/java-8-openjdk-amd64/include:/usr/lib/jvm/java-8-openjdk-amd64/include/linux/
        ========= Building core =========
        ========= Building threading =========
        ========= Building java =========
        ========= Building release =========
        make: Nothing to be done for 'daal'.
        
    or, if you have installed Oracle JDK:
    
        $ make daal PLAT=lnx32e COMPILER=gnu CPATH=/usr/lib/jvm/java-8-oracle/include:/usr/lib/jvm/java-8-oracle/include/linux/

4. Set up environment variables of DAAL. Find out the necessary command:

        mathemage@mathemage-MacBookPro:~/h2oai/daal/__release_lnx_gnu/daal/bin$ ./daalvars.sh daal_help
        Syntax: source daalvars.sh <arch>
        Where <arch> is one of:
          ia32      - setup environment for IA-32 architecture
          intel64   - setup environment for Intel(R) 64 architecture
        
        If the arguments to the sourced script are ignored (consult docs for
        your shell) the alternative way to specify target is environment
        variables COMPILERVARS_ARCHITECTURE or DAALVARS_ARCHITECTURE to pass
        <arch> to the script.
        
    Set up the environment variables:
   
        mathemage@mathemage-MacBookPro:~/h2oai/daal/__release_lnx_gnu/daal/bin$ source daalvars.sh intel64

    Verify:
   
        mathemage@mathemage-MacBookPro:~/h2oai/daal/__release_lnx_gnu/daal/bin$ echo $DAALROOT 
        /home/mathemage/h2oai/daal/__release_lnx_gnu/daal

    
  ## DAAL via Java
  
  ### Examples by Intel
  There are some prepared examples of DAAL in Java by Intel, e.g., *kmeans-clustering* examples in [h2oai/daal/examples/java/com/intel/daal/examples/kmeans/](../examples/java/com/intel/daal/examples/kmeans/)
  
  ### Build
  The [Intel DAAL Getting Started](https://software.intel.com/en-us/get-started-with-daal-for-linux) page mentions using DAAL via Java:
   
  > To build and run Java code examples, use the version of the Java Virtual Machine* corresponding to the architecture parameter you provided to the daalvars.sh script during setting environment variables.
  > 
  > 1. Free 4 gigabytes of memory on your system.
  > 2. Build examples:
  >
  > Go to the Java examples directory and execute the launcher command with the build parameter:
  > 
  >     cd <install dir>/daal/examples/java
  >     launcher.sh build $PATH_TO_JAVAC
  >
In my case, `<install dir>` was `/home/mathemage/h2oai/daal/__release_lnx_gnu`, so the `cd` command was
    
    mathemage@mathemage-MacBookPro:~/h2oai/daal/examples/java$ cd ../../__release_lnx_gnu/daal/examples/java/
    mathemage@mathemage-MacBookPro:~/h2oai/daal/__release_lnx_gnu/daal/examples/java$

  > The command builds executables `*.class` (for example, `CholeskyBatch.class`) in the
  > `daal/examples/java/com/intel/daal/examples/<example name>` directory.
  > 3. Run examples:
  >
  > Go to the Java examples directory and execute the launcher command with the run parameter:
  >
  >     cd <install dir>/daal/examples/java
  >     launcher.sh {ia32|intel64} run $PATH_TO_JAVAC
  >
In my case:

    mathemage@mathemage-MacBookPro:~/h2oai/daal/__release_lnx_gnu/daal/examples/java$ ./launcher.sh intel64 run /usr/bin/javac 
    11:00:18 PASSED         AssocRulesAprioriBatch
    11:00:18 PASSED         BrownBoostDenseBatch
    11:00:18 PASSED         AdaBoostDenseBatch
    11:00:19 PASSED         LogitBoostDenseBatch
    11:00:19 PASSED         CholeskyDenseBatch
    11:00:19 PASSED         CompressorExample
    11:00:20 PASSED         CompressionBatch
    11:00:20 PASSED         CompressionOnline
    11:00:20 PASSED         CovDenseBatch
    11:00:20 PASSED         CovDenseDistr
    11:00:20 PASSED         CovDenseOnline
    11:00:21 PASSED         CovCSRBatch
    11:00:21 PASSED         CovCSRDistr
    11:00:21 PASSED         CovCSROnline
    11:00:21 PASSED         DataStructuresHomogen
    11:00:21 PASSED         DataStructuresHomogenTensor
    11:00:22 PASSED         DataStructuresAOS
    11:00:22 PASSED         DataStructuresSOA
    11:00:22 PASSED         DataStructuresCSR
    11:00:22 PASSED         DataStructuresMerged
    11:00:22 PASSED         DataStructuresMatrix
    11:00:23 PASSED         DataStructuresPackedSymmetric
    11:00:23 PASSED         DataStructuresPackedTriangular
    11:00:23 PASSED         DataStructuresRowMerged
    11:00:25 PASSED         DfClsDenseBatch
    11:00:26 PASSED         DfClsTraverseModel
    11:00:26 PASSED         DfRegDenseBatch
    11:00:26 PASSED         DfRegTraverseModel
    11:00:27 PASSED         DtClsDenseBatch
    11:00:27 PASSED         DtRegDenseBatch
    11:00:27 PASSED         CorDistDenseBatch
    11:00:27 PASSED         CosDistDenseBatch
    11:00:28 PASSED         EmGmmDenseBatch
    11:00:28 PASSED         KDTreeKNNDenseBatch
    11:00:28 PASSED         KMeansDenseBatch
    11:00:28 PASSED         KMeansCSRBatch
    11:00:29 PASSED         KMeansDenseBatchAssign
    11:00:29 PASSED         KMeansInitDenseBatch
    11:00:29 PASSED         KMeansCSRBatchAssign
    11:00:29 PASSED         KMeansDenseDistr
    11:00:30 PASSED         KMeansCSRDistr
    11:00:30 PASSED         KMeansInitDenseDistr
    11:00:31 PASSED         KMeansInitCSRDistr
    11:00:31 PASSED         LinRegNormEqDenseBatch
    11:00:31 PASSED         LinRegNormEqDenseDistr
    11:00:31 PASSED         LinRegNormEqDenseOnline
    11:00:32 PASSED         LinRegQRDenseBatch
    11:00:32 PASSED         LinRegQRDenseDistr
    11:00:32 PASSED         LinRegQRDenseOnline
    11:00:32 PASSED         LowOrderMomsDenseBatch
    11:00:32 PASSED         LowOrderMomsDenseDistr
    11:00:33 PASSED         LowOrderMomsDenseOnline
    11:00:33 PASSED         LowOrderMomsCSRBatch
    11:00:33 PASSED         LowOrderMomsCSRDistr
    11:00:33 PASSED         LowOrderMomsCSROnline
    11:00:34 PASSED         MnNaiveBayesDenseBatch
    11:00:34 PASSED         MnNaiveBayesCSRBatch
    11:00:34 PASSED         MnNaiveBayesDenseOnline
    11:00:34 PASSED         MnNaiveBayesDenseDistr
    11:00:35 PASSED         MnNaiveBayesCSROnline
    11:00:35 PASSED         MnNaiveBayesCSRDistr
    11:00:35 PASSED         OutDetectUniDenseBatch
    11:00:36 PASSED         OutDetectBaconDenseBatch
    11:00:36 PASSED         OutDetectMultDenseBatch
    11:00:36 PASSED         PCACorDenseBatch
    11:00:36 PASSED         PCACorDenseDistr
    11:00:36 PASSED         PCACorDenseOnline
    11:00:37 PASSED         PCACorCSRBatch
    11:00:37 PASSED         PCACorCSRDistr
    11:00:37 PASSED         PCACorCSROnline
    11:00:37 PASSED         PCASVDDenseBatch
    11:00:38 PASSED         PCASVDDenseDistr
    11:00:38 PASSED         PCASVDDenseOnline
    11:00:38 PASSED         QRDenseBatch
    11:00:39 PASSED         QRDenseDistr
    11:00:39 PASSED         QRDenseOnline
    11:00:39 PASSED         RidgeRegNormEqDenseBatch
    11:00:40 PASSED         RidgeRegNormEqDenseOnline
    11:00:40 PASSED         RidgeRegNormEqDistr
    11:00:40 PASSED         SerializationExample
    11:00:40 PASSED         StumpDenseBatch
    11:00:41 PASSED         SVDDenseBatch
    11:00:41 PASSED         SVDDenseDistr
    11:00:41 PASSED         SVDDenseOnline
    11:00:41 PASSED         SVMMultiClassDenseBatch
    11:00:42 PASSED         SVMMultiClassCSRBatch
    11:00:43 PASSED         SVMTwoClassDenseBatch
    11:00:44 PASSED         SVMTwoClassCSRBatch
    11:00:44 PASSED         LibraryVersionInfoExample
    11:00:44 PASSED         QuantilesDenseBatch
    11:00:45 PASSED         PivotedQRDenseBatch
    11:00:45 PASSED         LinRegMetricsDenseBatch
    11:00:46 PASSED         SVMTwoClassMetricsDenseBatch
    11:00:46 PASSED         SVMMultiClassMetricsDenseBatch
    11:00:47 PASSED         KernelFuncLinDenseBatch
    11:00:47 PASSED         KernelFuncLinCSRBatch
    11:00:47 PASSED         KernelFuncRbfDenseBatch
    11:00:47 PASSED         KernelFuncRbfCSRBatch
    11:00:47 PASSED         ImplAlsCSRBatch
    11:00:48 PASSED         ImplAlsCSRDistr
    11:00:48 PASSED         ImplAlsDenseBatch
    11:00:48 PASSED         SetNumberOfThreads
    11:00:49 PASSED         DataStructuresMerged
    11:00:49 PASSED         SortingDenseBatch
    11:00:49 PASSED         ErrorHandling
    11:00:49 PASSED         SoftmaxDenseBatch
    11:00:50 PASSED         AbsDenseBatch
    11:00:50 PASSED         AbsCSRBatch
    11:00:50 PASSED         SmoothReLUDenseBatch
    11:00:50 PASSED         LogisticDenseBatch
    11:00:50 PASSED         ReLUCSRBatch
    11:00:51 PASSED         ReLUDenseBatch
    11:00:51 PASSED         TanhDenseBatch
    11:00:51 PASSED         TanhCSRBatch
    11:00:51 PASSED         ZScoreDenseBatch
    11:00:51 PASSED         MinMaxDenseBatch
    11:00:52 PASSED         SGDDenseBatch
    11:00:52 PASSED         SGDMiniDenseBatch
    11:00:52 PASSED         SGDMomentDenseBatch
    11:00:52 PASSED         SGDMomentOptResDenseBatch
    11:00:53 PASSED         LBFGSDenseBatch
    11:00:53 PASSED         LBFGSOptResDenseBatch
    11:00:53 PASSED         AdagradDenseBatch
    11:00:53 PASSED         AdagradOptResDenseBatch
    11:00:53 PASSED         MSEDenseBatch
    11:00:54 PASSED         ReLULayerDenseBatch
    11:00:54 PASSED         ReshapeLayerDenseBatch
    11:00:54 PASSED         AbsLayerDenseBatch
    11:00:54 PASSED         BatchNormLayerDenseBatch
    11:00:55 PASSED         DropoutLayerDenseBatch
    11:00:55 PASSED         FullyconLayerDenseBatch
    11:00:55 PASSED         SplitLayerDenseBatch
    11:00:56 PASSED         SmoothReLULayerDenseBatch
    11:00:56 PASSED         SoftmaxLayerDenseBatch
    11:00:56 PASSED         PReLULayerDenseBatch
    11:00:57 PASSED         TanhLayerDenseBatch
    11:00:57 PASSED         LRNLayerDenseBatch
    11:00:57 PASSED         Conv2DLayerDenseBatch
    11:00:57 PASSED         TransConv2DLayerDenseBatch
    11:00:57 PASSED         LogisticLayerDenseBatch
    11:00:58 PASSED         AvePool1DLayerDenseBatch
    11:00:58 PASSED         MaxPool1DLayerDenseBatch
    11:00:58 PASSED         AvePool2DLayerDenseBatch
    11:00:58 PASSED         MaxPool2DLayerDenseBatch
    11:00:59 PASSED         StochPool2DLayerDenseBatch
    11:00:59 PASSED         AvePool3DLayerDenseBatch
    11:00:59 PASSED         MaxPool3DLayerDenseBatch
    11:00:59 PASSED         ConcatLayerDenseBatch
    11:00:59 PASSED         LossSoftmaxEntrLayerDenseBatch
    11:01:00 PASSED         LossLogisticEntrLayerDenseBatch
    11:01:00 PASSED         NeuralNetDenseBatch
    11:01:00 PASSED         NeuralNetPredicDenseBatch
    11:01:00 PASSED         Locallycon2DLayerDenseBatch
    11:01:01 PASSED         LCNLayerDenseBatch
    11:01:01 PASSED         SpatAvePool2DLayerDenseBatch
    11:01:01 PASSED         SpatMaxPool2DLayerDenseBatch
    11:01:01 PASSED         SpatStochPool2DLayerDenseBatch
    11:01:02 PASSED         NeuralNetDenseDistr
    11:01:02 PASSED         InitializersDenseBatch
    11:01:02 PASSED         UniformDenseBatch
    11:01:02 PASSED         NormalDenseBatch

  > Choose the same architecture parameter as you provided to the `daalvars.sh` script.
  > The output for each example is written to the file `<example name>.res` located in the `./_results/ia32 or ./_results/intel64` directory, depending on the specified architecture.
  
In my case for *kmeans*:

    mathemage@mathemage-MacBookPro:~/h2oai/daal/__release_lnx_gnu/daal/examples/java/_results/intel64/kmeans$ cat KMeansDenseDistr.res 
    First 10 cluster assignments from 1st node:
    9,000    
    2,000    
    10,000   
    6,000    
    2,000    
    0,000    
    19,000   
    9,000    
    16,000   
    12,000   
    
    First 10 dimensions of centroids:
    28,444   69,338   -19,093   -84,508   -62,736   -75,870   71,257   -3,230   90,362   91,283   
    -1,587   57,712   -69,800   6,939    0,767    -97,030   24,695   57,525   -60,300   10,854   
    9,997    -9,940   14,924   -12,928   29,625   -8,136   -38,225   -16,525   14,812   46,411   
    -17,099   -70,045   -36,512   -87,732   -81,019   -77,647   1,536    59,940   10,108   9,260    
    41,014   4,046    93,251   75,132   88,100   -56,137   -1,563   -18,380   99,718   -27,253   
    -60,972   55,569   -56,923   63,789   -98,431   9,136    39,535   9,164    1,274    -76,180   
    29,018   -71,785   34,412   61,973   2,323    93,152   58,346   -72,968   95,721   69,131   
    95,934   -24,241   100,191   -72,032   -99,445   52,278   28,387   42,388   -31,064   31,566   
    31,405   73,528   -51,049   -2,325   -48,781   37,337   -79,049   21,605   -57,648   28,843   
    31,456   49,779   31,556   65,321   65,147   19,129   22,527   -40,798   20,346   44,843   
    -95,370   9,115    52,739   74,720   86,138   -24,872   -50,892   -70,787   -80,180   -92,828   
    46,964   53,593   -52,974   -39,790   -70,001   71,254   -47,509   -98,568   -61,581   -78,374   
    11,605   -21,225   -74,744   98,645   -96,515   -57,990   81,264   70,134   -38,341   -21,786   
    -61,472   25,601   -71,051   29,644   -30,970   88,035   64,038   34,698   46,026   -39,366   
    -27,267   -70,020   -35,709   -88,781   -77,864   -78,816   -1,979   24,031   11,320   7,931    
    -13,312   -68,303   -38,313   -90,097   -81,847   -77,311   0,028    74,674   10,366   17,586   
    -63,519   -33,372   35,277   50,594   -59,256   -84,848   -36,420   50,814   22,489   74,710   
    0,228    88,671   -51,474   10,444   0,405    -94,135   27,030   56,224   -58,386   13,389   
    7,768    -20,950   -45,879   97,497   -95,093   -57,047   82,718   73,297   -39,379   -22,131   
    62,344   -27,571   97,339   47,101   -39,448   -18,952   -40,233   -59,481   69,412   77,490   
    
    Objective function value:
    169077568,000   
    


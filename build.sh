infra=../daal-infra/bin
 
cc=clang
 
source $infra/env/lnx32e.sh build icc_clang
 
algorithm="pca" 
oneapi_alg="decision_forest"
 
make daal oneapi COMPILER=$cc CORE.ALGORITHMS.CUSTOM=$algorithm -j56 1>_build_lnx32e_${cc}.log 2>&1 &

# !/bin/bash

current_dir=`dirname ${BASH_SOURCE[0]}`

if [ "${DAALROOT}" == "" ]; then
    echo "DAALROOT is undefined, setup oneDAL env and try again"
    exit
fi

if [ "$(uname)" == "Linux" ]; then
    os=lnx
else
    os=mac
fi

daal_lib_dir_lnx=${DAALROOT}/lib/intel64
daal_lib_dir_mac=${DAALROOT}/lib
daal_lib_dir=daal_lib_dir_${os}

compiler_lnx=g++
compiler_mac=clang++
compiler=compiler_${os}

link_dynamic_par="-ldaal_core -ldaal_thread"
link_dynamic_seq="-ldaal_core -ldaal_sequential"
link_static_par="${!daal_lib_dir}/libdaal_core.a ${!daal_lib_dir}/libdaal_thread.a"
link_static_seq="${!daal_lib_dir}/libdaal_core.a ${!daal_lib_dir}/libdaal_sequential.a"

run() {
    local linking=$1
    local threading=$2
    local out=${current_dir}/compat_${linking}_${threading}
    local daal_link_line=link_${linking}_${threading}
    local cmd=${!compiler}
    cmd+=" -w ${current_dir}/compat.cpp"
    cmd+=" -I${DAALROOT}/include -L${!daal_lib_dir}"
    cmd+=" ${!daal_link_line} -ltbb -ltbbmalloc -pthread -ldl -o${out}"
    echo "${cmd}"
    local msg="PASSED   ${compiler} ${linking} ${threading}"
    ${cmd} && ./${out} && echo ${msg} && rm -rf ${out}
    echo
}

for l in dynamic static ; do
    for t in par seq ; do
        run $l $t
    done
done

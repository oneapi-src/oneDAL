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

daal_lib=${DAALROOT}/lib/intel64

compiler_lnx=g++
compiler_mac=clang++
compiler=compiler_${os}

link_dynamic_par="-ldaal_core -ldaal_thread"
link_dynamic_seq="-ldaal_core -ldaal_sequential"
link_static_par="$daal_lib/libdaal_core.a $daal_lib/libdaal_thread.a"
link_static_seq="$daal_lib/libdaal_core.a $daal_lib/libdaal_sequential.a"

run() {
    local linking=$1
    local threading=$2
    local out=${current_dir}/compat_${linking}_${threading}
    local daal_link_line=link_${linking}_${threading}
    ${!compiler} -w ${current_dir}/compat.cpp \
        -I${DAALROOT}/include \
        -L${DAALROOT}/lib/intel64 \
        ${!daal_link_line} \
        -ltbb -ltbbmalloc -pthread -ldl \
        -o${out}
    ./${out} && echo "PASSED   ${compiler} ${linking} ${threading}"
    rm -rf ${out}
}

for l in dynamic static ; do
    for t in par seq ; do
        run $l $t
    done
done

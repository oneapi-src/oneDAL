# !/bin/bash

current_dir=`dirname ${BASH_SOURCE[0]}`

run() {
    local compiler=$1
    local linking=$2
    local threading=$3
    local out=${current_dir}/compat_${linking}_${threading}
    ${compiler} -w ${current_dir}/compat.cpp \
        -I${DAALROOT}/include \
        -Wl,-B${linking} -ldaal_core \
        -Wl,-B${linking} -ldaal_${threading} \
        -Wl,-Bdynamic -ltbb -ltbbmalloc -pthread -ldl \
        -o${out}
    ./${out}
    echo "PASSED   ${compiler} ${linking} ${threading}"
    rm -rf ${out}
}

if [ "$(uname)" == "Linux" ]; then
    CC=g++
else
    CC=clang++
fi

for l in dynamic static ; do
    for t in sequential thread ; do
        run $CC $l $t
    done
done

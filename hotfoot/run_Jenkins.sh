#!/bin/bash

# this function looks for errors or test failures
parse_output() {

ERRORS=`grep -l "[error|killed]:" ${OUTPUT}/$1/stderr/*`
if [ -z "$ERRORS" ]
then
  echo "$1 has build, link, etc. errors"
  cat ${ERRORS}
  exit 10
fi

FAILURES=`grep -l -F -i "fail" ${OUTPUT}/$1/stdout/*
if [ -z "$FAILURES" ]
then
  echo "$1 has test failures at runtime"
  cat ${FAILURES}
  echo "Possible warnings associated with test failures"
  cat ${FAILURES//stdout/stderr}
  exit 20
fi

exit 0
}

# All environmental variables are exported when calling qsub
# These first two are supposedly exported by Jenkins
export GIT_COMMIT=`git rev-parse HEAD`
export GIT_BRANCH=`git rev-parse --abbrev-ref HEAD`
export STAN_HOME=/hpc/stats/projects/stan

export CCACHE_SLOPPINESS=include_file_mtime
export CCACHE_DIR=/hpc/tmp/stan/.ccache/
mkdir -p ${CCACHE_DIR}
CC="ccache clang++ -Qunused-arguments"

cd ${STAN_HOME}

# Tweak CFLAGS but revert on exit
if [[ ${GIT_BRANCH} != "master" && ${GIT_BRANCH} != hotfix* && ${GIT_BRANCH} != release* ]]
then
  sed -i 's@^CFLAGS =@CFLAGS = --std=c++11 -DGTEST_HAS_PTHREAD=0 -pedantic -Wextra @' makefile
  sed -i '/-Wno/d' make/os_linux
fi

# No need to shuffle tests on Hotfoot
git revert 83e1b2eed4298ba0cd2b519bce7fe25289440df7
trap "git reset --hard HEAD" EXIT

mkdir -p hotfoot/${GIT_BRANCH}/${GIT_COMMIT}
export OUTPUT=${STAN_HOME}/hotfoot/${GIT_BRANCH}/${GIT_COMMIT}
trap "rm -rf hotfoot/${GIT_BRANCH}" EXIT

alias QSUB='qsub -W group_list=hpcstats -l nodes=1,mem=2gb -M bg2382@columbia.edu -m n -V -I -q batch1'

CODE=1
# Create dependencies of all tests
make clean-all > /dev/null
if [ $? -ne 0 ]
then
  echo "make clean-all failed; aborting"
  exit ${CODE}
fi

((CODE++))
nice make CC=${CC} -j4 test/libgtest.a
if [ $? -ne 0 ]
then
  echo "make test/libgtest.a failed; aborting"
  exit ${CODE}
fi

((CODE++))
nice make CC=${CC} -j4 bin/libstan.a
if [ $? -ne 0 ]
then
  echo "make bin/libstan.a failed; aborting"
  exit ${CODE}
fi

((CODE++))
QSUB -N stanc -l walltime=0:00:04:59 -x "bash hotfoot/stanc.sh"
if [ ! -e "bin/stanc" ]
then
  cat ${OUTPUT}/stanc_stdout.txt
  cat ${OUTPUT}/stanc_stderr.txt
  echo "make bin/stanc failed; aborting"
  exit ${CODE}
fi

((CODE++))
nice make CC=${CC} -j4 src/test/agrad/distributions/generate_tests
if [ $? -ne 0 ]
then
  echo "make generate_tests failed; aborting"
  exit ${CODE}
fi

# test-headers
TARGET='test-headers'
mkdir -p ${OUTPUT}/${TARGET}/stdout/
mkdir -p ${OUTPUT}/${TARGET}/stderr/

find src/stan/ -type f -name "*.hpp" -print | \
sed 's@.hpp@.pch@g' > ${OUTPUT}/tests.txt

TEST_MAX=`wc -l ${OUTPUT}/tests.txt | cut -f1 -d ' '`
TEST_MAX=`expr ${TEST_MAX} - 1`
# FIXME: enable this
#QSUB -N "${TARGET}" -t 0-${TEST_MAX} -l walltime=0:00:00:29 -x "bash hotfoot/test.sh"
#CODE = parse_output "${TARGET}"
#[ ${CODE} -ne 0 ] && exit ${CODE}

# test-libstan
TARGET='test-libstan'
mkdir -p ${OUTPUT}/${TARGET}/stdout/
mkdir -p ${OUTPUT}/test-libstan/stderr/

find src/test/ -type f -name "*_test.cpp" -print | \
grep -v -F "src/test/models" |
sed 's@src/@@g' | sed 's@_test.cpp@@g' > ${OUTPUT}/tests.txt

TEST_MAX=`wc -l ${OUTPUT}/tests.txt | cut -f1 -d ' '`
TEST_MAX=`expr ${TEST_MAX} - 1`
QSUB -N "${TARGET}" -t 0-${TEST_MAX} -l walltime=0:00:00:59 -x "bash hotfoot/test.sh"
CODE = parse_output "${TARGET}"
[ ${CODE} -ne 0 ] && exit ${CODE}

# test-gm
TARGET='test-gm'
mkdir -p ${OUTPUT}/${TARGET}/stdout/
mkdir -p ${OUTPUT}/${TARGET}/stderr/

find src/test/gm/model_specs/compiled -type f -name "*.stan" -print | \
sed 's@.stan@@g' > ${OUTPUT}/tests.txt

TEST_MAX=`wc -l ${OUTPUT}/tests.txt | cut -f1 -d ' '`
TEST_MAX=`expr ${TEST_MAX} - 1`
QSUB -N "${TARGET}" -t 0-${TEST_MAX} -l walltime=0:00:01:59 -x "bash hotfoot/test.sh"
CODE = parse_output "${TARGET}"
[ ${CODE} -ne 0 ] && exit ${CODE}

# test-models
TARGET='test-models'
mkdir -p ${OUTPUT}/${TARGET}/stdout/
mkdir -p ${OUTPUT}/${TARGET}/stderr/

find src/test/models -type f -name "*_test.cpp" -print | \
sed 's@src/@@g' | sed 's@_test.cpp@@g' > ${OUTPUT}/tests.txt

TEST_MAX=`wc -l ${OUTPUT}/tests.txt`
TEST_MAX=`expr ${TEST_MAX} - 1`
QSUB -N "${TARGET}" -t 0-${TEST_MAX} -l walltime=0:00:4:59 -x "bash hotfoot/test.sh"
CODE = parse_output "${TARGET}"
[ ${CODE} -ne 0 ] && exit ${CODE}

#unset TEST_ARRAY
#unset i
#while IFS= read -r -d $'\0' file; do
#    TEST=${file/"src/stan/prob"/"test/agrad"}
#    TEST=${TEST/".hpp"/""}
#    TEST_FILE=src/${TEST}_test.hpp
#    if [ ! -e  $TEST_FILE ]
#      then continue
#    fi
#    ARGS=`grep "^// Arguments:" ${TEST_FILE} | sed 's@// Arguments: @@' | wc -w`
#    let "N_BATCHES=$[(8 ** $ARGS) / 100]"
#    for ((j=0;j<=N_BATCHES;j++)); do
#      NUM=`printf "_%05d_generated" ${j}`
#      TEST_ARRAY[i++]=${TEST}${NUM}
#    done
#done < <(find src/stan/prob/distributions/univariate/ -type f -name "*.hpp" -print0)

# test-distributions
TARGET='test-distributions'
mkdir -p ${OUTPUT}/${TARGET}/stdout/
mkdir -p ${OUTPUT}/${TARGET}/stderr/

find src/test/agrad/distributions/univariate/ -type f -name "*_test.hpp" -print | \
grep -v -F "cdf" | \
sed 's@src/@@g' | sed 's@_test.hpp@_00000_generated@g' > ${OUTPUT}/tests.txt

# FIXME: generate rest of distribution tests

TEST_MAX=`wc -l ${OUTPUT}/tests.txt`
TEST_MAX=`expr ${TEST_MAX} - 1`
QSUB -N "${TARGET}" -t 0-${TEST_MAX} -l walltime=0:00:00:40 -x "bash hotfoot/test.sh"
CODE = parse_output "${TARGET}"
[ ${CODE} -ne 0 ] && exit ${CODE}

# Finish up
echo "All tests passed on Hotfoot for branch ${GIT_BRANCH} and commit ${GIT_COMMIT}"
echo "But the following are all the unique lines with warnings:"
grep -r -h -F "warning:" ${OUTPUT} --exclude-dir=*stdout/ | grep ^src | sort | uniq
echo "The walltimes of the tests were:"
cat ${OUTPUT}/test_timings.txt

make clean-all > /dev/null
exit 0

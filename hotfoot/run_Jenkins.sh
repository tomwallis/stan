#!/bin/bash

# this function looks for errors or test failures
parse_output() {

ERRORS=`grep -l -e "error:" -e "killed:" -e "Aborted" ${OUTPUT}/$1/stderr/*`
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

# this function prepares the ingrediants for calling qsub
# call it AFTER setting up ${OUTPUT}/tests.txt
setup() {
SO=${OUTPUT}/${1}/stdout/
SE=${OUTPUT}/${1}/stderr/
mkdir -p ${SO}
mkdir -p ${SE}
TEST_MAX=`wc -l ${OUTPUT}/tests.txt | cut -f1 -d ' '`
TEST_MAX=`expr ${TEST_MAX} - 1`
exit 0
}

# All environmental variables are exported when calling qsub
# These first two are supposedly exported by Jenkins
export GIT_COMMIT=`git rev-parse --short HEAD`
export GIT_BRANCH=`git rev-parse --abbrev-ref HEAD`
export STAN_HOME=/hpc/stats/projects/stan

# prepare ccache (not working well yet)
export CCACHE_LOGFILE=${STAN_HOME}/.ccache/logfile.txt
export CCACHE_SLOPPINESS=include_file_mtime
export CCACHE_DIR=${STAN_HOME}/.ccache
mkdir -p ${CCACHE_DIR}
export CC="ccache clang++ -Qunused-arguments"

cd ${STAN_HOME}

# Tweak CFLAGS for feature/* branches and develop
if [[ ${GIT_BRANCH} != "master" && ${GIT_BRANCH} != hotfix* && ${GIT_BRANCH} != release* ]]
then
  sed -i 's@^CFLAGS =@CFLAGS = --std=c++11 -DGTEST_HAS_PTHREAD=0 -pedantic -Wextra @' makefile
  sed -i '/-Wno/d' make/os_linux
fi

# No need to shuffle tests on Hotfoot
git revert --no-edit --no-commit 83e1b2eed4298ba0cd2b519bce7fe25289440df7
trap "git reset --hard HEAD" EXIT

# make a directory for test output
export OUTPUT=${STAN_HOME}/hotfoot/${GIT_BRANCH}/${GIT_COMMIT}
mkdir -p ${OUTPUT}
trap "rm -rf hotfoot/${GIT_BRANCH}" EXIT

# make an alias with default arguments to qsub
alias QSUB='qsub -W group_list=hpcstats -l mem=2gb -M bg2382@columbia.edu -m n -V'
# in general we have to wait for the job array to finish, so
# note the ugly while loop that follows almost all QSUB calls

# Create dependencies of all tests (mostly) using submit node
CODE=1
make clean-all > /dev/null
if [ $? -ne 0 ]
then
  echo "make clean-all failed; aborting"
  exit ${CODE}
fi

((CODE++))
nice make CC="${CC}" -j4 test/libgtest.a
if [ $? -ne 0 ]
then
  echo "make test/libgtest.a failed; aborting"
  exit ${CODE}
fi

((CODE++))
nice make CC="${CC}" -j4 bin/libstan.a
if [ $? -ne 0 ]
then
  echo "make bin/libstan.a failed; aborting"
  exit ${CODE}
fi

# make bin/stanc overwhelms the submit node
# so use an execute node with 5 processors
# this blocks until finished, so no ugly while loop
((CODE++))
QSUB -N stanc -l nodes=1:ppn=5 -l walltime=0:00:04:59 -I -q batch1 -x "bash hotfoot/stanc.sh"
if [ ! -e "bin/stanc" ]
then
  cat ${OUTPUT}/stanc_stdout.txt
  cat ${OUTPUT}/stanc_stderr.txt
  echo "make bin/stanc failed; aborting"
  exit ${CODE}
fi

((CODE++))
nice make CC="${CC}" -j4 src/test/agrad/distributions/generate_tests
# FIXME: Generate all the distribution tests at this point
if [ $? -ne 0 ]
then
  echo "make generate_tests failed; aborting"
  exit ${CODE}
fi

# test-headers
find src/stan/ -type f -name "*.hpp" -print | \
sed 's@.hpp@.pch@g' > ${OUTPUT}/tests.txt

TARGET='test-headers'
setup ${TARGET}

# FIXME: enable this
#QSUB -N "${TARGET}" -t 0-${TEST_MAX} -l walltime=0:00:00:29 \
#-o localhost:${SO} -e localhost:${SE} hotfoot/test.sh
#while [ $(ls ${SO} | wc -l) -le ${TEST_MAX} ]; do sleep 10; done
#CODE = parse_output "${TARGET}"
#[ ${CODE} -ne 0 ] && exit ${CODE}

# test-libstan
find src/test/ -type f -name "*_test.cpp" -print | \
grep -v -F "src/test/models" | # exclude tests under src/test/models
sed 's@src/@@g' | sed 's@_test.cpp@@g' > ${OUTPUT}/tests.txt

TARGET='test-libstan'
setup ${TARGET}

QSUB -N "${TARGET}" -t 0-${TEST_MAX} -l walltime=0:00:01:59 \
-o localhost:${SO} -e localhost:${SE} hotfoot/test.sh
while [ $(ls ${SO} | wc -l) -le ${TEST_MAX} ]; do sleep 10; done
CODE = parse_output "${TARGET}"
[ ${CODE} -ne 0 ] && exit ${CODE}

# test-gm
find src/test/gm/model_specs/compiled -type f -name "*.stan" -print | \
sed 's@.stan@@g' > ${OUTPUT}/tests.txt

TARGET='test-gm'
setup ${TARGET}

QSUB -N "${TARGET}" -t 0-${TEST_MAX} -l walltime=0:00:04:59 \
-o localhost:${SO} -e localhost:${SE} hotfoot/test.sh
while [ $(ls ${SO} | wc -l) -le ${TEST_MAX} ]; do sleep 10; done
CODE = parse_output "${TARGET}"
[ ${CODE} -ne 0 ] && exit ${CODE}

# test-models
find src/test/models -type f -name "*_test.cpp" -print | \
sed 's@src/@@g' | sed 's@_test.cpp@@g' > ${OUTPUT}/tests.txt

TARGET='test-models'
setup ${TARGET}

QSUB -N "${TARGET}" -t 0-${TEST_MAX} -l walltime=0:00:5:59 \
-o localhost:${SO} -e localhost:${SE} hotfoot/test.sh
while [ $(ls ${SO} | wc -l) -le ${TEST_MAX} ]; do sleep 10; done
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
find src/test/agrad/distributions/ -type f -name "*_test.hpp" -print | \
sed 's@src/@@g' | sed 's@_test.hpp@_00000_generated@g' > ${OUTPUT}/tests.txt
# FIXME: generate rest of distribution tests above

TARGET='test-distributions'
setup ${TARGET}

QSUB -N "${TARGET}" -t 0-${TEST_MAX} -l walltime=0:00:01:59 \
-o localhost:${SO} -e localhost:${SE} hotfoot/test.sh
while [ $(ls ${SO} | wc -l) -le ${TEST_MAX} ]; do sleep 10; done
CODE = parse_output "${TARGET}"
[ ${CODE} -ne 0 ] && exit ${CODE}

# success so finish up
echo "All tests passed on Hotfoot for branch ${GIT_BRANCH} and commit ${GIT_COMMIT}"
echo "But the following are all the unique warnings from Stan"
grep -r -h -F "warning:" --exclude-dir=*stdout/ ${OUTPUT} | grep ^src | sort | uniq
echo "The walltimes of the tests were:"
cat ${OUTPUT}/test_timings.txt

make clean-all > /dev/null
exit 0

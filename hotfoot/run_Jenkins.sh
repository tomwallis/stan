#!/bin/bash

# this function looks for errors or test failures
parse_output() {

ERRORS=`grep -l -F "error:" ${OUTPUT}/$1/stderr/*`
if [ -z "$ERRORS" ]
then
  echo "$1 has build, link, etc. errors"
  echo ${ERRORS}
  exit 10
fi

FAILURES=`grep -l -F -i "fail" ${OUTPUT}/$1/stdout/*
if [ -z "$FAILURES" ]
then
  echo "$1 has test failures at runtime"
  echo ${FAILURES}
  echo "Possible warnings associated with test failures"
  echo ${FAILURES//stdout/stderr}
  exit 20
fi

exit 0
}

# All environmental variables are exported when calling qsub
# These two are supposedly exported by Jenkins
GIT_COMMIT=`git rev-parse HEAD`
GIT_BRANCH=`git rev-parse --abbrev-ref HEAD`
STAN_HOME=/hpc/stats/projects/stan

CCACHE_SLOPPINESS=include_file_mtime
CCACHE_DIR=/tmp/stan/.ccache/
CC="ccache clang++ -Qunused-arguments"

cd ${STAN_HOME}

# Tweak CFLAGS but revert on exit
if [[ ${GIT_BRANCH} != "master" && ${GIT_BRANCH} != hotfix* && ${GIT_BRANCH} != release* ]]
then
  sed -i 's@^CFLAGS =@CFLAGS = --std=c++11 -DGTEST_HAS_PTHREAD=0 -pedantic -Wextra @' makefile
  sed -i '/-Wno/d' make/os_linux
  trap "git reset --hard HEAD" EXIT
fi

CODE=1
# Create dependencies of tests
make clean-all &> /dev/null
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
nice make CC=${CC} -j4 bin/stanc
if [ $? -ne 0 ]
then
  echo "make bin/stanc failed; aborting"
  exit ${CODE}
fi

OUTPUT=${STAN_HOME}/hotfoot/${GIT_BRANCH}/${GIT_COMMIT}
trap "rm -rf hotfoot/$GIT_BRANCH}" EXIT

# Gather *_test.cpp files under src/test/, excluding src/test/models/
unset TEST_ARRAY
unset i
while IFS= read -r -d $'\0' file; do
  if [[ "$file" != *src/test/models/* ]]
  then
    TEST=${file#"src/"}
    TEST_ARRAY[i++]=${TEST%"_test.cpp"}
  fi
done < <(find src/test/ -type f -name "*_test.cpp" -print0)

# Execute these tests individually but in parallel using a job array
TUSO=${OUTPUT}/test-unit/stdout
TUSE=${OUTPUT}/test-unit/stderr
mkdir -p ${TUSO}
mkdir -p ${TUSE}

TEST_ARRAY_MAX=`expr ${#TEST_ARRAY[@]} - 1`
qsub -N 'test-unit' -t 0-${TEST_ARRAY_MAX} -l walltime 0:00:00:30 \
-o localhost:${TUSO} -e localhost:${TUSE} -I -x hotfoot/test.sh
CODE = parse_output()
[ ${CODE} -ne 0 ] && exit ${CODE}

# Gather model tests
unset TEST_ARRAY
unset i
while IFS= read -r -d $'\0' file; do
  TEST=${file#"src/"}
  TEST_ARRAY[i++]=${TEST%"_test.cpp"}
done < <(find src/test/models -type f -name "*_test.cpp" -print0)

# Execute them
TMSO=${OUTPUT}/test-models/stdout
TMSE=${OUTPUT}/test-models/stderr
mkdir -p ${TDSO}
mkdir -p ${TDSE}

TEST_ARRAY_MAX=`expr ${#TEST_ARRAY[@]} - 1`
qsub -N 'test-models' -t 0-${TEST_ARRAY_MAX} -l walltime=0:00:10:00 \
-o localhost:${TMSO} -e localhost:${TMSE} -I -x hotfoot/test.sh
CODE = parse_output()
[ ${CODE} -ne 0 ] && exit ${CODE}

# Gather distribution tests
unset TEST_ARRAY
unset i
while IFS= read -r -d $'\0' file; do
    TEST=${file/"src/stan/prob"/"test/agrad"}
    TEST=${TEST/".hpp"/""}
    TEST_FILE=src/${TEST}_test.hpp
    if [ ! -e  $TEST_FILE ]
      then continue
    fi
    ARGS=`grep "^// Arguments:" ${TEST_FILE} | sed 's@// Arguments: @@' | wc -w`
    let "N_BATCHES=$[(8 ** $ARGS) / 100]"
    for ((j=0;j<=N_BATCHES;j++)); do
      NUM=`printf "_%05d_generated" ${j}`
      TEST_ARRAY[i++]=${TEST}${NUM}
    done
done < <(find src/stan/prob/distributions/univariate/ -type f -name "*.hpp" -print0)

# Execute them
TDSO=${OUTPUT}/test-distributions/stdout
TDSE=${OUTPUT}/test-distributions/stderr
mkdir -p ${TDSO}
mkdir -p ${TDSE}

TEST_ARRAY_MAX=`expr ${#TEST_DISTRIBUTIONS_ARRAY[@]} - 1`
# temporarily disable execution of tests
#qsub -N 'test-distributions' -t 0-${TEST_ARRAY_MAX} -l walltime=0:00:00:40 \
#-o localhost:${TDSO} -e localhost${TDSE} -I -x hotfoot/test.sh
#CODE = parse_output()
#[ ${CODE} -ne 0 ] && exit ${CODE}

echo "All tests passed on Hotfoot for branch ${GIT_BRANCH} and commit ${GIT_COMMIT}"
echo "But the following are all the unique lines with warnings:"
grep -r -h -F "warning:" ${OUTPUT} \
--exclude-dir=${TUSO} --exclude-dir=${TMSO} --exclude-dir=${TDSO}  | grep ^src | sort | uniq
echo "The walltimes of the tests were:"
cat $OUTPUT/*_timings.txt

make clean-all > /dev/null
exit 0

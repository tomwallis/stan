#!/bin/sh -e
cd ${STAN_HOME}
make CC="${CC}" -j5 bin/stanc > ${OUTPUT}/stanc_stdout.txt 2> ${OUTPUT}/stanc_stderr.txt
exit 0

#!/bin/sh
# Overcome the issues with Python's bz2 by decompressing and the compresssing
# the data.

#set -x

# Collect absolute paths to binaries.

mv='/bin/mv'
find='/usr/bin/find'
cat='/bin/cat'
basename='/bin/basename'
echo='/bin/echo'
touch='/bin/touch'
grep='/bin/grep'
sed='/bin/sed'

bzip2='/usr/bin/bzip2'

# Remember files that we have converted.

processed_log='processed.log'

# On any signal, just quit.
trap "exit 1" SIGHUP SIGINT SIGTERM

usage() {
  script_name=`${basename} $0`
  ${cat} <<EOF
Usage: ${script_name} DATA_DIR

  Uncompress and then compress all files in DATA_DIR using bzip2.

  DATA_DIR    data file directory
EOF
}

# Check arguments.
if [ $# -ne 1 ]; then
  ${echo} Invalid arguments 1>&2
  usage
  exit 1
fi

data_dir=$1; shift


# Check that data directory is valid.
if [ ! -d "${data_dir}" ]; then
  ${echo} Invalid data dir: \"${data_dir}\" 2>&2
  exit 1
fi

pushd "${data_dir}" >/dev/null

# Make sure processed log file exists.
${touch} ${processed_log}

# Get list of files that have been rolled over (have a dot in them).
for file in `${find} ./ -type f -name '*.*' | ${sed} 's/^\.\///'`; do

  # Filter ignored file types.
  if ${echo} "${file}" | ${grep} -Eq '(\.bz2$|\.tmp$|\.log$)' \
    || ${echo} "${file}" | ${grep} -f ${processed_log}; then
    echo "Skipping ${file}"
    continue
  fi

  ${echo} "Re-compressing ${file}"
  tmp_file="${file}.tmp"
  ${bzip2} -cd "${file}" >"${tmp_file}"
  ${bzip2} -z "${tmp_file}"
  ${mv} "${tmp_file}.bz2" "${file}"

  # Add processed record.
  ${echo} "${file}" >> ${processed_log}
done

popd >/dev/null

exit 0

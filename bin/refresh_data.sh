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

bzip2='/usr/bin/bzip2'
grep='/bin/grep'

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

# Get list of files that have been rollwed over (have a dot in them).
for file in `${find} "${data_dir}" -type f -name '*.*'`; do
  if ${echo} "${file}" | ${grep} -Eq '(\.bz2$|\.tmp$)'; then
    continue
  fi
  ${echo} "Re-compressing ${file}"
  tmp_file="${file}.tmp"
  ${bzip2} -cd "${file}" >"${tmp_file}"
  ${bzip2} -z "${tmp_file}"
  ${mv} "${tmp_file}.bz2" "${file}"
done

exit 0

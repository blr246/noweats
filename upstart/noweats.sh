#!/bin/bash

COLLECT_HOME="/home/reissb/noweats_data"

VENV_HOME="/home/reissb/.virtualenvs/noweats"

REMOTE_PATH="brandonr@brandonreiss.com:~/public_html/noweats/data/"

script_dir=`dirname $0`
pushd "${script_dir}/.." >/dev/null
noweats_home=`pwd`
bin_dir="`pwd`/bin"
popd >/dev/null

cleanup() {
  kill $!
  exit 0
}

usage() {
cat <<EOF
usage: $(basename $0) [-i]

Start the noweats upstart job.

   -i  install the virtualenv
   -h  show help
EOF
}

try_start_collect() {
  if ! pgrep 'collect_nyc' >/dev/null 2>&1
  then
    # Collect tweets in the background.
    collect_nyc ${COLLECT_HOME}/collect 2>&1 | /usr/bin/logger -t noweats &
  fi
}

run() {
    trap "cleanup" SIGINT SIGTERM EXIT

    /bin/mkdir -p ${COLLECT_HOME}/collect
    /bin/mkdir -p ${COLLECT_HOME}/output
    . "${VENV_HOME}/bin/activate"
    # Process tweets every 5 minutes. This helps when we fall behind.
    while true
    do
        try_start_collect
        /usr/bin/nice -n 19 process_new "${COLLECT_HOME}/collect" "${COLLECT_HOME}/output"
        /usr/bin/nice -n 19 /usr/bin/rsync -r "${COLLECT_HOME}/output/" "${REMOTE_PATH}"
        sleep 300.0
    done
}

setup_venv() {
    # Create virtualenv, install noweats, and setup nltk.
    virtualenv -p `which pypy` "${VENV_HOME}"
    . "${VENV_HOME}/bin/activate"
    pip install -r "${noweats_home}/requirements.txt" "${noweats_home}"
    "${bin_dir}/link_numpy"
    python -mnltk.downloader punkt
}

# Parse arguments and run requested mode.
if (( $# >= 1 ))
then
    if (( $# == 1 )) && [[ "${@}" == "-i" ]]
    then
        setup_venv
    else
        usage
        exit 0
    fi
else
    run
fi

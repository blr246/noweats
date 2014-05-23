#!/bin/bash

cleanup() {
  kill $!
  exit 0
}

try_start_collect() {
  if ! pgrep 'collect_nyc' >/dev/null 2>&1; then
    # Collect tweets in the background.
    collect_nyc /home/reissb/noweats_data/collect 2>&1 | /usr/bin/logger -t noweats &
  fi
}

trap "cleanup" SIGINT SIGTERM EXIT

/bin/mkdir -p /home/reissb/noweats_data/collect
. /home/reissb/.virtualenvs/noweats/bin/activate
# Process tweets every 5 minutes. This helps when we fall behind.
while true; do
  try_start_collect
  /usr/bin/nice -n 19 process_new \
    /home/reissb/noweats_data/collect /home/reissb/noweats_data/output
  /usr/bin/rsync -cr /home/reissb/noweats_data/output/ \
    brandonr@brandonreiss.com:~/public_html/noweats/data/
  sleep 300.0
done


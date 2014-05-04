#!/bin/bash

pid_collect=

cleanup() {
  kill $!
  exit 0
}

trap "cleanup" SIGINT SIGTERM EXIT

/bin/mkdir -p /home/reissb/noweats_data/collect
. /home/reissb/.virtualenvs/noweats/bin/activate
# Collect tweets in the background.
collect_nyc /home/reissb/noweats_data/collect &
# Process tweets every 5 minutes. This helps when we fall behind.
while true; do
  /usr/bin/nice -n 19 process_new \
    /home/reissb/noweats_data/collect /home/reissb/noweats_data/output
  /usr/bin/rsync -cr /home/reissb/noweats_data/output/ \
    brandonr@brandonreiss.com:~/public_html/noweats/data/
  sleep 300.0
done


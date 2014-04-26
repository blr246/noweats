#!/usr/bin/env python
"""
Collect data for NYC.
"""
from argparse import ArgumentParser, FileType
import json
import os

from noweats.data_collection import ApiKeys, StreamFilterRunner


def main():

    bin_dir = os.path.dirname(os.path.abspath(__file__))
    default_conf_dir = os.path.abspath(os.path.join(bin_dir, "../"))
    default_conf = os.path.join(default_conf_dir, 'twitter.conf')

    parser = ArgumentParser(description="Collect Twitter stream data for NYC.")
    parser.add_argument('-a', '--auth', help="Path to Twitter API keys",
                        type=FileType('r'), default=default_conf)

    args = parser.parse_args()

    apikeys = ApiKeys(json.load(args.auth))

if __name__ == '__main__':
    main()

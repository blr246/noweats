noweats
=======

What are people eating (on Twitter)?

Confguration
------------
The noweats configuration lives in `~/.noweats` and consists of a
`twitter.conf` file containing API keys and a `filters.conf` file containing
pruning filters for tweets. See `conf/filters.conf` for an example.

Installation
------------
It is recommended to use noweats in a virtualenv. To install it, run

    $ mkvirtualenv noweats
    $ workon noweats
    $ python setup.py install

This should install everything you need into the `noweats` virtualenv.

Usage
-----
To collect data from Twitter, create your `twitter.conf` file and then run a
command like

    $ collect_nyc DATA_DIR

where `DATA_DIR` is the location where you would like to save Twitter stream
data.

To process files into their top food and interesting foods, run

    $ process_fil DATA_FILE [DATA_FILE...]

where `DATA_FILE` are paths to files output by the `collect_nyc` script.

Note that due to known issues in the Python `bz2` library, all files must be
recompressed using the `recompress_data` script.

All of the libraries supporting the scrips are also usable by importing them.
See the code under the `noweats` package for more information.

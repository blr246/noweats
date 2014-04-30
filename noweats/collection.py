"""
Objects used to access Twitter streaming APIs.
"""
from copy import copy
from logging.handlers import TimedRotatingFileHandler
from logging import getLogger, LogRecord
from tweepy.streaming import StreamListener
from tweepy.error import TweepError
from tweepy import Stream, OAuthHandler

import os
import sys

_LOGGER = getLogger(__name__)


class ApiKeys(object):
    """ Helper to get an OAuthHandler for an API key dict. """

    _KEYNAME_API_KEY = 'api_key'
    _KEYNAME_API_SECRET = 'api_secret'
    _KEYNAME_ACCESS_TOKEN = 'access_token'
    _KEYNAME_ACCESS_SECRET = 'access_secret'

    def __init__(self, keys):

        if self._KEYNAME_API_KEY not in keys \
                or self._KEYNAME_API_SECRET not in keys \
                or self._KEYNAME_ACCESS_TOKEN not in keys \
                or self._KEYNAME_ACCESS_SECRET not in keys:
            raise ValueError('Keys dict does not contain api key/secret '
                             'and access token/secret')

        self._api_key = keys[self._KEYNAME_API_KEY]
        self._api_secret = keys[self._KEYNAME_API_SECRET]
        self._access_token = keys[self._KEYNAME_ACCESS_TOKEN]
        self._access_secret = keys[self._KEYNAME_ACCESS_SECRET]
        self._auth = OAuthHandler(self._api_key, self._api_secret)
        self._auth.set_access_token(self._access_token, self._access_secret)

    @property
    def auth(self):
        """ Get OAuthHandler for API. """
        auth = OAuthHandler(self._api_key, self._api_secret)
        auth.set_access_token(self._access_token, self._access_secret)
        return auth

    @property
    def api_key(self):
        """ Get the API key. """
        return copy(self._api_key)

    @property
    def api_secret(self):
        """ Get the API secret. """
        return copy(self._api_secret)

    @property
    def access_token(self):
        """ Get the access token. """
        return copy(self._access_token)

    @property
    def access_secret(self):
        """ Get the access secret. """
        return copy(self._access_secret)


class TimedRotatingStreamListener(StreamListener):
    """
    A Twitter stream listener that writes to compressed, timed rotating file.

    N.B. Uses the logger encoding 'bz2' for compression of raw JSON.
    """

    def __init__(self, log_dir, prefix, when_interval=None):
        """
        Init with output path and name prefix.

        Default rollover interval is four hours.

        :param str log_dir: Where to write logged data.
        :param str prefix: Filename prefix for logged data.
        :param tuple when_interval: Listener rollover interval. See
        https://docs.python.org/2.7/library/logging.handlers.html
        #logging.handlers.TimedRotatingFileHandler for more information.
        """

        log_dir = os.path.abspath(log_dir)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        if not isinstance(prefix, str):
            raise ValueError("Prefix string required for output files")

        if when_interval is None:
            when, interval = 'H', 1
        else:
            when, interval = when_interval
            if not (isinstance(when, str) and isinstance(interval, int)):
                raise ValueError("Rollover interval should be like ('h', 1)")

        log_path = os.path.join(log_dir, prefix)

        _LOGGER.info("Saving data to path {}".format(log_path))
        _LOGGER.info("Rollover interval {}"
                     .format("{}{}".format(when, interval)))

        # Create the logger.
        self._logger = TimedRotatingFileHandler(filename=log_path,
                                                encoding='bz2_codec',
                                                when=when,
                                                interval=interval)

    def on_data(self, data):
        """ Log stream data. """
        # Skip keep-alive newlines.
        data_stripped = data.strip()
        if len(data) > 0:
            record = LogRecord(None, None, None, None, data_stripped, (), None)
            self._logger.emit(record)

    def on_error(self, status):
        """ Print status to stderr and raise exception. """
        sys.stderr.write('{}\n'.format(status))
        raise TweepError(status)


class StreamFilterRunner(object):
    """ Run a stream filter. """

    class _MyStreamFilter(object):
        """ Wrapper used to run and to close a stream. """

        def __init__(self, runner):

            self._listener = TimedRotatingStreamListener(runner.log_dir,
                                                         runner.prefix,
                                                         runner.when_interval)
            self._stream = Stream(runner.auth, self._listener)
            self._track = runner.track
            self._locations = runner.locations

        def run(self):
            """ Run the collector. """
            self._stream.filter(track=self._track, locations=self._locations)

        def close(self):
            """ Close the collector. """
            self._stream.disconnect()
            self._listener.flush()
            self._listener.close()

    def __init__(self, api_keys, track, locations,
                 log_dir, prefix, when_interval=None):
        """
        Run a stream filter.

        :param ApiKeys keys: The keys used for OAuth.
        :param list track: List of phrases to track.
        :param list locations: List of location coordinates to filter.
        :param str log_dir: Directory used to write tweets.
        :param str prefix: File output prefix.
        :param str when: Output file rollover interval.
        """
        self._api_keys = api_keys
        self._track = track
        self._locations = locations
        self._log_dir = log_dir
        self._prefix = prefix
        self._when_interval = when_interval
        self._stream_filter = None

    def __enter__(self):
        if self._stream_filter is None:
            self._stream_filter = self._MyStreamFilter(self)
            self._stream_filter.run()
        else:
            raise ValueError("Cannot start again")

    def __exit__(self, valtype, value, traceback):
        self._stream_filter.close()
        return False  # do not suppress exceptions

    @property
    def auth(self):
        """ Get API auth. """
        return self._api_keys.auth

    @property
    def track(self):
        """ Get filter track. """
        return self._track

    @property
    def locations(self):
        """ Get filter locations. """
        return self._locations

    @property
    def log_dir(self):
        """ Get output directory. """
        return self._log_dir

    @property
    def prefix(self):
        """ Get output prefix. """
        return self._prefix

    @property
    def when_interval(self):
        """ Get output interval. """
        return self._when_interval

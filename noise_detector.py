"""
Detect noise changes in the environment

:on_noise str
    The command to run when noise is detected

:-v --trigger-volume float 0.9
    The average abnormality exceeded that is considered an activation. Between 0.0 and 1.0

:-d --delay float 1.0
    The delay in seconds between subsequent activations

:-b --bands str -
    A range/list of bands to listen for
"""
from collections import deque

import json
import numpy as np
import os
import re
from logzero import logger
from prettyparse import create_parser
from pylisten import FeatureListener
from sonopy import power_spec, safe_log
from subprocess import Popen


def parse_bands(s, m):
    s = re.sub(r'\s', '', s)
    if ',' in s:
        return sum([parse_bands(i, m) for i in s.split(',')], [])
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a or 0), int(b or m) + 1))
    return [int(s)]


def get_bands(args, parser, num_filt):
    if args.bands:
        try:
            return parse_bands(args.bands, num_filt)
        except ValueError:
            parser.error('Invalid brands expression')
            return
    else:
        return list(range(num_filt))


def averager_gen(adjustment_speed=0.01):
    averages = feature = yield
    av_diff = np.zeros_like(feature)

    while True:
        diff = feature - averages
        av_diff += adjustment_speed * (diff ** 2 - av_diff)
        averages += adjustment_speed * diff
        av_freq_stds = np.sqrt(av_diff)
        freq_stds = diff / np.clip(av_freq_stds, np.finfo(float).eps, None)
        feature = yield freq_stds


class BandCounter:
    def __init__(self, memory):
        self.memory = memory
        self.coords = deque()
        self.counts = {}

    def update(self, coord: np.ndarray) -> float:
        """Returns the abnormality of this band as a float from 0.0 to 1.0"""
        k = self._key(coord)
        self.coords.append(k)
        existing_count = self.counts.get(k, 0)
        self.counts[k] = existing_count + 1
        if len(self.coords) > self.memory:
            ok = self.coords.popleft()
            nv = self.counts[ok] - 1
            if nv:
                self.counts[ok] = nv
            else:
                del self.counts[ok]
        return 1.0 - existing_count / self.memory

    def _key(self, x):
        return (x / self._calc_space()).astype(int).data.tobytes()

    def _calc_space(self):
        return 1.5  # TODO: Periodically regenerate this and recalculate counts


class SpecCounter:
    def __init__(self, memory, bands: list):
        self.bands = {band: BandCounter(memory) for band in bands}

    def update(self, frames):
        """Returns a dict mapping bands to the abnormality ratio"""
        coords = frames.T
        return {band: counter.update(coords[band]) for band, counter in self.bands.items()}


def main():
    parser = create_parser(__doc__)

    sample_rate = 16000
    stride = 400
    width = 4
    delay = 6.0
    num_bands = 257
    chunk_seconds = stride / sample_rate
    averager = averager_gen()
    next(averager)

    args = parser.parse_args()
    bands = get_bands(args, parser, num_bands)

    def processor(audio):
        log_spec = safe_log(power_spec(
            audio, window_stride=(2 * stride, stride),
            fft_size=512
        ))
        return np.array([averager.send(i) for i in log_spec] or log_spec)

    counts = SpecCounter(200, bands)
    for features in FeatureListener(processor, stride, width, dict(rate=sample_rate)):
        abnormalities = counts.update(features)
        av_abnormality = sum(abnormalities.values()) / len(abnormalities)

        if delay > 0:
            delay -= chunk_seconds
            if delay <= 0:
                logger.info('Listening...')
        else:
            if av_abnormality >= args.trigger_volume:
                logger.info('Activation of {:.2f}'.format(av_abnormality))
                Popen(args.on_noise, shell=True, env=dict(
                    os.environ,
                    VOLUME='{:.2f}'.format(av_abnormality),
                    BANDS=json.dumps(abnormalities, sort_keys=True)
                ))
                delay = args.delay


if __name__ == '__main__':
    main()

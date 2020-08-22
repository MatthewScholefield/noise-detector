from argparse import ArgumentParser
from base64 import b64encode, b64decode
from collections import deque

import math

import json
import numpy as np
import os
import re
from logzero import logger
from prettyparse import Usage
from pylisten import FeatureListener
from sonopy import power_spec, safe_log
from subprocess import Popen

usage = Usage('A tool to detect noise changes in the environment')
detect_usage = Usage('''
    Detect noise changes in the environment
    
    ...
    
    :-v --trigger-volume float 0.9
        The average abnormality exceeded that is considered an activation. Between 0.0 and 1.0
    
    :-d --delay float 1.0
        The delay in seconds between subsequent activations
    
    :-b --bands str -
        A range/list of bands to listen for
    
    :-m --model str -
        A model file to use
    
    :-s --memory-size int 200
        Number of samples to save in memory
''')
detect_usage.add_argument('command', nargs='?', default='true', help='The command to run when noise is detected')
collect_usage = Usage('''
    Collect noise and save into a model
    
    :noise_model str
        Noise model json file to write to
    
    :-b --bands str -
        A range/list of bands to listen for
''')


def add_subparsers(parser: ArgumentParser):
    sp = parser.add_subparsers(dest='action')
    sp.required = True
    detect_usage.apply(sp.add_parser('detect'))
    collect_usage.apply(sp.add_parser('collect'))


usage.add_customizer(add_subparsers)

sample_rate = 16000
stride = 400
width = 4
num_bands = 257
chunk_seconds = stride / sample_rate


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
            self._remove_one_coord()
        return 1.0 - existing_count / self.memory

    def remember_noise(self, coord: np.ndarray):
        """Remembers given coordinate as normal coordinates"""
        k = self._key(coord)
        self.counts[k] = self.counts.get(k, 0) + 1

    def clear_state(self):
        """Forgets recent coordinates"""
        while self.coords:
            self._remove_one_coord()

    def log_squash(self, base):
        self.counts = {k: math.log(count, base) for k, count in self.counts.items()}

    def serialize(self):
        return {'memory': self.memory, 'counts': {b64encode(k).decode(): v for k, v in self.counts.items()}}

    @classmethod
    def from_json(cls, obj):
        counter = cls(obj['memory'])
        counter.counts = {b64decode(k): v for k, v in obj['counts'].items()}
        return counter

    def _key(self, x):
        return (x / self._calc_space()).astype(int).data.tobytes()

    def _calc_space(self):
        return 1.5  # TODO: Periodically regenerate this and recalculate counts

    def _remove_one_coord(self):
        ok = self.coords.popleft()
        nv = self.counts[ok] - 1
        if nv:
            self.counts[ok] = nv
        else:
            del self.counts[ok]


class SpecCounter:
    def __init__(self, memory, bands: list):
        self._memory = memory
        self.bands = {band: BandCounter(memory) for band in bands}

    def update(self, frames):
        """Returns a dict mapping bands to the abnormality ratio"""
        coords = frames.T
        return {band: counter.update(coords[band]) for band, counter in self.bands.items()}

    def clear_state(self):
        for counter in self.bands.values():
            counter.clear_state()

    def remember_noise(self, frames):
        """Remembers incoming audio as background noise"""
        coords = frames.T
        for band, counter in self.bands.items():
            counter.remember_noise(coords[band])

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        for band in self.bands.values():
            band.memory = value
        self._memory = value

    def serialize(self):
        return {str(band): counter.serialize() for band, counter in self.bands.items()}

    @classmethod
    def from_json(cls, obj):
        instance = cls(0, [])
        instance.bands = {int(band): BandCounter.from_json(data) for band, data in obj.items()}
        return instance


def main():
    parser = ArgumentParser()
    usage.apply(parser)
    args = usage.render_args(parser.parse_args())

    averager = averager_gen()
    next(averager)

    bands = get_bands(args, parser, num_bands)

    def processor(audio):
        log_spec = safe_log(power_spec(
            audio, window_stride=(2 * stride, stride),
            fft_size=512
        ))
        return np.array([averager.send(i) for i in log_spec] or log_spec)

    listener = FeatureListener(processor, stride, width, dict(rate=sample_rate))

    if args.action == 'detect':
        if args.model and args.bands:
            parser.error('Cannot specify bands when using a noise model')
        run_detect(args, listener, bands)
    elif args.action == 'collect':
        run_collect(args, listener, bands)
    else:
        raise RuntimeError


def run_detect(args, listener, bands):
    iterator = iter(listener)
    logger.info('Collecting ambient noise...')
    delay = chunk_seconds * args.memory_size

    if args.model:
        with open(args.model) as f:
            counts = SpecCounter.from_json(json.load(f))
            counts.memory = args.memory_size
    else:
        counts = SpecCounter(args.memory_size, bands)

    for features in iterator:
        abnormalities = counts.update(features)
        av_abnormality = sum(abnormalities.values()) / len(abnormalities)

        if delay > 0:
            delay -= chunk_seconds
            if delay <= 0:
                logger.info('Listening...')
        else:
            if av_abnormality >= args.trigger_volume:
                logger.info('Activation of {:.2f}'.format(av_abnormality))
                Popen(args.command, shell=True, env=dict(
                    os.environ,
                    VOLUME='{:.2f}'.format(av_abnormality),
                    BANDS=json.dumps(abnormalities, sort_keys=True)
                ))
                delay = args.delay


def run_collect(args, listener, bands):
    # Ensure writable noise model
    with open(args.noise_model, 'w') as f:
        f.write('{}')

    counts = SpecCounter(0, bands)
    try:
        iterator = iter(listener)
        logger.info('Collecting (press ctrl+c to end)...')
        for features in iterator:
            counts.remember_noise(features)
    except KeyboardInterrupt:
        print()
        logger.info('Saving model...')
        with open(args.noise_model, 'w') as f:
            json.dump(counts.serialize(), f)


if __name__ == '__main__':
    main()

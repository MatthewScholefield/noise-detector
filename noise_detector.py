from argparse import ArgumentParser
from base64 import b64encode, b64decode
from collections import deque, namedtuple

import math

import json
from typing import Callable, Iterable

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
    
    :model str
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

DetectionData = namedtuple('DetectionData', 'volume bands')


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
        self.num_saved_chunks = 0
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
        self.num_saved_chunks += 1

    def clear_state(self):
        """Forgets recent coordinates"""
        while self.coords:
            self._remove_one_coord()

    def log_squash(self, base):
        self.counts = {k: math.log(count, base) for k, count in self.counts.items()}

    def serialize(self):
        return {
            'memory': self.memory,
            'num_saved_chunks': self.num_saved_chunks,
            'counts': {b64encode(k).decode(): v for k, v in self.counts.items()}
        }

    @classmethod
    def from_json(cls, obj):
        counter = cls(obj['memory'])
        counter.counts = {b64decode(k): v for k, v in obj['counts'].items()}
        counter.num_saved_chunks = obj['num_saved_chunks']
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


class NoiseDetector:
    def __init__(self, bands: list = None, memory_size=200, model: str = None):
        self.bands = bands or list(range(num_bands))
        self.model = model

        if model:
            with open(model) as f:
                self.counter = SpecCounter.from_json(json.load(f))
                self.counter.memory = memory_size
        else:
            self.counter = SpecCounter(memory_size, self.bands)

    def default_listener(self):
        averager = averager_gen()
        next(averager)

        def processor(audio):
            log_spec = safe_log(power_spec(
                audio, window_stride=(2 * stride, stride),
                fft_size=512
            ))
            return np.array([averager.send(i) for i in log_spec] or log_spec)

        return FeatureListener(processor, stride, width, dict(rate=sample_rate))

    def listen(self, on_detection: Callable, on_listening=lambda: None, trigger_volume=0.8, activation_delay=0.25,
               listener: Iterable = None):
        listener = listener or self.default_listener()
        delay = chunk_seconds * self.counter.memory

        for features in listener:
            abnormalities = self.counter.update(features)
            av_abnormality = sum(abnormalities.values()) / len(abnormalities)

            if delay > 0:
                delay -= chunk_seconds
                if delay <= 0:
                    on_listening()
            else:
                if av_abnormality >= trigger_volume:
                    on_detection(DetectionData(av_abnormality, abnormalities))
                    delay = activation_delay

    def collect(self, listener: Iterable = None):
        listener = listener or self.default_listener()
        for features in listener:
            self.counter.remember_noise(features)

    def save(self):
        with open(self.model, 'w') as f:
            json.dump(self.counter.serialize(), f)


def run_detect(args, detector):
    iterator = iter(detector.default_listener())
    logger.info('Collecting ambient noise...')

    def on_detection(detection_data: DetectionData):
        logger.info('Activation of {:.2f}'.format(detection_data.volume))
        Popen(args.command, shell=True, env=dict(
            os.environ,
            VOLUME='{:.2f}'.format(detection_data.volume),
            BANDS=json.dumps(detection_data.bands, sort_keys=True)
        ))

    detector.listen(
        on_detection,
        on_listening=lambda: logger.info('Listening...'),
        trigger_volume=args.trigger_volume,
        activation_delay=args.delay,
        listener=iterator
    )


def run_collect(args, detector):
    # Ensure writable noise model
    with open(detector.model, 'w') as f:
        f.write('{}')

    iterator = iter(detector.default_listener())
    try:
        logger.info('Collecting (press ctrl+c to end)...')
        detector.collect(iterator)
    except KeyboardInterrupt:
        print()
        logger.info('Saving model...')
        detector.save()


def main():
    parser = ArgumentParser()
    usage.apply(parser)
    args = usage.render_args(parser.parse_args())

    detector = NoiseDetector(
        bands=get_bands(args, parser, num_bands),
        memory_size=args.memory_size if args.action == 'detect' else 0,
        model=args.model
    )

    if args.action == 'detect':
        if args.model and args.bands:
            parser.error('Cannot specify bands when using a noise model')
        run_detect(args, detector)
    elif args.action == 'collect':
        run_collect(args, detector)
    else:
        raise RuntimeError


if __name__ == '__main__':
    main()

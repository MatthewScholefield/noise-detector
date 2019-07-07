"""
Detect noise changes in the environment

:on_noise str
    The command to run when noise is detected

:-v --trigger-volume float 3.5
    The number of standard deviations from the average volume to consider a noise change

:-a --adjustment-speed float 0.01
    The speed at which the ambient noise level changes. Between 0 and 1

:-d --delay float 1.0
    The delay in seconds between subsequent activations

:-b --bands str -
    A range/list of bands to listen for
"""
import os
import re
from functools import partial
from subprocess import Popen

import numpy as np
from logzero import logger
from prettyparse import create_parser
from pylisten import FeatureListener
from sonopy import mel_spec


def parse_bands(s, m):
    s = re.sub(r'\s', '', s)
    if ',' in s:
        return sum([parse_bands(i, m) for i in s.split(',')], [])
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a or 0), int(b or m) + 1))
    return [int(s)]


def main():
    parser = create_parser(__doc__)

    sample_rate = 16000
    stride = 500
    width = 2
    num_filt = 20
    averages = None
    av_diff = None
    chunk_seconds = stride / sample_rate
    delay = 2.0

    args = parser.parse_args()
    try:
        bands = parse_bands(args.bands, num_filt)
        erase_bands = np.array([i not in bands for i in range(num_filt)])
    except ValueError:
        parser.error('Invalid brands expression')
        return
    print(bands)

    processor = partial(
        mel_spec, sample_rate=sample_rate, window_stride=(2 * stride, stride),
        fft_size=512, num_filt=num_filt
    )
    for features in FeatureListener(processor, stride, width, dict(rate=sample_rate)):
        feature = features[0] + features[1]  # Average across two frames

        if averages is None:
            logger.info('Calculating ambient noise...')
            averages = feature
            av_diff = np.zeros_like(feature)
        else:
            diff = feature - averages
            av_diff += args.adjustment_speed * (diff ** 2 - av_diff)
            averages += args.adjustment_speed * diff

            if delay > 0:
                delay -= chunk_seconds
                if delay <= 0:
                    logger.info('Listening...')
            else:
                av_freq_stds = np.sqrt(av_diff)
                freq_stds = diff / av_freq_stds
                max_band = np.argmax(np.where(erase_bands, float('-inf'), freq_stds))
                max_freq_std = freq_stds[max_band]
                if max_freq_std > args.trigger_volume and max_band in bands:
                    logger.info('Activation of {:.2f} on band {}!'.format(max_freq_std, max_band))
                    Popen(args.on_noise, shell=True, env=dict(
                        os.environ,
                        VOLUME='{:.2f}'.format(max_freq_std),
                        BAND=str(max_band),
                        BANDS=' '.join(map(str, freq_stds)),
                        AMBIENT_VOLUME=str(averages[max_band]),
                        AMBIENT_STD=str(av_diff[max_band])
                    ))
                    delay = args.delay


if __name__ == '__main__':
    main()

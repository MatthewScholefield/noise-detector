# Noise Detector

*An ambient noise detector*

This is a simple script that runs a command when the noise in the
environment changes. It computes a running average of the volumes
per frequency band and standard deviations per band to figure out
when there's a change in the environment. When this happens, it
runs the given command with a few extra environment variables.

## Usage

Install with:

```
pip install noise-detector
```

Use like:

```bash
$ noise-detector 'echo "Activation of $VOLUME on band $BAND"' --trigger-volume 3.5
```

You can see a list of all the environment variables available by
looking in the source code.

To reduce false activations, you can specify which bands you would like to listen to:

```bash
$ noise-detector 'echo "Heard middle-range activation"' --bands "4-8"
```

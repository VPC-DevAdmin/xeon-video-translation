"""Keep the animation conditioning copy separate from the audible master."""
from pathlib import Path
import wave


def inspect_audio_pair(conditioning_path, playback_path=None, expected_rate=16000):
    def inspect(path):
        with wave.open(str(path), 'rb') as stream:
            result = {'sample_rate': stream.getframerate(), 'channels': stream.getnchannels(),
                      'sample_width': stream.getsampwidth(), 'frames': stream.getnframes()}
        if result['frames'] <= 0 or result['sample_width'] != 2 or result['channels'] != 1:
            raise ValueError(f'{path}: expected nonempty mono PCM16 WAV')
        result['duration'] = result['frames'] / result['sample_rate']
        return result

    conditioning = inspect(conditioning_path)
    if conditioning['sample_rate'] != expected_rate:
        raise ValueError(f'Animation conditioning must be {expected_rate} Hz')
    playback = inspect(playback_path or conditioning_path)
    # Resampling can round one sample; a perceptible timeline shift is not allowed.
    tolerance = max(1 / conditioning['sample_rate'], 1 / playback['sample_rate']) + 1e-9
    if abs(conditioning['duration'] - playback['duration']) > tolerance:
        raise ValueError('Conditioning and playback duration differ; regenerate conditioning from the final master')
    return {'conditioning': conditioning, 'playback': playback,
            'playback_path': str(Path(playback_path or conditioning_path))}

# based on https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np
import soundfile as sf
import os


class MelGen():
    def __init__(self, hp):
        self.hp = hp

    def get_normalized_mel(self, x):
        x = librosa.feature.melspectrogram(
            y=x,
            sr=self.hp.audio.sr,
            n_fft=self.hp.audio.win_length,
            hop_length=self.hp.audio.hop_length,
            win_length=self.hp.audio.win_length,
            n_mels=self.hp.audio.n_mels
        )
        x = self.pre_spec(x)
        return x

    def reconstruct_audio(self, normalized_mel):
        x = self.post_spec(normalized_mel)
        y = librosa.feature.inverse.mel_to_audio(
          M=x,
          sr=self.hp.audio.sr,
          n_fft=self.hp.audio.win_length,
          hop_length=self.hp.audio.hop_length,
          win_length=self.hp.audio.win_length
        )
        return y

    def pre_spec(self, x):
        return self.normalize(librosa.power_to_db(x) - self.hp.audio.ref_level_db)

    def post_spec(self, x):
        return librosa.db_to_power(self.denormalize(x) + self.hp.audio.ref_level_db)

    def normalize(self, x):
        return np.clip(x / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, x):
        return (np.clip(x, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db

    def save_audio(self, filename, y):
        filename = os.path.basename(filename)
        save_dir = 'reconstructed_audio'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('Saving %s' % filename)
        sf.write('./reconstructed_audio/%s' % filename, y, self.hp.audio.sr, 'PCM_24')
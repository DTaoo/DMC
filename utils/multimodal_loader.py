import numpy as np
import os
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from scipy.io import wavfile
import mel_features
import sound_params as vggish_params
import resampy
from keras.utils import Sequence
import matplotlib.pyplot as plt
import librosa.display
import random

def preprocess_image(image_path):
    try:
        img = image.load_img(image_path)
    except:
        print "Failed to load image."
        img = np.ones(shape=(256,256,3))*255
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img,mode='tf')
    return img

def preprocess_sound(data, sample_rate):
  """Converts audio waveform into an array of examples for VGGish.

  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.

  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate != vggish_params.SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

  # Compute log mel spectrogram features.
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=vggish_params.SAMPLE_RATE,
      log_offset=vggish_params.LOG_OFFSET,
      window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=vggish_params.NUM_MEL_BINS,
      lower_edge_hertz=vggish_params.MEL_MIN_HZ,
      upper_edge_hertz=vggish_params.MEL_MAX_HZ)

  # Frame features into examples.
  features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
      vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
  return log_mel_examples



def process_sound_image(current_videos):

    length = len(current_videos)
    image_data = None
    sound_data = None
    fake_sound_data = None
    pair_label = []
    cosine_ind = []
    k = 0
    for i in range(length):
        #print i

        current_video=current_videos[i]


        ###################### load right video #########################
        video_dir = '/mnt/hudi_disk/frames/' + current_video
        video_dir = video_dir.replace("\n", "")

        if os.path.exists(video_dir):
            # load image for current video
            current_images = os.listdir(video_dir)
            current_images.sort()
            image_num = len(current_images)
            for j in range(image_num):
                current_img_dir = video_dir + "/" + current_images[j]    # j
                current_img_data = preprocess_image(current_img_dir)
                if image_data is None:
                    image_data = current_img_data
                else:
                    image_data = np.concatenate((image_data, current_img_data))


                ###################### load sound #########################
                current_sound_video = current_video
                sound_dir = '/mount/hudi/moe/soundnet_data/mp3/' + current_sound_video
                sound_dir = sound_dir.replace("\n", "")
                sound_file = sound_dir + '.wav'
                try:
                    sr, wav_data = wavfile.read(sound_file)
                except:
                    # maybe the file do not exist (never) or NULL files
                    print "Failed to load sound : %s" % sound_file
                    sr = 16000
                    wav_data = np.zeros((1,), dtype=np.int16)
                assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype

                # make the sound length equal to the video length, each image has 5s duration
                video_length = 5 * image_num * sr
                sound_length = len(wav_data)
                if sound_length < video_length:
                    new_wav_data = np.zeros((video_length,))
                    new_wav_data[0:sound_length] = wav_data
                    wav_data = new_wav_data
                else:
                    wav_data = wav_data[0:video_length]

                samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
                samples = samples[5*j*sr:5*(j+1)*sr]
                current_sound_data = preprocess_sound(samples, sr)

                if sound_data is None:
                    sound_data = current_sound_data
                else:
                    sound_data = np.concatenate((sound_data, current_sound_data))


                ###################### load fake sound #########################
                inc = random.randrange(1, length)
                dn = (i + inc) % length
                current_sound_video = current_videos[dn]
                sound_dir = '/mount/hudi/moe/soundnet_data/mp3/' + current_sound_video
                sound_dir = sound_dir.replace("\n", "")
                sound_file = sound_dir + '.wav'
                try:
                    sr, wav_data = wavfile.read(sound_file)
                except:
                    # maybe the file do not exist (never) or NULL files
                    print "Failed to load sound : %s" % sound_file
                    sr = 16000
                    wav_data = np.zeros((1,), dtype=np.int16)
                assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype

                # make the sound length equal to the video length, each image has 5s duration
                video_length = 5 * image_num * sr
                sound_length = len(wav_data)
                if sound_length < video_length:
                    new_wav_data = np.zeros((video_length,))
                    new_wav_data[0:sound_length] = wav_data
                    wav_data = new_wav_data
                else:
                    wav_data = wav_data[0:video_length]

                samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
                samples = samples[5*j*sr:5*(j+1)*sr]
                current_sound_data = preprocess_sound(samples, sr)

                if fake_sound_data is None:
                    fake_sound_data = current_sound_data
                else:
                    fake_sound_data = np.concatenate((fake_sound_data, current_sound_data))

        else:
            pass


    sound_data = np.expand_dims(sound_data, 3)
    fake_sound_data = np.expand_dims(fake_sound_data, 3)
    dummy_label = np.zeros(len(sound_data))
    return sound_data, image_data, fake_sound_data, dummy_label



class sound_image_generator_cosine(Sequence):
    def __init__(self, videos, batch_size):
        self.videos = videos
        self.batch_size = batch_size

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        # print "idx is %d processing" % idx
        batch_x = self.videos[idx * self.batch_size:(idx + 1) * self.batch_size]

        sound_data, image_data, fake_sound_data, dummy_label  = process_sound_image(batch_x)

        return [sound_data, image_data, fake_sound_data], dummy_label

import torch
import torchaudio
import argparse
import numpy as np
import os
from tqdm import tqdm
from glob import glob


def main(args):
    
    # Define parameters
    num_mels=80
    n_fft=1024
    hop_size=256
    win_size=1024
    sampling_rate=22050
    fmin=0
    fmax=8000
    
    to_stft = torchaudio.transforms.InverseMelScale(n_stft=n_fft//2 + 1, n_mels=num_mels, sample_rate=sampling_rate, f_min=fmin, f_max=fmax).cuda()
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=1024, win_length=win_size, hop_length=hop_size).cuda()
    to_melspectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft, n_mels=num_mels, hop_length=hop_size, win_length=win_size, f_min=fmin, f_max=fmax, pad=int((n_fft-hop_size)/2), center=False).cuda()
    
    # Load mel spectrogram (input/*.npy file or input/*/*.wav files)
    if args.input == 'data/testing_mel':
        bar = tqdm(glob(args.input + '/*.npy'))
    else:
        bar = tqdm(glob(args.input + '/*/*.wav'))
        
    for file in bar:
        bar.set_description(f'Processing {file}')
        if args.input != 'data/testing_mel':
            # Load waveform
            waveform, sr = torchaudio.load(file)
            # Create the MelSpectrogram transform using gpu
            mel_spectrogram = to_melspectrogram(waveform.cuda()).squeeze(0)
            output_filename = args.output + '/' + '-'.join(file.split('/')[-2:])
        else:    
            mel_spectrogram = torch.from_numpy(np.load(file)).cuda()
            output_filename = args.output + '/' + file.split('/')[-1].split('.')[0] + '.wav'
        # Create the InverseMelScale transform using gpu    
        linear_spec = to_stft(mel_spectrogram)
        # Create the GriffinLim transform using gpu
        waveform = griffin_lim(linear_spec)

        # Save waveform
        torchaudio.save(output_filename, waveform.unsqueeze(0).cpu(), sampling_rate)

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/output/valid/Griffin-Lim', choices=['data/output/valid/Griffin-Lim', 'data/output/test/Griffin-Lim'], help='output directory')
    parser.add_argument('--input', default='data/m4singer_valid', choices=['data/m4singer_valid', 'data/testing_mel'], help='input directory')
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    main(args)
    
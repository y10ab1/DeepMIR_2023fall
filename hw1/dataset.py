import torch
import torchaudio
import glob
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, split='train', load_vocals=True, sample_rate=16000, trim_silence=True, do_augment=True, segment_length=5):
        assert split in ['train', 'valid', 'test'], f'Invalid split: {split}, must be one of [train, valid, test]'
        
        self.split = split
        self.load_vocals = load_vocals
        self.sample_rate = sample_rate
        self.trim_silence = trim_silence
        self.do_augment = do_augment
        self.segment_length = segment_length
        
        self.audio_files = []
        self.labels = []
        
        if self.split == 'train':
            dirpath = 'artist20/train_separated'
            for path in glob.glob(f'{dirpath}/*/*/*/*'):
                if self.load_vocals:
                    self.audio_files.append(f'{path}/vocals.mp3')
                else:
                    self.audio_files.append(f'{path}/no_vocals.mp3')
                self.labels.append(path.split('/')[-4])
                
        elif self.split == 'valid':
            dirpath = 'artist20/valid_separated'
            for path in glob.glob(f'{dirpath}/*/*/*/*'):
                if self.load_vocals:
                    self.audio_files.append(f'{path}/vocals.mp3')
                else:
                    self.audio_files.append(f'{path}/no_vocals.mp3')
                self.labels.append(path.split('/')[-4])
                
        elif self.split == 'test':
            dirpath = 'artist20/test_separated'
            for path in glob.glob(f'{dirpath}/*/*/*'):
                if self.load_vocals:
                    self.audio_files.append(f'{path}/vocals.mp3')
                else:
                    self.audio_files.append(f'{path}/no_vocals.mp3')
                self.labels.append(path.split('/')[-1])
                
        # encode labels
        self.label_encoder = {label: i for i, label in enumerate(sorted(set(self.labels)))}
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        print(f'Classes: {self.label_encoder}')
        
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        signal, sr = torchaudio.load(audio_file)
        
        # resample
        signal = torchaudio.transforms.Resample(sr, self.sample_rate)(signal)
        
        # mono
        signal = signal.mean(0, keepdim=True)
        
        # apply voice activity detection (VAD) to trim silence
        if self.trim_silence:
            signal = torchaudio.functional.vad(signal, sample_rate=self.sample_rate)
    
        
        # randomly sample segment_length seconds
        if self.segment_length > 0 and self.split == 'train':
            assert signal.shape[1] > self.sample_rate * self.segment_length, \
                f'Audio signal is too short: {signal.shape[1] / self.sample_rate} < {self.segment_length}'
            
            start = torch.randint(0, signal.shape[1] - self.sample_rate * self.segment_length, (1,)).item()
            signal = signal[:, start:start + self.sample_rate * self.segment_length]

            
            
        # augmentations
        if self.split == 'train' and self.do_augment:
            # random volume
            signal = signal * torch.empty(1).uniform_(0.5, 1.5)
            
            # random noise
            noise = torch.randn_like(signal)
            signal = signal + noise * torch.empty(1).uniform_(0, 0.05)
            
            # random shift
            shift = torch.randint(-self.sample_rate // 4, self.sample_rate // 4, (1,)).item()
            signal = torch.roll(signal, shifts=(shift,), dims=(1,))
            
            # random clipping
            signal = torch.clamp(signal, -1, 1)
        
                
        return signal.squeeze(0), torch.tensor(self.label_encoder[label])
    
    def get_label(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.label_decoder[idx]
    
if __name__ == '__main__':
    print(f'torchaudio backend: {torchaudio.get_audio_backend()}')
    print(f'available effects: {torchaudio.sox_effects.effect_names()}')
    
    dataset = AudioDataset(split='train', load_vocals=True, trim_silence=False)
    print(len(dataset))
    for i in range(5):
        signal, label = dataset[i]
        print(signal.shape, label.shape)
        
        # save to file
        # torchaudio.save(f'nt_{i}.wav', signal, dataset.sample_rate)

    
    
    
    
    
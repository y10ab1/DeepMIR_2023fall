from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')#, run_opts={"device":"cuda:0"})

# for custom file, change path
est_sources = model.separate_file(path='/home/yuehpo/coding/DeepMIR_2023fall/hw1/artist20/train/radiohead/Pablo_Honey/09-Prove_Yourself.mp3') 

torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)


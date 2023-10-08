import torchaudio
import torch
from speechbrain.pretrained import EncoderClassifier #,MelSpectrogramEncoder
spk_emb_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})

INPUT_SPEECH_1 = "artist20/train_seperated/aerosmith/Aerosmith/mdx_extra/01-Make_It/vocals.mp3"
INPUT_SPEECH_2 = "artist20/train_seperated/aerosmith/Aerosmith/mdx_extra/02-Somebody/vocals.mp3"


ref_signal_1, sr_1 = torchaudio.load(INPUT_SPEECH_1)
ref_signal_2, sr_2 = torchaudio.load(INPUT_SPEECH_2)

# resample
ref_signal_1 = torchaudio.transforms.Resample(sr_1, 16000)(ref_signal_1).mean(0, keepdim=True)
ref_signal_2 = torchaudio.transforms.Resample(sr_2, 16000)(ref_signal_2).mean(0, keepdim=True)

print(ref_signal_1.shape, ref_signal_2.shape, 'before')
    
spk_embedding_1 = spk_emb_encoder.encode_batch(ref_signal_1.to("cuda"))
spk_embedding_2 = spk_emb_encoder.encode_batch(ref_signal_2.to("cuda"))

print(spk_embedding_1.shape, spk_embedding_2.shape)
# similarity score
similarity_score = torch.nn.functional.cosine_similarity(spk_embedding_1, spk_embedding_2, dim=-1)
print(similarity_score)
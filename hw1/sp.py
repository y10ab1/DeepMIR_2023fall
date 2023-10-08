import torchaudio
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
# score, prediction = verification.verify_files("speechbrain/spkrec-ecapa-voxceleb/example1.wav", "speechbrain/spkrec-ecapa-voxceleb/example2.flac")
score, prediction = verification.verify_files("artist20/train_seperated/aerosmith/Aerosmith/mdx_extra/01-Make_It/vocals.mp3", 
                                              "artist20/train_seperated/aerosmith/Aerosmith/mdx_extra/02-Somebody/vocals.mp3",
                                              )

# score, prediction = verification.verify_files("artist20/train_seperated/aerosmith/Aerosmith/mdx_extra/01-Make_It/vocals.mp3", 
#                                               "artist20/train_seperated/cure/Disintegration/mdx_extra/03-Closedown/vocals.mp3",
#                                               )

# score, prediction = verification.verify_files("artist20/train_seperated/aerosmith/Aerosmith/mdx_extra/02-Somebody/vocals.mp3",
#                                               "artist20/train_seperated/aerosmith/Aerosmith/mdx_extra/02-Somebody/vocals.mp3",
#                                               )

# score, prediction = verification.verify_files("artist20/train_seperated/aerosmith/Aerosmith/mdx_extra/02-Somebody/vocals.mp3",
#                                               "artist20/train_seperated/aerosmith/Draw_the_Line/mdx_extra/01-Draw_the_Line/vocals.mp3",
#                                               )




print(score, prediction)
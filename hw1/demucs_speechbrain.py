# Assume that your command is `demucs --mp3 --two-stems vocals -n mdx_extra "track with space.mp3"`
# The following codes are same as the command above:
import demucs.separate
filepath = "/home/yuehpo/coding/DeepMIR_2023fall/hw1/artist20/train/radiohead/Pablo_Honey/09-Prove_Yourself.mp3"
ret = demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", filepath])

print(ret)
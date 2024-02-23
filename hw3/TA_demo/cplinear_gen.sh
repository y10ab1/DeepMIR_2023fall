save_dir=/home/yuehpo/coding/DeepMIR_2023fall/hw3/cplinear_genwave/gen_midis_30
root_dir=/home/yuehpo/coding/compound-word-transformer/workspace/uncond/cp-linear/gen_midis_30

# midi2audio $root_dir/get_0.mid $save_dir/get_0.wav

# iterate all midis
for i in {0..19}
do
    midi2audio $root_dir/get_$i.mid $save_dir/get_$i.wav
done
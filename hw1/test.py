import torch
import argparse
import os
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm
from dataset import AudioDataset
from model import SingerClassifier

def main(args):
    model = SingerClassifier(num_classes=20)

    val_dataset = AudioDataset(split='valid', load_vocals=True, sample_rate=16000, trim_silence=True, do_augment=False)
    
    test_dataset = AudioDataset(split='test', load_vocals=True, sample_rate=16000, trim_silence=True, do_augment=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model)['model_state_dict'])
        print(f'Load model from {args.load_model}')
    
    model.eval()
    
    # Inferece to get top 3 predictions for each test sample, and save to submission.csv
    # e.g. idx, top1, top2, top3
    df = pd.DataFrame(columns=['id', 'top1', 'top2', 'top3'])
    with torch.no_grad():
        logger = tqdm(test_dataloader)
        for data, target in logger:
            
            # segment data(audio) into non-overlapping 5s segments
            # e.g. data.shape = (1, 80000), segment_length = 5s, segment_length * sample_rate = 80000
            for i in tqdm(range(data.shape[1] // (args.segment_length * args.sample_rate))):
                # predict each segment and aggregate the results for this sample
                start = i * args.segment_length * args.sample_rate
                end = (i + 1) * args.segment_length * args.sample_rate
                output = model(data[:, start:end].to(model.device))
                if i == 0:
                    predicted = F.softmax(output, dim=1)
                else:
                    predicted += F.softmax(output, dim=1)
                
                    
            # get top 3 predictions for each sample in the batch
            for i in range(predicted.shape[0]):
                top3 = torch.topk(predicted, 3, dim=1)
                
                original_idx = test_dataset.get_label(target[i].item())
                df = df._append({'id': original_idx, 
                                 'top1': val_dataset.get_label(top3.indices[i][0]), 
                                 'top2': val_dataset.get_label(top3.indices[i][1]), 
                                 'top3': val_dataset.get_label(top3.indices[i][2])}, ignore_index=True)
                print(f"Predicting {original_idx}, Prob. of top3: {top3.values[i]}")
                print(val_dataset.get_label(top3.indices[i][0]), val_dataset.get_label(top3.indices[i][1]), val_dataset.get_label(top3.indices[i][2]))

            

            
            
    # sort id in ascending order
    df = df.sort_values(by=['id'])
    df.to_csv(args.save_path, index=False)
    print(f'Save submission to {args.save_path}')
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Singer Classifier')
    parser.add_argument('--sample_rate', type=int, default=16000, metavar='N',
                        help='sample rate (default: 16000)')
    parser.add_argument('--num_workers', type=int, default=16, metavar='N',
                        help='number of workers (default: 16)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--trim_silence', type=bool, default=False, metavar='N',
                        help='trim silence (default: False)')
    parser.add_argument('--load_model', type=str, default=None, metavar='N',
                        help='load model (default: None), e.g. ./model/models/model_best.pth')
    parser.add_argument('--save_path', type=str, default='./submission.csv', metavar='N',
                        help='save path (default: ./submission.csv)')
    parser.add_argument('--segment_length', type=int, default=5, metavar='N',
                        help='segment length in seconds (default: 5)')

    
    args = parser.parse_args()
    main(args)
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
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model)['model_state_dict'])
        print(f'Load model from {args.load_model}')
    
    model.eval()
    
    # Inferece to get top 3 predictions for each test sample, and save to submission.csv
    # e.g. idx, top1, top2, top3
    top1acc = 0
    top3acc = 0
    pred, ground_truth = [], []
    with torch.no_grad():
        logger = tqdm(val_dataloader)
        for idx, (data, target) in enumerate(logger, 1):
            
            # segment data(audio) into non-overlapping 5s segments
            # e.g. data.shape = (1, 80000), segment_length = 5s, segment_length * sample_rate = 80000
            for i in range(max(1, data.shape[1] // (args.segment_length * args.sample_rate))):
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
                print(target[i].item(), top3.indices[i])
                # calculate top1 accuracy
                if target[i].item() == top3.indices[i][0]:
                    top1acc += 1
                    
                # calculate top3 accuracy
                if target[i].item() in top3.indices[i]:
                    top3acc += 1
                    
                print(f"Predicting {val_dataset.get_label(target[i].item())}, Prob. of top3: {top3.values[i]}")
                
                print(val_dataset.get_label(top3.indices[i][0]), val_dataset.get_label(top3.indices[i][1]), val_dataset.get_label(top3.indices[i][2]))
                
                logger.set_description(f"Top1 accuracy: {top1acc / idx}, Top3 accuracy: {top3acc / idx}")
                
                pred.append(top3.indices[i][0].item())
                ground_truth.append(target[i].item())
                
    print(f"Top1 accuracy: {top1acc / len(val_dataset)}")
    print(f"Top3 accuracy: {top3acc / len(val_dataset)}")
                
    # plot confusion matrix
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    # Assuming ground_truth and pred are defined elsewhere
    ground_truth = np.array(ground_truth)
    pred = np.array(pred)

    cm = confusion_matrix(ground_truth, pred)

    # Assuming val_dataset.label_encoder.keys() is defined
    df_cm = pd.DataFrame(cm, index=[i for i in val_dataset.label_encoder.keys()],
                        columns=[i for i in val_dataset.label_encoder.keys()])

    # show in percentage
    df_cm_percentage = df_cm.astype('float') / df_cm.sum(axis=1).values[:, np.newaxis]
    df_cm_percentage = df_cm_percentage.round(2)

    plt.figure(figsize=(20, 12))
    sn.heatmap(df_cm_percentage, annot=True, cmap='Blues', annot_kws={"size": 16}, fmt='.0%')  # Changed annot_kws and fmt
    # plt.xlabel('Predicted', fontsize=20)
    # plt.ylabel('Ground Truth', fontsize=20)
    plt.title('Confusion Matrix', fontsize=24)
    plt.xticks(fontsize=15)  # Added to change x-axis labels font size and orientation
    plt.yticks(fontsize=15)  # Added to change y-axis labels font size
    plt.tight_layout()
    
    plt.savefig('confusion_matrix.png')

    
    

    

            
def custom_collate(batch):
    data, target = zip(*batch)
    max_length = max([len(item) for item in data])
    
    padded_data = [F.pad(item, (0, max_length - len(item))) for item in data]
    data = torch.stack(padded_data)
    
    return data, torch.tensor(target)

    
    



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
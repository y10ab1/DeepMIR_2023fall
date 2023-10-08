import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import torch.nn.functional as F

from tqdm import tqdm
from dataset import AudioDataset
from model import SingerClassifier

def main(args):
    model = SingerClassifier(num_classes=20)
    dataset = AudioDataset(split='train', load_vocals=True, sample_rate=16000, trim_silence=args.trim_silence, do_augment=args.do_augment)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_dataset = AudioDataset(split='valid', load_vocals=True, sample_rate=16000, trim_silence=True, do_augment=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume)['model_state_dict'])
        optimizer.load_state_dict(torch.load(args.resume)['optimizer_state_dict'])
        
        print(f'Resume from {args.resume}')
    
    model.train()
    
    for epoch in range(1, args.epochs + 1):
        logger = tqdm(dataloader)
        for batch_idx, (data, target) in enumerate(logger):
            optimizer.zero_grad()
            output = model(data.to(model.device))
            
            loss = criterion(output.squeeze(1), target.to(model.device))
            loss.backward()
            optimizer.step()
            # print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
            logger.set_description(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
            
        scheduler.step(loss)            
        
        # In your main function:
        accuracy = validate(model, val_dataloader, model.device)
        print(f'Epoch: {epoch}, Accuracy: {accuracy}')
        
        # save model every 5 epochs
        if epoch % 5 == 0:
            os.makedirs('./model', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'./model/model_{epoch}.pth')

            
        # save best model
        if epoch == 1 or accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                # ... any other state information ...
            }, f'./model/model_best.pth')

            print(f'Save best model with accuracy: {best_acc}')

def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            output = model(data.to(device))
            predicted = torch.argmax(output.squeeze(1), dim=1)
            total += target.size(0)
            correct += (predicted == target.to(device)).sum().item()
    return correct / total

def custom_collate(batch):
    data, target = zip(*batch)
    max_length = max([len(item) for item in data])
    
    padded_data = [F.pad(item, (0, max_length - len(item))) for item in data]
    data = torch.stack(padded_data)
    
    return data, torch.tensor(target)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Singer Classifier')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--sample_rate', type=int, default=16000, metavar='N',
                        help='sample rate (default: 16000)')
    parser.add_argument('--num_workers', type=int, default=16, metavar='N',
                        help='number of workers (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='N',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--trim_silence', type=bool, default=False, metavar='N',
                        help='trim silence (default: False)')
    parser.add_argument('--resume', type=str, default=None, metavar='N',
                        help='resume from checkpoint (default: None), e.g. ./model/model_5.pth')
    parser.add_argument('--do_augment', type=bool, default=True, metavar='N',
                        help='do augment (default: False)')
    
    args = parser.parse_args()
    main(args)
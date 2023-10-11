import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import json

from tqdm import tqdm
from dataset import AudioDataset
from model import SingerClassifier

def main(args):
    writer = SummaryWriter(log_dir=f'{args.save_dir}/runs')
    # Convert args to a dictionary and then to a JSON string
    args_dict = vars(args)
    args_json_str = json.dumps(args_dict, indent=2)
    
    # Log args to TensorBoard
    writer.add_text('Arguments', args_json_str, 0)
    
    model_save_dir = f'{args.save_dir}/models'
    os.makedirs(model_save_dir, exist_ok=True)  # Ensure the directory exists

    
    model = SingerClassifier(num_classes=20, linear_config=args.linear_config)
    dataset = AudioDataset(split='train', load_vocals=True, sample_rate=16000, trim_silence=args.trim_silence, do_augment=args.do_augment, segment_length=args.segment_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_dataset = AudioDataset(split='valid', load_vocals=True, sample_rate=16000, trim_silence=True, do_augment=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 1
    
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume)['model_state_dict'])
        optimizer.load_state_dict(torch.load(args.resume)['optimizer_state_dict'])
        start_epoch = torch.load(args.resume)['epoch']
        prev_acc = torch.load(args.resume)['val_acc']
        print(f'Resume from {args.resume}, start_epoch: {start_epoch}, prev_acc: {prev_acc}')
    
    model.train()
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        
        logger = tqdm(dataloader)
        for batch_idx, (data, target) in enumerate(logger):
            optimizer.zero_grad()
            output = model(data.to(model.device))
            
            loss = criterion(output.squeeze(1), target.to(model.device))
            loss.backward()
            optimizer.step()
            # print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
            logger.set_description(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
            
            writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + batch_idx)

            
        
        # In your main function:
        accuracy, val_loss = validate(model, val_dataloader, model.device)
        print(f'Epoch: {epoch}, Accuracy: {accuracy}, val_loss: {val_loss}')
        
        scheduler.step(val_loss)
        
        writer.add_scalar('Validation Accuracy', accuracy, epoch)
        writer.add_scalar('Validation Loss', val_loss, epoch)
    

        # Save model every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': accuracy,
                'epoch': epoch,
            }, f'{model_save_dir}/model_{epoch}.pth')
        
        # Save best model
        if epoch == 1 or accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': accuracy,
                'epoch': epoch,
            }, f'{model_save_dir}/model_best.pth')
            print(f'Save best model with accuracy: {best_acc}')
            
    writer.close()

def validate(model, target_dataloader, device):
    model.eval()
    correct = 0
    total = 0
    loss = float('inf')
    with torch.no_grad():
        for data, target in tqdm(target_dataloader):
            output = model(data.to(device))
            predicted = torch.argmax(output.squeeze(1), dim=1) 
            total += target.size(0)
            correct += (predicted == target.to(device)).sum().item()
            loss = min(loss, F.cross_entropy(output.squeeze(1), target.to(device)).item())
    return correct / total, loss

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
    parser.add_argument('--save_dir', type=str, default='./model', metavar='N',
                        help='save directory (default: ./model)')
    parser.add_argument('--linear_config', type=int, default=[512], metavar='N', nargs='+',
                        help='linear config (default: [512])')
    parser.add_argument('--segment_length', type=int, default=5, metavar='N',
                        help='segment length for training (default: 5s), if 0, use full length')
    args = parser.parse_args()
    main(args)
import torch
import argparse
import os
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from dataset import AudioDataset
from model import SingerClassifier
import seaborn as sn
from matplotlib.lines import Line2D


def main(args):
    model = SingerClassifier(num_classes=20)

    val_dataset = AudioDataset(split='valid', load_vocals=True, sample_rate=16000, trim_silence=True, do_augment=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model)['model_state_dict'])
        print(f'Load model from {args.load_model}')
    
    model.eval()
    print(model)
    
    # register hook to get layer output
    class OutputHook:
        def __init__(self):
            self.output = None

        def hook_fn(self, module, input, output):
            self.output = output

    # Create instances of the hook container
    output_hook_embedding = OutputHook()
    output_hook_linear = OutputHook()

    # Register the hooks on the layers you're interested in
    hook_embedding = model.pretrained_model.mods.embedding_model.fc.register_forward_hook(output_hook_embedding.hook_fn)
    hook_linear = model.linear[3].register_forward_hook(output_hook_linear.hook_fn)  # Assuming the last layer is at index 4


    pretrain_embeddings = []
    linear_embeddings = []
    labels = []
    
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            # Assuming your model has a method to extract embeddings
            # Modify this line to match your model's architecture
            
            # Run a forward pass
            with torch.no_grad():
                model(data.to(model.device))
            # Access the layer outputs
            layer_output_embedding = output_hook_embedding.output
            layer_output_linear = output_hook_linear.output

            
            
            pretrain_embeddings.append(layer_output_embedding.cpu())
            linear_embeddings.append(layer_output_linear.cpu())     
            labels.append(target.cpu())
            
            
    # Remove the hooks when you're done
    hook_embedding.remove()
    hook_linear.remove()
    

    # Concatenate the embeddings from all the batches
    embeddings = torch.cat(pretrain_embeddings, dim=0)
    embeddings = embeddings.squeeze(1)
    embeddings = embeddings.numpy()
    
    linear_embeddings = torch.cat(linear_embeddings, dim=0)
    linear_embeddings = linear_embeddings.squeeze(1)
    linear_embeddings = linear_embeddings.numpy()
    
    labels = torch.cat(labels, dim=0)
    labels = labels.numpy()
    
    # Squeeze the embeddings to 2D
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    linear_embeddings = linear_embeddings.reshape(linear_embeddings.shape[0], -1)
    
    
    
    # Apply t-SNE for pretrain embeddings
    tsne = TSNE(n_components=2, random_state=0)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Apply t-SNE for linear embeddings
    tsne = TSNE(n_components=2, random_state=0)
    tsne_linear_embeddings = tsne.fit_transform(linear_embeddings)
    
    # For pretrain embeddings
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels, cmap='tab20')
    # plt.colorbar(scatter, label='Class Label')
    plt.title('t-SNE of Pretrain Embeddings')
    # Create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'{val_dataset.get_label(i)}',
                          markersize=10, markerfacecolor=plt.cm.tab20(i/20)) for i in range(20)]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('tsne_pretrain.png')
    plt.clf()

    # For linear embeddings
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(tsne_linear_embeddings[:, 0], tsne_linear_embeddings[:, 1], c=labels, cmap='tab20')
    # plt.colorbar(scatter, label='Class Label')
    plt.title('t-SNE of Linear Embeddings')
    # Create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'{val_dataset.get_label(i)}',
                          markersize=10, markerfacecolor=plt.cm.tab20(i/20)) for i in range(20)]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('tsne_linear.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Singer Classifier')
    parser.add_argument('--sample_rate', type=int, default=16000, metavar='N', help='sample rate (default: 16000)')
    parser.add_argument('--num_workers', type=int, default=16, metavar='N', help='number of workers (default: 16)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--trim_silence', type=bool, default=False, metavar='N', help='trim silence (default: False)')
    parser.add_argument('--load_model', type=str, default='model_new/models/model_best.pth', metavar='N', help='load model (default: None), e.g. ./model_new/models/model_best.pth')
    parser.add_argument('--save_path', type=str, default='./submission.csv', metavar='N', help='save path (default: ./submission.csv)')
    parser.add_argument('--segment_length', type=int, default=5, metavar='N', help='segment length in seconds (default: 5)')

    args = parser.parse_args()
    main(args)

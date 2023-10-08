import torch
import torch.nn as nn
import torch.optim as optim
from speechbrain.pretrained import EncoderClassifier

from dataset import AudioDataset

class SingerClassifier(nn.Module):
    def __init__(self, num_classes=20, num_features=512):
        super(SingerClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":f"{self.device}"})
        self.pretrained_model.fc = nn.Identity()
        self.linear = nn.Sequential(
            nn.Flatten(1),  # Add this line to flatten the data
            nn.Linear(192, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_classes)
        ).to(self.device)

            

    def forward(self, x):
        with torch.no_grad():
            x = self.pretrained_model.encode_batch(x.to(self.device)) # get embedding, shape: (batch_size, embedding_size)
        x = self.linear(x)
        return x

# unit test
if __name__ == '__main__':
    from tqdm import tqdm
    model = SingerClassifier(num_classes=20)
    dataset = AudioDataset(split='train', load_vocals=True, sample_rate=16000, trim_silence=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
    
    val_dataset = AudioDataset(split='valid', load_vocals=True, sample_rate=16000, trim_silence=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1,11):
        for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            output = model(data.to(model.device))
            
            loss = criterion(output.squeeze(1), target.to(model.device))
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in tqdm(val_dataloader):
                output = model(data.to(model.device))
                predicted = torch.argmax(output.squeeze(1), dim=1)
                total += target.size(0)
                correct += (predicted == target.to(model.device)).sum().item()
        print(f'Epoch: {epoch}, Accuracy: {correct / total}')
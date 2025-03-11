from datasets.RodoSolDataset import RodoSolDataset
from torch.utils.data import DataLoader
from models.YOLOv2 import YOLOv2
import torch
import torch.optim as optim

train_dataset = RodoSolDataset(root='data/', split='train')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = YOLOv2()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):  # 50 epochs
    model.train()
    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}: Loss = {loss.item()}')

torch.save(model.state_dict(), 'models/yolov2_trained.pth')
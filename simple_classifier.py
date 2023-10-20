from PIL import Image
import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
datas = DataLoader(train, 32)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(1,32,(3,3)), nn.ReLU(),nn.Conv2d(32,64,(3,3)), nn.Flatten(), nn.Linear(30976, 10))
    def forward(self,x): return self.model(x) 
    
clf1 = Classifier().to('cuda')
opt = Adam(clf1.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

if __name__ == "__main__":
    for e in range(10):
        for batch in datas:
            x,y=batch
            x,y=x.to('cuda'), y.to('cuda')
            yhat = clf1(x)
            loss = lossf(yhat,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("epoch : "+str(e)+" | loss : "+str(loss.item()))
    with open("model_state.pt", "wb") as f:
        save(clf1.state_dict(), f)
        
    with open('model_state.pt', 'rb') as f: 
        clf1.load_state_dict(load(f))  

    img = Image.open('img_3.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    print(torch.argmax(clf1(img_tensor)))
            

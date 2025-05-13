import torch.nn as nn

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        return self.net(x)

    
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 6 output channels
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),            # 16 output channels
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 64),                  
            nn.ReLU(),
            nn.Linear(64, 32),                
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)



# # trained by:
# def train_and_save_model(model_fn, X_train, Y_train, save_path=None, epochs=500, device="cpu"):

#     model = model_fn().to(device)
#     model.train()
#     optimizer = Adam(model.parameters(), lr=1e-3)

#     for epoch in range(epochs):
#         logits = model(X_train)
#         loss = F.cross_entropy(logits, Y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if epoch%500 == 0:
#             print(f"Epoch {epoch+1}: Loss = {loss.item()}")
#     print(f"final loss: {loss.item()}")
#     if save_path:
#         torch.save(model.state_dict(), save_path)
#         print(f"Model saved to {save_path}")
#     return model

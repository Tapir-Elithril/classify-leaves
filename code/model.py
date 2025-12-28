from base_model import *

import torchvision.models as models
from torchvision.models import ResNet50_Weights

class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedResNet50, self).__init__()

        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # last layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)  
        # print(f"x.shape={x.shape}")

        return self.backbone(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, valid_loader, num_classes = create_dataloaders(batch_size=32)
    
    model = PretrainedResNet50(num_classes=num_classes).to(device)
    max_epoch = 50

    optimizer, scheduler = get_optimizer_with_scheduler(model)
    
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), 
    #     lr=1e-3, 
    #     weight_decay=1e-4
    # )
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=0.01,
    #     momentum=0.9,
    #     weight_decay=1e-4,
    #     nesterov=True
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=20, 
    #     eta_min=1e-6
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=5,      
    #     T_mult=2,     
    #     eta_min=1e-4
    # )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.003,         
    #     epochs=15,             
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.3,        
    #     anneal_strategy='cos', 
    #     cycle_momentum=True
    # )

    checkpoint_path = './../checkpoints/temp1.pth'
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"continue from epoch {start_epoch}")
    except:
        start_epoch = 0
        print("start training from epoch 0")

    history = train(model, train_loader, valid_loader, optimizer, scheduler, max_epoch, device=device)
    
    # torch.save(model.state_dict(), './../checkpoints/pretrained_leaf_net.pth')
    
    plot_path = './../checkpoints/pretrainedresnet50_training_history.png'
    plot_history(history=history, save_path=plot_path)

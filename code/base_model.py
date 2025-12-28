# conda activate cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Device: {device}")

class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224x224x3 -> 224x224x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 224x224x32 -> 224x224x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224x224x32 -> 112x112x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 112x112x32 -> 112x112x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 112x112x64 -> 112x112x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112x64 -> 56x56x64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 56x56x64 -> 56x56x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 56x56x128 -> 56x56x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56x128 -> 28x28x128
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 28x28x128 -> 28x28x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 28x28x256 -> 28x28x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28x256 -> 14x14x256
        )
        
        # fully-connected layer classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  
            nn.Linear(256 * 14 * 14, 512),  # 256 * 14 * 14 = 50176
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)  # output
        )

    def forward(self, x):
        # (batch, height, width, channels) -> (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# def get_optimizer(net, lr):
#     optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
#     return optimizer

def get_optimizer_with_scheduler(net, initial_lr=1e-2, total_epochs=50):
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=1e-4)
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=2
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,      
        T_mult=2,     
        eta_min=1e-4
    )

    print('Using optimizer:SGDmomentum, with scheduler:CosineAnnealingWarmRestarts')

    return optimizer, scheduler

def load_model(net, path):
    checkpoint = torch.load(path)
    if 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint)
    print(f"load model from {path}")

def save_model(net, path):
    torch.save(net.state_dict(), path)

def forward_step(net, inputs, labels, device):
    inputs, labels = inputs.to(device), labels.to(device)
    criterion = nn.CrossEntropyLoss()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    return outputs, loss, labels

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def train(net, train_loader, valid_loader, optimizer, scheduler, max_epoch, device):

    torch.cuda.empty_cache() 
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_model_state = None
    temp_model_state = None

    net.train()
    N = len(train_loader)
    print_interval = (N // 8 // 100 + 1) * 100

    for epoch in range(max_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print(f"inputs.shape = {inputs.shape}")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, loss, labels = forward_step(net, inputs, labels, device)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)
            if (i + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                total_batches = len(train_loader)
                print(f'Epoch [{epoch+1}/{max_epoch}], Batch [{i+1}/{total_batches}], '
                      f'Loss: {running_loss/100:.4f}, Acc: {100*running_acc/100:.2f}%, '
                      f'LR: {current_lr:.6f}')
                running_loss = 0.0
                running_acc = 0.0

        net.eval()
        train_loss, train_acc = 0.0, 0.0
        with torch.no_grad():
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                train_loss += loss.item()
                train_acc += accuracy(outputs, labels)
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_acc / len(train_loader)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()
                val_acc += accuracy(outputs, labels)
        
        val_loss /= len(valid_loader)
        val_acc = 100 * val_acc / len(valid_loader)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = net.state_dict().copy()
            # patience_counter = 0  
            print(f'✅ new best, acc: {val_acc:.2f}%')
            best_checkpoint_path = './../checkpoints/best_model1.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'history': history,
                'best_val_acc': best_val_acc
            }, best_checkpoint_path)
            
            print(f'model saved to: {best_checkpoint_path}')
        else:
            temp_model_state = net.state_dict().copy()
            temp_checkpoint_path = './../checkpoints/temp1.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'history': history,
                'best_val_acc': best_val_acc
            }, temp_checkpoint_path)
            
            print(f'model saved to: {temp_checkpoint_path}')
        
        print(f'Epoch [{epoch+1}/{max_epoch}] summary:')
        print(f'  training loss: {train_loss:.4f}, training acc: {train_acc:.2f}%')
        print(f'  validation loss: {val_loss:.4f}, validation_acc: {val_acc:.2f}%')
        print(f'  learning_rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        print('-' * 60)
    
    if best_model_state:
        net.load_state_dict(best_model_state)
        print(f'🎯 best model validation acc: {best_val_acc:.2f}%')
    
    print('Training Complete!')
    return history

class LeafDataset(TensorDataset):
    def __init__(self, images, labels, is_train=True):
        self.images = images
        self.labels = labels
        self.is_train = is_train
        
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(180),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                transforms.ElasticTransform(alpha=50.0, sigma=5.0),
                transforms.RandomAffine(
                    degrees=15, 
                    translate=(0.15, 0.15), 
                    scale=(0.85, 1.15)
                ),
                
                transforms.ColorJitter(
                    brightness=0.3,  
                    contrast=0.3,      
                    saturation=0.3,  
                    hue=0.15         
                ),

                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
            print("training data augmented.")

        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # if len(image.shape) == 3 and image.shape[0] == 3:
        #     image = np.transpose(image, (2, 0, 1))  # [C, H, W] -> [H, W, C]
        
        if image.dtype in [np.float32, np.float64]:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def create_augmented_datasets(train_data, train_labels, valid_data, valid_labels, batch_size=32):
    
    train_dataset = LeafDataset(train_data, train_labels, is_train=True)
    valid_dataset = LeafDataset(valid_data, valid_labels, is_train=False)
    
    print("creating dataloader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, 
        pin_memory=True,  
        drop_last=False,  
        prefetch_factor=None  
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=None
    )
    
    print(f"training set: {len(train_dataset)} samples")
    print(f"validation set: {len(valid_dataset)} samples")
    
    return train_loader, valid_loader

def create_dataloaders(batch_size=32):

    train_data = np.load('./../classify-leaves/normalized/train_images.npy',allow_pickle=True)
    valid_data = np.load('./../classify-leaves/normalized/valid_images.npy',allow_pickle=True)
    
    train_labels = np.load('./../classify-leaves/normalized/train_labels.npy',allow_pickle=True)
    valid_labels = np.load('./../classify-leaves/normalized/valid_labels.npy',allow_pickle=True)

    print(f"training set: {len(train_data)} samples")
    print(f"validation set: {len(valid_data)} samples")

    encoder = LabelEncoder()
    all_labels = np.concatenate([train_labels, valid_labels])
    encoder.fit(all_labels)
    train_labels_encoded = encoder.transform(train_labels)
    valid_labels_encoded = encoder.transform(valid_labels)

    print(train_data.shape, train_data.dtype)
    print(f"data range: [{train_data.min():.6f}, {train_data.max():.6f}]")

    train_loader, valid_loader = create_augmented_datasets(
        train_data, train_labels_encoded, 
        valid_data, valid_labels_encoded, batch_size
    )
    # train_tensor = torch.FloatTensor(train_data)
    # valid_tensor = torch.FloatTensor(valid_data)
    # train_labels_tensor = torch.LongTensor(train_labels_encoded)
    # valid_labels_tensor = torch.LongTensor(valid_labels_encoded)
    
    # train_dataset = TensorDataset(train_tensor, train_labels_tensor)
    # valid_dataset = TensorDataset(valid_tensor, valid_labels_tensor)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(np.unique(train_labels))

    print("DataLoader Creation...Done!")
    
    return train_loader, valid_loader, num_classes

def plot_history(history, save_path=None):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    best_val_loss_epoch = np.argmin(history['val_loss'])
    best_val_loss = history['val_loss'][best_val_loss_epoch]
    axes[0].axvline(x=best_val_loss_epoch, color='r', linestyle='--', alpha=0.5)
    axes[0].text(best_val_loss_epoch, best_val_loss, f' Best: {best_val_loss:.3f}', 
                verticalalignment='bottom')
    
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    best_val_acc_epoch = np.argmax(history['val_acc'])
    best_val_acc = history['val_acc'][best_val_acc_epoch]
    axes[1].axvline(x=best_val_acc_epoch, color='r', linestyle='--', alpha=0.5)
    axes[1].text(best_val_acc_epoch, best_val_acc, f' Best: {best_val_acc:.1f}%', 
                verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"training history figure saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    train_loader, valid_loader, num_classes = create_dataloaders(batch_size=32)
    
    if train_loader is None:
        exit(1)

    leaf_net = NeuralNet(num_classes=num_classes).to(device)
    lr = 1e-2
    optimizer, scheduler = get_optimizer_with_scheduler(leaf_net, lr)

    history = train(
        net=leaf_net, 
        train_loader=train_loader,  
        valid_loader=valid_loader,  
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_epoch=8, 
        device=device
    )

    PATH = './../checkpoints/leaf_net.pth'

    save_model(leaf_net, PATH)
    print(f"model saved to: {PATH}")

    plot_path = './../checkpoints/training_history.png'
    plot_history(history=history, save_path=plot_path)


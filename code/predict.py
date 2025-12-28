from model import * 
import os
import pandas as pd

MODEL_PATH = './../checkpoints/best_model1.pth'
TEST_DATA_PATH = './../classify-leaves/normalized/test_images.npy'
OUTPUT_PATH = './../classify-leaves/final_submission.csv'

def create_label_mapping():

    train_labels = np.load('./../classify-leaves/normalized/train_labels.npy', allow_pickle=True)
    valid_labels = np.load('./../classify-leaves/normalized/valid_labels.npy', allow_pickle=True)
    
    all_labels = np.concatenate([train_labels, valid_labels])
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)

    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    np.save('./../classify-leaves/normalized/idx_to_label.npy', idx_to_label)
    np.save('./../classify-leaves/normalized/label_to_idx.npy', label_to_idx)
    
    print(f"create label mapping: {num_classes} classes")
    return idx_to_label, num_classes

def load_label_mapping():

    idx_to_label_path = './../classify-leaves/normalized/idx_to_label.npy'
    
    if os.path.exists(idx_to_label_path):
        idx_to_label = np.load(idx_to_label_path, allow_pickle=True).item()
        num_classes = len(idx_to_label)
        print(f"load: {num_classes} classes")
        return idx_to_label, num_classes
    else:
        print("creating label mapping...")
        return create_label_mapping()

class TestDataset(TensorDataset):
    def __init__(self, images):
        self.images = images
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
        
        if image.dtype in [np.float32, np.float64]:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        
        return image

def predict(net, device, batch_size=32):
    test_data = np.load(TEST_DATA_PATH, allow_pickle=True)
    print(test_data.shape, test_data.dtype)
    print(f"data range: [{test_data.min():.6f}, {test_data.max():.6f}]")
    # test_tensor = torch.FloatTensor(test_data)
    # test_dataset = TensorDataset(test_tensor)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TestDataset(test_data)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=None
    )

    idx_to_label, num_classes = load_label_mapping()

    net.eval()
    all_predictions = []
    
    with torch.no_grad():
        for inputs in test_loader:  
            inputs = inputs.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for pred_idx in predicted.cpu().numpy():
                label_name = idx_to_label[pred_idx]
                all_predictions.append(label_name)
    
    print(f"prediction completes: {len(all_predictions)} samples")
    return all_predictions

def output(predictions):

    test_csv_path = './../classify-leaves/test.csv'
    df = pd.read_csv(test_csv_path)
    image_names = df['image'].tolist()
    
    submission_df = pd.DataFrame({
        'image': image_names,
        'label': predictions
    })
    
    submission_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"submission.csv in : {OUTPUT_PATH}")
    print(f"sample num: {len(submission_df)}")
    
    return submission_df

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    net = PretrainedResNet50(num_classes=176).to(device)
    load_model(net, MODEL_PATH)
    
    predictions = predict(net, device)
    
    output(predictions)
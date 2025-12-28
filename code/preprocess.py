from glance import *

class ImageNormalizer:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    
    def normalize_single(self, image_path):
        with Image.open(image_path) as img:
            
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            normalized = (img_array - self.mean) / self.std
            
        return normalized
    
    def normalize_batch(self, image_paths):
        normalized_images = []
        original_sizes = []
        
        for img_path in tqdm(image_paths):
            norm_img = self.normalize_single(img_path)
            normalized_images.append(norm_img)
        
        return np.array(normalized_images)
    
    def display_normalization_info(self):
        print("=== image normalization ===")
        print(f"mean (R, G, B): {self.mean}")
        print(f"variance (R, G, B): {self.std}")

def batch_normalization(train_images, valid_images, test_images):
    all_images = train_images + valid_images + test_images
    normalized_batch = normalizer.normalize_batch(all_images)

    train_count = len(train_images)
    valid_count = len(valid_images)
    test_count = len(test_images)

    train_normalized = normalized_batch[:train_count]
    valid_normalized = normalized_batch[train_count:train_count+valid_count]
    test_normalized = normalized_batch[train_count+valid_count:]
    print(f"Image normalized complete.")  

    save_dir = "./../classify-leaves/normalized"
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "train_images.npy"), train_normalized)
    np.save(os.path.join(save_dir, "valid_images.npy"), valid_normalized)
    np.save(os.path.join(save_dir, "test_images.npy"), test_normalized)
    print(f"Normalized data saved to: {save_dir}")
    print(f"  - train_images.npy: {train_normalized.shape}")
    print(f"  - valid_images.npy: {valid_normalized.shape}")
    print(f"  - test_images.npy: {test_normalized.shape}")

def extract_labels_from_splits(train_split, valid_split):
    print("\n=== label extracting ===")
    
    label_column = train_split.columns[-1] 
    
    train_labels = train_split[label_column].values
    valid_labels = valid_split[label_column].values
    
    all_labels = np.concatenate([train_labels, valid_labels])
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)

    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    np.save('./../classify-leaves/normalized/idx_to_label.npy', idx_to_label)
    
    print(f"num_classes: {num_classes}")

    train_labels_idx = np.array([label_to_idx[label] for label in train_labels])
    valid_labels_idx = np.array([label_to_idx[label] for label in valid_labels])

    np.save('./../classify-leaves/normalized/train_labels.npy', train_labels)
    np.save('./../classify-leaves/normalized/valid_labels.npy', valid_labels)
    print(f"label saved to ./../classify-leaves/normalized/")
    
    return train_labels_idx, valid_labels_idx, num_classes, label_to_idx, idx_to_label


if __name__ == "__main__":

    train_data, test_data = read_and_observe()
    _, test_images = size_check(train_data=train_data, test_data=test_data)
    #mean, std = dataset_statistics(train_data=train_data, test_data=test_data)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    train_split, valid_split, train_images, valid_images= train_valid_split(train_data)

    normalizer = ImageNormalizer(mean, std)
    normalizer.display_normalization_info()

    batch_normalization(train_images, valid_images, test_images)
    train_labels_idx, valid_labels_idx, num_classes, label_to_idx, idx_to_label = extract_labels_from_splits(train_split, valid_split)
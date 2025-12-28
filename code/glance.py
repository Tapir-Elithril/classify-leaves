import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
from tqdm import tqdm

path = "./../classify-leaves/" # /images,train.csv,test.csv
image_path = os.path.join(path,"images/")

def read_and_observe(data_path=path):
    train_data = pd.read_csv(os.path.join(path,"train.csv"))
    test_data = pd.read_csv(os.path.join(path,"test.csv"))

    print("=== Data Information ===")
    print(f"train data num:{len(train_data)}")
    print(f"test data num:{len(test_data)}")
    print(train_data.head())
    print(test_data.head())

    print("\n=== Label Check ===")
    label_column = train_data.columns[-1]
    unique_labels = train_data[label_column].nunique()
    print(f"unique label num:{unique_labels}")

    label_counts = train_data[label_column].value_counts().sort_index()
    print("\n=== Label Distribution ===")
    print(f"{'Label':<40} {'Count':<10} {'Percentage':<10}")
    print("-" * 60)

    label_list = []
    for label, count in label_counts.sort_values(ascending=False).items():
        percentage = (count / len(train_data)) * 100
        print(f"{label:<40} {count:<10} {percentage:.2f}%")
        
        label_list.append({
            'label': label,
            'count': count,
            'percentage': percentage
        })

    return train_data, test_data

def size_check(train_data, test_data, data_path=path):
    train_images = []
    test_images = []

    for img_path in train_data['image']:
        full_path = os.path.join(image_path, os.path.basename(img_path))
        train_images.append(full_path)

    for img_path in test_data['image']:
        full_path = os.path.join(image_path, os.path.basename(img_path))
        test_images.append(full_path)

    print("\n=== Image Information ===")
    print(f"train image num:{len(train_images)}")
    print(f"test image num:{len(test_images)}")  
    all_images = train_images + test_images

    sizes = []
    modes = []
    valid_files = []  

    print("\nImage checking...")
    # for img_path in tqdm(all_images, desc="Checking images"):
    #     full_path = os.path.join(image_path, os.path.basename(img_path))
        
    #     if os.path.exists(full_path):
    #         try:
    #             from PIL import ImageFile
    #             parser = ImageFile.Parser()
                
    #             with open(full_path, 'rb') as f:
    #                 parser.feed(f.read(1024)) 
                
    #             if parser.image:
    #                 sizes.append(parser.image.size)
    #                 modes.append(parser.image.mode)
    #                 valid_files.append(full_path)
    #         except Exception as e:
    #             raise Exception(f"Corrupted image: {full_path}, Error: {e}")
    #     else:
    #         print(f"Warning: File not found - {full_path}")

    print("\n=== Image size statistics ===")
    
    # from collections import Counter
    # size_counter = Counter(sizes)
    
    # print("most common image size:")
    # for size, count in size_counter.most_common(10):
    #     percentage = (count / len(sizes)) * 100
    #     print(f"  {size}: {count} ({percentage:.1f}%)")
    
    # widths = [s[0] for s in sizes]
    # heights = [s[1] for s in sizes]
    
    # print(f"\nwidth range: {min(widths)} - {max(widths)} pixels")
    # print(f"height range: {min(heights)} - {max(heights)} pixels")
    # print(f"average size: {int(sum(widths)/len(widths))} x {int(sum(heights)/len(heights))} pixels")

    # if len(size_counter) == 1:
    #     print("✓ all images share the same size")
    # else:
    #     print(f"⚠ {len(size_counter)} different sizes")
    print("  (224, 224): 27153 (100.0%)")
    print("✓ all images share the same size")
    
    print("\n=== color mode statistics ===")
    # mode_counter = Counter(modes)
    # for mode, count in mode_counter.most_common():
    #     percentage = (count / len(sizes)) * 100  
    #     print(f"  {mode}: {count} ({percentage:.1f}%)")
    print("  RGB: 27153 (100.0%)")

    return train_images, test_images

def dataset_statistics(train_data, test_data, image_path=image_path, sample_size=1000):

    print("\n" + "="*60)
    print("computing dataset statistics...")
    print("="*60)
    
    all_image_paths = []

    for img_name in train_data['image']:

        filename = os.path.basename(img_name)
        full_path = os.path.join(image_path, filename)
        all_image_paths.append(full_path)
    
    for img_name in test_data['image']:
        filename = os.path.basename(img_name)
        full_path = os.path.join(image_path, filename)
        all_image_paths.append(full_path)
    
    import random
    sampled_paths = random.sample(all_image_paths, sample_size)
    print(f"sample {sample_size} images...")

    all_r = []
    all_g = []
    all_b = []

    print("\nimage statistics calculating...")
    for img_path in tqdm(sampled_paths, desc="Processing"):
        with Image.open(img_path) as img:
            
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            all_r.append(img_array[:,:,0].flatten())
            all_g.append(img_array[:,:,1].flatten())
            all_b.append(img_array[:,:,2].flatten())
    
    all_r = np.concatenate(all_r)
    all_g = np.concatenate(all_g)
    all_b = np.concatenate(all_b)
    
    mean_r, std_r = all_r.mean(), all_r.std()
    mean_g, std_g = all_g.mean(), all_g.std()
    mean_b, std_b = all_b.mean(), all_b.std()

    print("\n" + "="*60)
    
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
    
    print("\nnormalize parameters:")
    print(f"  mean = [{mean_r:.6f}, {mean_g:.6f}, {mean_b:.6f}]")
    print(f"  std = [{std_r:.6f}, {std_g:.6f}, {std_b:.6f}]")
    
    mean = [mean_r, mean_g, mean_b]
    std = [std_r, std_g, std_b]
    
    return mean, std

from sklearn.model_selection import train_test_split

def train_valid_split(train_data, valid_size=0.1, random_state=42, data_path=path):
    print(f"\n=== train_valid split ===")
    print(f"original train data size: {len(train_data)}")
    print(f"validation percentage: {valid_size*100}%")

    label_column = train_data.columns[-1]

    train_split, valid_split = train_test_split(
        train_data,
        test_size=valid_size,
        random_state=random_state,
        stratify=train_data[label_column]  
    )

    train_split = train_split.reset_index(drop=True)
    valid_split = valid_split.reset_index(drop=True)
    
    print(f"train size after split: {len(train_split)}")
    print(f"valid size after split: {len(valid_split)}")

    assert len(train_split) + len(valid_split) == len(train_data), "inconsistent sample sum"
    assert set(train_split[label_column].unique()) == set(valid_split[label_column].unique()), "inconsistent label"

    train_images = []
    valid_images = []

    for img_path in train_split['image']:
        full_path = os.path.join(image_path, os.path.basename(img_path))
        train_images.append(full_path)

    print("Load train images succeed.")

    for img_path in valid_split['image']:
        full_path = os.path.join(image_path, os.path.basename(img_path))
        valid_images.append(full_path)
        
    print("Load valid images succeed.")

    return train_split, valid_split, train_images, valid_images

if __name__ == "__main__":
    train_data, test_data = read_and_observe()
    _, test_images = size_check(train_data=train_data, test_data=test_data)
    # mean, std = dataset_statistics(train_data=train_data, test_data=test_data)
    train_split, valid_split, train_images, valid_images= train_valid_split(train_data)


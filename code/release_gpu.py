import torch
import gc

def cleanup_gpu():
    print("releasing GPU...")
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"allocated: {allocated:.2f} GB, cache: {cached:.2f} GB")
        # output should be 0, 0 if cleaned succeed
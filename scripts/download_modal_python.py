import modal
import os
import sys
import time

def download_volume_robust(volume_name, file_name, out_dir):
    vol = modal.Volume.from_name(volume_name)
    out_path = os.path.join(out_dir, file_name)
    print(f"Downloading {file_name} from {volume_name} to {out_path}...")
    
    start_time = time.time()
    last_print = start_time
    bytes_written = 0
    
    with open(out_path, "wb") as f:
        for chunk in vol.read_file(file_name):
            f.write(chunk)
            bytes_written += len(chunk)
            
            now = time.time()
            if now - last_print > 5.0:
                mb = bytes_written / (1024 * 1024)
                print(f"  ... {mb:.2f} MB downloaded so far")
                last_print = now
                
    total_mb = bytes_written / (1024 * 1024)
    print(f"✓ Finished {file_name}: {total_mb:.2f} MB in {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    out_dir = os.path.expanduser("~/throng_backup")
    os.makedirs(out_dir, exist_ok=True)
    download_volume_robust("throng-runs", "signal_corpus.jsonl", out_dir)
    download_volume_robust("throng-runs", "signal_corpus_red.jsonl", out_dir)

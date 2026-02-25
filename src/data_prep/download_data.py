from kaggle.api.kaggle_api_extended import KaggleApi
import os
import zipfile

# Tải ONLY Home_and_Kitchen từ Kaggle
category = "Home_and_Kitchen"
dataset_id = "McAuley-Lab/amazon-reviews-2023"

# Authenticate với Kaggle
api = KaggleApi()
api.authenticate()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
download_dir = os.path.join(project_root, "data", "raw")
os.makedirs(download_dir, exist_ok=True)

# Tải file meta_Home_and_Kitchen.jsonl từ Kaggle
print(f"Đang tải file meta_{category}.jsonl từ Kaggle...")

try:
	# Tải dataset từ Kaggle
	api.dataset_download_files(
		dataset_id,
		path=download_dir,
		unzip=False
	)
	
	save_dir = os.path.join(download_dir, "meta_categories")
	os.makedirs(save_dir, exist_ok=True)
	
	# Tìm file meta_Home_and_Kitchen.jsonl
	meta_file = f"meta_{category}.jsonl"
	found = False
	
	for root, dirs, files in os.walk(download_dir):
		for file in files:
			if file == meta_file:
				source_path = os.path.join(root, file)
				dest_path = os.path.join(save_dir, file)
				os.rename(source_path, dest_path)
				print(f"✓ Hoàn tất! File đã tải vào: {dest_path}")
				found = True
				break
		if found:
			break
	
	if not found:
		print(f"⚠ Không tìm thấy file {meta_file}")
		
except Exception as e:
	print(f"❌ Lỗi tải file: {e}")
	print("Vui lòng kiểm tra:")
	print("1. Kaggle API credentials (~/.kaggle/kaggle.json)")
	print("2. Dataset ID đúng: McAuley-Lab/amazon-reviews-2023")

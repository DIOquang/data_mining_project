import os
import json
import pandas as pd
from pathlib import Path

def convert_jsonl_to_parquet(jsonl_file, output_file):
    """
    Convert a JSONL file to Parquet format.
    For large files, uses pandas read_json with lines=True for efficient processing.
    
    Args:
        jsonl_file: Path to the input JSONL file
        output_file: Path to the output Parquet file
    """
    jsonl_path = Path(jsonl_file)
    output_path = Path(output_file)
    
    if not jsonl_path.exists():
        print(f"❌ Error: File not found at {jsonl_path}")
        return False
    
    try:
        # Check file size
        file_size_gb = jsonl_path.stat().st_size / (1024**3)
        print(f"📖 Reading JSONL file: {jsonl_path.name}")
        print(f"   File size: {file_size_gb:.2f} GB")
        print("   This may take a few minutes for large files...")
        
        # Use pandas read_json with lines=True for efficient JSONL processing
        # This is much faster than manual line-by-line parsing
        df = pd.read_json(jsonl_path, lines=True)
        
        print(f"✅ Loaded {len(df)} records")
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to Parquet
        print(f"💾 Writing to Parquet: {output_path.name}")
        print("   This may also take a few minutes...")
        df.to_parquet(output_path, index=False, compression='snappy')
        
        output_size_mb = output_path.stat().st_size / (1024**2)
        output_size_gb = output_size_mb / 1024
        print(f"✅ Successfully converted to Parquet!")
        print(f"   Output file size: {output_size_gb:.2f} GB ({output_size_mb:.2f} MB)")
        print(f"   Output location: {output_path}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Convert meta_Home_and_Kitchen
    input_file = "raw/meta_categories/meta_Home_and_Kitchen.jsonl"
    output_file = "processed/meta_categories/meta_Home_and_Kitchen.parquet"
    
    print("=" * 60)
    print("🚀 Starting JSONL to Parquet conversion...")
    print("=" * 60)
    
    success = convert_jsonl_to_parquet(input_file, output_file)
    
    if success:
        print("=" * 60)
        print("✨ Conversion completed successfully!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ Conversion failed!")
        print("=" * 60)

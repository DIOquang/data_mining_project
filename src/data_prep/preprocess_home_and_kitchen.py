"""
Tiền xử lý dữ liệu cho meta_Home_and_Kitchen.jsonl
Contains data cleaning, transformation, and feature engineering
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import re
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HomeAndKitchenPreprocessor:
    """
    Tiền xử lý dữ liệu cho dataset Home and Kitchen
    """
    
    def __init__(self, input_file: str, output_file: str):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Tải dữ liệu từ JSONL file"""
        logger.info(f"📖 Đang tải dữ liệu từ {self.input_file.name}...")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"File không tìm thấy: {self.input_file}")
        
        self.df = pd.read_json(self.input_file, lines=True)
        logger.info(f"✅ Đã tải {len(self.df):,} records")
        logger.info(f"📊 Shape: {self.df.shape}")
        
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Xử lý các giá trị bị thiếu"""
        logger.info("🔧 Đang xử lý các giá trị bị thiếu...")
        
        initial_nulls = self.df.isnull().sum().sum()
        
        # Xử lý từng cột
        self.df['average_rating'].fillna(0.0, inplace=True)
        self.df['rating_number'].fillna(0, inplace=True)
        self.df['price'].fillna(0.0, inplace=True)
        
        # Xử lý features, description, images, videos
        self.df['features'] = self.df['features'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.df['description'] = self.df['description'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.df['images'] = self.df['images'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.df['videos'] = self.df['videos'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.df['categories'] = self.df['categories'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        # Xử lý details
        self.df['details'] = self.df['details'].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
        
        # Điền giá trị mặc định cho store
        self.df['store'].fillna('Unknown', inplace=True)
        self.df['main_category'].fillna('Unknown', inplace=True)
        self.df['parent_asin'].fillna('', inplace=True)
        
        final_nulls = self.df.isnull().sum().sum()
        logger.info(f"✅ Xử lý xong: {initial_nulls} → {final_nulls} missing values")
        
        return self.df
    
    def clean_text(self) -> pd.DataFrame:
        """Làm sạch text data"""
        logger.info("🧹 Đang làm sạch text data...")
        
        def clean_string(s):
            if not isinstance(s, str):
                return s
            # Loại bỏ khoảng trắng dự phòng
            s = ' '.join(s.split())
            # Loại bỏ ký tự đặc biệt không cần thiết nhưng giữ lại từ viết tắt
            s = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', s)
            return s.strip()
        
        self.df['title'] = self.df['title'].apply(clean_string)
        self.df['store'] = self.df['store'].apply(clean_string)
        self.df['main_category'] = self.df['main_category'].apply(clean_string)
        
        logger.info("✅ Đã làm sạch text data")
        return self.df
    
    def handle_outliers(self) -> pd.DataFrame:
        """Xử lý các giá trị ngoại lệ"""
        logger.info("📊 Đang xử lý các giá trị ngoại lệ...")
        
        # Rating phải nằm trong [0, 5]
        self.df['average_rating'] = self.df['average_rating'].clip(0, 5)
        
        # Giá phải dương
        self.df['price'].loc[self.df['price'] < 0] = 0
        
        # Số đánh giá phải không âm
        self.df['rating_number'].loc[self.df['rating_number'] < 0] = 0
        
        logger.info("✅ Đã xử lý outliers")
        return self.df
    
    def create_features(self) -> pd.DataFrame:
        """Tạo các feature mới từ dữ liệu hiện có"""
        logger.info("🆕 Đang tạo các feature mới...")
        
        # Số lượng đặc điểm
        self.df['num_features'] = self.df['features'].apply(len)
        
        # Số lượng mô tả
        self.df['num_descriptions'] = self.df['description'].apply(len)
        
        # Số lượng ảnh
        self.df['num_images'] = self.df['images'].apply(len)
        
        # Số lượng video
        self.df['num_videos'] = self.df['videos'].apply(len)
        
        # Số lượng danh mục
        self.df['num_categories'] = self.df['categories'].apply(len)
        
        # Số lượng chi tiết kỹ thuật
        self.df['num_details'] = self.df['details'].apply(len)
        
        # Độ dài title
        self.df['title_length'] = self.df['title'].apply(len)
        
        # Có hình ảnh không
        self.df['has_images'] = self.df['num_images'] > 0
        
        # Có video không
        self.df['has_videos'] = self.df['num_videos'] > 0
        
        # Có được đánh giá không
        self.df['has_rating'] = self.df['rating_number'] > 0
        
        # Giá > 0
        self.df['has_price'] = self.df['price'] > 0
        
        # Rating category (Low, Medium, High, Not rated)
        def categorize_rating(rating):
            if rating == 0:
                return 'Not Rated'
            elif rating < 2.5:
                return 'Low'
            elif rating < 4.0:
                return 'Medium'
            else:
                return 'High'
        
        self.df['rating_category'] = self.df['average_rating'].apply(categorize_rating)
        
        # Price category (Free, Budget, Mid-range, Premium)
        def categorize_price(price):
            if price == 0:
                return 'Unknown'
            elif price < 25:
                return 'Budget'
            elif price < 100:
                return 'Mid-range'
            else:
                return 'Premium'
        
        self.df['price_category'] = self.df['price'].apply(categorize_price)
        
        # Engagement score (combination of ratings and price presence)
        self.df['engagement_score'] = (
            (self.df['rating_number'] / (self.df['rating_number'].max() + 1)) * 0.5 +
            (self.df['num_images'] / (self.df['num_images'].max() + 1)) * 0.3 +
            (self.df['num_videos'] / (self.df['num_videos'].max() + 1)) * 0.2
        )
        
        logger.info("✅ Đã tạo 15 feature mới")
        return self.df
    
    def validate_data(self) -> Dict:
        """Kiểm tra chất lượng dữ liệu"""
        logger.info("✔️ Đang kiểm tra chất lượng dữ liệu...")
        
        validation_report = {
            'total_records': len(self.df),
            'null_values': self.df.isnull().sum().to_dict(),
            'duplicate_rows': self.df.duplicated().sum(),
            'duplicate_asins': self.df['parent_asin'].duplicated().sum(),
            'price_statistics': {
                'min': float(self.df['price'].min()),
                'max': float(self.df['price'].max()),
                'mean': float(self.df['price'].mean()),
                'median': float(self.df['price'].median()),
            },
            'rating_statistics': {
                'min': float(self.df['average_rating'].min()),
                'max': float(self.df['average_rating'].max()),
                'mean': float(self.df['average_rating'].mean()),
                'median': float(self.df['average_rating'].median()),
            },
            'rating_count_statistics': {
                'min': int(self.df['rating_number'].min()),
                'max': int(self.df['rating_number'].max()),
                'mean': float(self.df['rating_number'].mean()),
                'median': float(self.df['rating_number'].median()),
            }
        }
        
        logger.info("✅ Kiểm tra xong")
        return validation_report
    
    def save_data(self, format: str = 'parquet') -> bool:
        """
        Lưu dữ liệu đã xử lý
        
        Args:
            format: 'parquet' hoặc 'csv'
        """
        logger.info(f"💾 Đang lưu dữ liệu (định dạng: {format})...")
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'parquet':
                self.df.to_parquet(self.output_file, index=False, compression='snappy')
            elif format == 'csv':
                self.df.to_csv(self.output_file, index=False)
            else:
                raise ValueError(f"Định dạng không hỗ trợ: {format}")
            
            file_size_mb = self.output_file.stat().st_size / (1024**2)
            logger.info(f"✅ Đã lưu thành công: {self.output_file}")
            logger.info(f"   Kích thước: {file_size_mb:.2f} MB")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu file: {e}")
            return False
    
    def print_summary(self, validation_report: Dict):
        """In tóm tắt xử lý"""
        print("\n" + "="*80)
        print("📋 TÓM TẮT TIỀN XỬ LÝ DỮ LIỆU HOME AND KITCHEN")
        print("="*80)
        
        print(f"\n📊 SỐ LƯỢNG:")
        print(f"   • Tổng records: {validation_report['total_records']:,}")
        print(f"   • Duplicate rows: {validation_report['duplicate_rows']:,}")
        print(f"   • Duplicate ASIN: {validation_report['duplicate_asins']:,}")
        
        print(f"\n💰 THỐNG KÊ GIÁ:")
        stats = validation_report['price_statistics']
        print(f"   • Min: ${stats['min']:.2f}")
        print(f"   • Max: ${stats['max']:.2f}")
        print(f"   • Mean: ${stats['mean']:.2f}")
        print(f"   • Median: ${stats['median']:.2f}")
        
        print(f"\n⭐ THỐNG KÊ RATING:")
        stats = validation_report['rating_statistics']
        print(f"   • Min: {stats['min']:.1f}")
        print(f"   • Max: {stats['max']:.1f}")
        print(f"   • Mean: {stats['mean']:.2f}")
        print(f"   • Median: {stats['median']:.2f}")
        
        print(f"\n📈 THỐNG KÊ SỐ ĐÁNH GIÁ:")
        stats = validation_report['rating_count_statistics']
        print(f"   • Min: {stats['min']:,}")
        print(f"   • Max: {stats['max']:,}")
        print(f"   • Mean: {stats['mean']:.0f}")
        print(f"   • Median: {stats['median']:.0f}")
        
        print(f"\n🆕 FEATURE MỚI ĐƯỢC TẠO:")
        print(f"   • num_features, num_descriptions, num_images, num_videos")
        print(f"   • num_categories, num_details, title_length")
        print(f"   • has_images, has_videos, has_rating, has_price")
        print(f"   • rating_category, price_category")
        print(f"   • engagement_score")
        
        print(f"\n📁 KẾT QUẢ:")
        print(f"   • Output: {self.output_file}")
        print(f"   • Columns: {len(self.df.columns)}")
        print("="*80 + "\n")
    
    def run(self, output_format: str = 'parquet'):
        """Chạy toàn bộ pipeline tiền xử lý"""
        logger.info("🚀 Bắt đầu pipeline tiền xử lý dữ liệu...")
        
        try:
            # Load dữ liệu
            self.load_data()
            
            # Xử lý
            self.handle_missing_values()
            self.clean_text()
            self.handle_outliers()
            self.create_features()
            
            # Kiểm tra
            validation_report = self.validate_data()
            
            # Lưu
            success = self.save_data(format=output_format)
            
            # In tóm tắt
            self.print_summary(validation_report)
            
            return success
        
        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình xử lý: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Cấu hình đường dẫn
    PROJECT_ROOT = Path(__file__).parent.parent
    
    INPUT_FILE = PROJECT_ROOT / "data/raw/meta_categories/meta_Home_and_Kitchen.jsonl"
    OUTPUT_FILE = PROJECT_ROOT / "data/processed/meta_categories/meta_Home_and_Kitchen_processed.parquet"
    
    # Khởi tạo preprocessor
    preprocessor = HomeAndKitchenPreprocessor(
        input_file=str(INPUT_FILE),
        output_file=str(OUTPUT_FILE)
    )
    
    # Chạy pipeline
    success = preprocessor.run(output_format='parquet')
    
    if success:
        logger.info("✨ Tiền xử lý dữ liệu hoàn tất thành công!")
    else:
        logger.error("❌ Tiền xử lý dữ liệu thất bại!")

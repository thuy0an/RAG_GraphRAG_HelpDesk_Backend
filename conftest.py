import sys
import os

# Thêm src/ vào sys.path để các import dạng "src.X" và "Features.X" đều hoạt động
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# YOLOv8 Waste Detection API

## Giới thiệu
Dự án cung cấp một dịch vụ REST API dùng **YOLOv8** để nhận diện và đếm số lượng các vật thể thuộc nhóm rác tái chế/phổ biến (ví dụ: chai nhựa, nắp chai, ống hút, giấy bìa,...). Ứng dụng hướng đến giáo dục môi trường (AI for Green) và các tình huống phân loại rác tự động/ bán tự động.

## Thành tích
- Giải Nhất Tin học trẻ vòng khu vực  
- Giải Nhì Tin học trẻ Quốc gia

## Công nghệ sử dụng
- Python, FastAPI, Uvicorn
- Ultralytics YOLOv8
- Pillow (xử lý ảnh)
- (Tuỳ chọn) `rembg` để loại nền/tiền xử lý ảnh
- cURL / Swagger UI cho kiểm thử API

## Cấu trúc thư mục
.
├── best.pt # Trained YOLOv8 model (custom dataset)
├── requirements.txt # Danh sách thư viện
├── yolo_fastapi_server.py # Mã nguồn FastAPI server
└── README.md # Tài liệu này

## Yêu cầu hệ thống
- Python 3.9+ (khuyến nghị 3.10/3.11)
- pip, venv
- GPU là lợi thế nhưng không bắt buộc

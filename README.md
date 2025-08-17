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

css
Sao chép
Chỉnh sửa

## Yêu cầu hệ thống
- Python 3.9+ (khuyến nghị 3.10/3.11)
- pip, venv
- GPU là lợi thế nhưng không bắt buộc

## Cài đặt và chạy thử

1) Clone mã nguồn:

git clone https://github.com/iamquyet0201/<ten-repo>.git
cd <ten-repo>
Tạo môi trường ảo (khuyến nghị):

bash
Sao chép
Chỉnh sửa
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
Cài đặt thư viện:

bash
Sao chép
Chỉnh sửa
pip install -r requirements.txt
Khởi chạy API:

bash
Sao chép
Chỉnh sửa
uvicorn yolo_fastapi_server:app --host 0.0.0.0 --port 8000
Sau khi chạy, tài liệu tương tác:

Swagger: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

API Reference
1) Dự đoán trên một ảnh
Endpoint: POST /predict

Body: multipart/form-data với trường file là ảnh đầu vào

Trả về: JSON gồm danh sách vật thể phát hiện, số lượng, và (tuỳ cấu hình) ảnh gắn nhãn ở dạng Base64

Ví dụ với cURL:

bash
Sao chép
Chỉnh sửa
curl -X POST "http://localhost:8000/predict" \
     -F "file=@sample.jpg"
Phản hồi mẫu:

json
Sao chép
Chỉnh sửa
{
  "items": [
    {"name": "plastic_bottle", "quantity": 2},
    {"name": "bottle_cap",   "quantity": 5},
    {"name": "straw",        "quantity": 3}
  ],
  "annotated_image": "<base64-encoded-image>"
}
Ghi chú:

Trường tên file trong form có thể là file hoặc image tuỳ bạn định nghĩa trong mã nguồn. Hãy đồng bộ với yolo_fastapi_server.py.

annotated_image có thể tắt nếu không cần để giảm dung lượng/phản hồi.

2) Health check (nếu có)
Endpoint: GET /health

Trả về: { "status": "ok" }

Nhãn và ánh xạ lớp
Cập nhật theo mô hình của bạn. Ví dụ:

makefile
Sao chép
Chỉnh sửa
0: chai_nuoc
1: nap_chai
2: que_de_luoi
3: que_xien
4: bong_bay
5: nit
6: giay_mau
7: ong_hut
8: bia_cat_tong
Nếu class mapping nằm trong mã (ví dụ một dict trong yolo_fastapi_server.py), hãy liệt kê rõ ở đây để người dùng đối chiếu.

Mô hình
Kiến trúc: YOLOv8 (Ultralytics)

Dữ liệu: bộ dữ liệu tuỳ biến gồm các vật thể rác/đồ tái chế phổ biến

Huấn luyện: fine-tune từ pretrained YOLOv8 trên tập dữ liệu nói trên

Kích thước ảnh, số epoch, augmentations… có thể tuỳ chỉnh theo nhu cầu (khuyến nghị ghi lại trong nhật ký huấn luyện để tái lập)

Hướng dẫn triển khai (tuỳ chọn)
Render / Railway / HF Spaces
Đóng gói requirements.txt

Dùng lệnh khởi động:

nginx
Sao chép
Chỉnh sửa
uvicorn yolo_fastapi_server:app --host 0.0.0.0 --port $PORT
Đảm bảo file mô hình best.pt được đưa kèm (hoặc tải khi khởi động nếu bạn lưu ở storage khác).

ngrok (chạy local, public tạm thời)
bash
Sao chép
Chỉnh sửa
uvicorn yolo_fastapi_server:app --host 0.0.0.0 --port 8000
ngrok http 8000
# Lấy URL public từ ngrok để gọi API từ bên ngoài
Tối ưu hiệu năng và độ ổn định
Giới hạn kích thước ảnh đầu vào (ví dụ: resize/preprocess về 640 hoặc 720) để giảm độ trễ.

Tắt rembg nếu không cần loại nền; nếu dùng, có thể áp dụng trên ảnh đã resize.

Khởi chạy nhiều worker khi triển khai thực tế (ví dụ: Gunicorn + Uvicorn workers).

Nạp sẵn mô hình khi start server, tránh load lại theo request.

Ghi log lỗi tổng quát để tránh server crash, trả về mã lỗi chuẩn (400/422/500).

Kiểm thử nhanh bằng JavaScript (fetch)
javascript
Sao chép
Chỉnh sửa
const form = new FormData();
form.append('file', myFile); // File từ <input type="file">

const res = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: form
});
const data = await res.json();
console.log(data.items); // danh sách vật thể và số lượng
Định hướng phát triển
Hỗ trợ batch ảnh và/hoặc video stream

Bổ sung endpoint trả ảnh đã gắn nhãn dưới dạng file (không Base64)

Thêm thống kê độ tin cậy theo từng nhãn

Dashboard trực quan hoá kết quả

Tác giả
Chủ nhiệm dự án: iamquyet0201

Thành tích: Giải Nhất Tin học trẻ vòng khu vực; Giải Nhì Tin học trẻ Quốc gia

Giấy phép
Mặc định: tất cả quyền được bảo lưu. Vui lòng liên hệ tác giả nếu muốn sử dụng mô hình/dữ liệu cho mục đích thương mại hoặc phân phối lại.

css
Sao chép
Chỉnh sửa

Bạn cứ dán nguyên khối này vào `README.md`. Nếu muốn, mình có thể thêm riêng phần “Kết quả minh hoạ” (ảnh đầu vào và ảnh gắn nhãn) để README trông nổi bật hơn.







Hỏi ChatGPT

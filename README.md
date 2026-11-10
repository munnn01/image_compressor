# NPP với ImageCompressor của tác giả

Folder này độc lập với `paper_fidelity`. Proxy là **ImageCompressor(192,192)** từ [PreprocessingICM](https://github.com/XingtongGe/PreprocessingICM/tree/6a77c4633cd5bb6750c77ea9d213d38ffc88ee7a), gồm hyperprior và context model. Không có dependency, khởi tạo dự phòng hay chuyển đổi checkpoint mbt2018/CompressAI. Checkpoint thiếu tensor hoặc sai kiến trúc bị từ chối bằng strict loading.

**Hiện chưa kèm checkpoint RD pretrained gốc đã được xác minh.** Mã nguồn đúng kiến trúc không có nghĩa là đã có trọng số pretrained. Nếu chưa có checkpoint native, cần chạy `pretrain-proxy` trước. Các checkpoint của pipeline mbt2018 cũ không dùng được trong folder này. Chưa có kết quả huấn luyện đầy đủ hoặc BD-rate của nhánh này.

## Cài và kiểm tra

```powershell
cd D:\STUDY\LAB\TEST\imagecompressor_reproduction
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m npp_proxy --help
```

Cần `bpgenc` và `bpgdec` của libbpg 0.9.8 trên PATH, hoặc truyền `--bpgenc <path> --bpgdec <path>`. Kaggle notebook tự build bản JCTVC; local smoke dùng bản Windows của nhà phát hành. Bitstream giữa các build cần được kiểm tra trước khi so sánh số liệu.

## Thứ tự thực nghiệm

1. Chuẩn bị Flickr30k: manifest cố định khoảng 800.000 crop 256×256, dành riêng ít nhất 64 ảnh nguồn cho calibration. Không dùng ảnh ImageNet validation để chọn checkpoint.
2. Có hoặc tự huấn luyện các checkpoint **native RD pretrained** ở nhiều λ. Calibration chọn checkpoint có PSNR gần BPG tại từng QP.
3. Fine-tune một proxy native cho mỗi QP để khớp ảnh giải mã BPG thật; mỗi QP đủ 100.000 update.
4. Train fixed NPP tại QP34 đủ 30.000 update; sau đó chuyển trọng số sang adaptive NPP và train 25.000 update, chọn ngẫu nhiên một QP cho mỗi batch.
5. Đánh giá BPG và NPP+BPG trên cùng ảnh, QP, task weights và codec. Tính BD-rate theo top-1 hoặc COCO AP, chỉ trên miền chất lượng giao nhau.

| Giai đoạn | Ngân sách 1/4 bài báo | LR đầu | Đổi sang 1e-5 sau |
|---|---:|---:|---:|
| Proxy BPG / mỗi QP | 100.000 | 1e-4 | 80.000 update |
| Fixed NPP | 30.000 | 1e-4 | 20.000 update |
| Adaptive NPP | 25.000 | 1e-4 | 15.000 update |

Update ngay sau mốc dùng LR mới. Lịch là step decay, được tính bằng tổng update qua các phiên resume. Adam và vị trí lấy mẫu được lưu cùng checkpoint. `--run-updates` giới hạn một phiên; `--steps` là tổng ngân sách. Không thay `--steps` giữa các phiên; không dùng checkpoint thiếu bước để đi qua cổng chuyển giai đoạn. Fixed bắt buộc QP34, BPG thật level 8; checkpoint chuyển sang adaptive phải khớp task, backbone và task weights.

QP: 28, 31, 34, 37, 41. λ proxy: 4096, 2048, 1024, 512, 256. λ task: 0.5, 1, 2, 4, 8. β distortion của NPP = 0.5. Loss proxy là estimated rate + λ·MSE với ảnh BPG; RD pretraining dùng MSE trước clipping của native model. Algorithm 1 dùng **RGB và số bit từ BPG thật trong forward**, đạo hàm RGB/rate từ proxy, đóng băng proxy và task network. NPP dùng khởi tạo mặc định của tác giả.

## Lệnh thực hiện

Các đường dẫn dưới đây là ví dụ cần thay bằng dữ liệu/trọng số thật. Chạy từ folder này.

```powershell
python -m npp_proxy make-crops --data D:/data/flickr30k --output outputs/crops.json
```

Nếu có pretrained native, đặt các ứng viên vào `outputs/rd_candidates/`. Nếu chưa có, ví dụ dưới đây chạy RD pretraining tại một λ, với **ngân sách do người chạy chọn**, tách biệt với 100.000 bước fine-tune BPG:

```powershell
# 400.000 ở đây là ví dụ ngân sách tự chọn, không phải số bước RD pretraining được bài báo xác nhận.
python -m npp_proxy pretrain-proxy --data outputs/crops.json --qp 34 --steps 400000 --decay 320000 --output outputs/rd_candidates/rd34.pt
python -m npp_proxy calibrate-proxy --data outputs/crops.json --candidate-dir outputs/rd_candidates --output outputs/calibration.json
python -m npp_proxy train-proxy --data outputs/crops.json --qp 34 --calibration outputs/calibration.json --output outputs/bank/qp34.pt
```

Huấn luyện RD nhiều λ để calibration có lựa chọn; một checkpoint duy nhất không tạo thành bank pretrained nhiều chất lượng. Đọc `calibration.json` và audit trước khi train NPP. Lặp `train-proxy` cho cả năm QP, mỗi checkpoint phải là native và đủ 100.000 update. Có thể dùng `--init <native-rd.pt>` nếu việc chọn checkpoint đã được thực hiện bên ngoài. Với state_dict thuần, mã kiểm tra kiến trúc đầy đủ nhưng không chứng minh lịch sử pretraining; người cung cấp phải xác nhận nguồn gốc.

```powershell
python -m npp_proxy audit-proxy --data outputs/crops.json --proxy-dir outputs/bank --output outputs/audit.json
python -m npp_proxy train-npp --stage fixed --task classification --data D:/data/imagenet/train --proxy-dir outputs/bank --output outputs/fixed.pt
python -m npp_proxy train-npp --stage adaptive --task classification --data D:/data/imagenet/train --proxy-dir outputs/bank --fixed outputs/fixed.pt --output outputs/adaptive.pt
python -m npp_proxy evaluate --task classification --data D:/data/imagenet/val --npp outputs/adaptive.pt --output outputs/evaluation
```

ImageNet cần đủ 1.000 thư mục WNID và nhãn đúng. Evaluation mặc định ResNet50 V1, resize 256/center crop 224; có thể chọn backbone chuyển giao. Không dùng `--limit` khi báo cáo kết quả đầy đủ. Detection dùng COCO và `--annotations`; mặc định FCOS, hỗ trợ Faster R-CNN/RetinaNet khi đánh giá.

Ví dụ chia một giai đoạn thành phiên và tiếp tục **cùng checkpoint native**:

```powershell
python -m npp_proxy train-proxy --data outputs/crops.json --qp 34 --init outputs/rd_candidates/rd34.pt --run-updates 10000 --output outputs/bank/qp34.pt
python -m npp_proxy train-proxy --data outputs/crops.json --qp 34 --resume outputs/bank/qp34.pt --run-updates 10000 --output outputs/bank/qp34.pt
```

Khi chuyển máy, manifest có `--image-root` để đổi gốc ảnh. Resume hiện kiểm tra đường dẫn dữ liệu/config chặt chẽ; trên Kaggle nên dùng cùng đường dẫn output và input qua các phiên. Trước khi resume, chép checkpoint vào output path ghi được; không ghi lên `/kaggle/input`.

## Đóng gói Kaggle

RD pretraining đã được chọn ngân sách riêng **100.000 bước/QP**, giảm LR sau 80.000; đây là ngân sách thực nghiệm được người dùng chốt, không phải ngân sách RD pretraining đã xác nhận từ bài báo. Có builder riêng để chạy từng QP độc lập, kiểm tra native GPU trước khi train và nối checkpoint theo phiên:

```powershell
python kaggle/prepare_rd.py --owner YOUR_OWNER --qps 34 --run-updates 16384 --output kaggle/generated/qp34
kaggle kernels push -p kaggle/generated/qp34
# Sau khi phiên 1 COMPLETE, giữ nguyên tổng đích và nối đúng trọng số/Adam:
python kaggle/prepare_rd.py --owner YOUR_OWNER --qps 34 --run-updates 16384 --previous YOUR_OWNER/imagecompressor-native-rd-qp34/1 --output kaggle/generated/qp34
kaggle kernels push -p kaggle/generated/qp34
```

Lặp cho QP28/31/37/41 trên các tài khoản được phép sử dụng. Mỗi job chỉ train QP được chỉ định; 16.384 là số update tối đa của một phiên, không phải tổng ngân sách. `gpu_smoke.json` và `progress.json` ghi kiến trúc, gradient kiểm tra, nguồn code, số bước, LR và hoàn tất giai đoạn. Khi RD đạt đủ 100.000, cần audit/calibration rồi mới bắt đầu BPG fine-tuning. Danh sách triển khai hiện tại nằm trong [DEPLOYMENT.md](docs/DEPLOYMENT.md).

`kaggle/prepare.py` tạo notebook **public**, GPU, Internet, chứa toàn bộ mã native và SHA256. Script chỉ tạo gói, không đăng nhập tài khoản hay tự submit. Không chứa pool tài khoản/token/checkpoint. Notebook tạo môi trường Python 3.12 + Torch 2.5.1 và build BPG JCTVC; cần Kaggle cho phép Internet và GPU khả dụng.

Sửa đường dẫn dataset trong `kaggle/commands.example.json`, gắn dataset Flickr30k, native RD bank và ImageNet tương ứng, rồi:

```powershell
python kaggle/prepare.py --id YOUR_OWNER/native-imagecompressor-npp --commands kaggle/commands.example.json --dataset YOUR_OWNER/flickr30k --dataset YOUR_OWNER/native-rd-checkpoints --dataset YOUR_OWNER/imagenet --output kaggle/generated
kaggle kernels push -p kaggle/generated
```

Danh sách lệnh là các mảng argv, chạy tuần tự, dừng khi một lệnh lỗi. `{WORK}`, `{BPGENC}`, `{BPGDEC}` được thay bằng đường dẫn trong notebook. Template dùng `--init` cho năm checkpoint RD đã được chọn; thay bằng `--calibration` nếu đã chuẩn bị report hợp lệ. Nên tạo một gói cho từng giai đoạn/QP, tải output checkpoint rồi resume khi cần; template toàn chuỗi không bảo đảm vừa giới hạn phiên Kaggle. Số bước mỗi phiên có thể điều chỉnh bằng `--run-updates` mà vẫn giữ lịch LR tổng.

## Mức độ khớp và giới hạn

- Nguồn tác giả được pin theo commit và kiểm tra 19 Git blob; `upstream/manifest.json` lưu hash từng file. `upstream/port_manifest.json` ghi hash và các sửa đổi tương thích. Chi tiết ở [PROVENANCE.md](docs/PROVENANCE.md).
- Proxy/NPP dùng module gốc; local kiểm thử CPU cho đầu ra, estimated rate và gradient đầu vào bằng mã gốc. Chưa xác minh parity GPU hoặc chạy đầy đủ trên Kaggle cho nhánh native này.
- Ngân sách 1/4 và các mốc LR tương ứng là lựa chọn đã thống nhất. RD pretraining không được tính vào ba ngân sách đó; chưa biết ngân sách/pretrained artifact chính xác của tác giả.
- Adaptive giữ QA MLP và khởi tạo của tác giả, đưa QP/51 vào theo mô tả QP của bài báo. Mã tích hợp công khai truyền lambda; phép chuẩn hóa này là lựa chọn triển khai, chưa thể coi là khớp tuyệt đối.
- Batch mặc định proxy=4 (config minh họa của tác giả), NPP=2 để triển khai; batch và task weights TorchVision V1 chưa được xác nhận trùng hoàn toàn thực nghiệm tác giả. Crop manifest tái lập tập crop ảo, không phải danh sách crop gốc của tác giả.
- BD-rate âm nghĩa là giảm bitrate tại cùng chất lượng task. Không có miền giao nhau hoặc đường chất lượng không hợp lệ thì trả trạng thái không tính được; không ngoại suy để tạo số đẹp. Bản này không bảo đảm bitrate thấp hơn BPG hoặc top-1 cao hơn trước khi có đánh giá thật.

Smoke thực dùng dữ liệu tổng hợp và task Conv2d đóng băng, chưa chạy task weights ResNet/FCOS đầy đủ; không thay thế training/benchmark:

```powershell
python tools/smoke.py --bpgenc /path/bpgenc --bpgdec /path/bpgdec --output docs/smoke.json
```

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY

# Device Selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Download and Load Model
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

net = ARCH_REGISTRY.get('CodeFormer')(
    dim_embd=768,  # Enhanced embedding dimensions
    codebook_size=2048,  # Increased codebook size for finer details
    n_head=12,  # More attention heads
    n_layers=12,  # Deeper architecture
    connect_list=['32', '64', '128', '256', '512']
).to(device)

ckpt_path = load_file_from_url(
    url=pretrain_model_url['restoration'],
    model_dir='weights/CodeFormer',
    progress=True,
    file_name=None
)
checkpoint = torch.load(ckpt_path, map_location=device)['params_ema']
net.load_state_dict(checkpoint)
net.eval()

def get_face_helper(img_shape):
    """Adaptive Face Restoration Helper."""
    height, width = img_shape[:2]
    upscale_factor = 1 if max(height, width) > 2048 else 2
    face_size = 1024 if max(height, width) > 2048 else 512
    return FaceRestoreHelper(
        upscale_factor=upscale_factor,
        face_size=face_size,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='jpg',
        use_parse=True,
        device=device
    )

def enhance_face(face_img, w_values):
    """Enhance a Single Face."""
    face_tensor = img2tensor(face_img / 255., bgr2rgb=True, float32=True).to(device)
    normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    face_tensor = face_tensor.unsqueeze(0)
    restored_faces = []

    for w in w_values:
        with torch.no_grad():
            output = net(face_tensor, w=w, adain=True)[0]
            restored_faces.append(tensor2img(output, rgb2bgr=True, min_max=(-1, 1)))

    return np.mean(restored_faces, axis=0).astype('uint8')

def _enhance_img(img, w_values):
    """Enhance an Image with CodeFormer."""
    face_helper = get_face_helper(img.shape)
    face_helper.clean_all()
    face_helper.read_image(img)
    num_faces = face_helper.get_face_landmarks_5(only_center_face=False, resize=1280, eye_dist_threshold=2)

    if num_faces == 0:
        return img  # No faces detected

    face_helper.align_warp_face()
    for cropped_face in face_helper.cropped_faces:
        restored_face = enhance_face(cropped_face, w_values)
        restored_face = cv2.bilateralFilter(restored_face, 15, 90, 90)  # Stronger noise reduction
        restored_face = cv2.addWeighted(restored_face, 2.0, restored_face, -1.0, 0)  # Sharpening
        face_helper.add_restored_face(restored_face)

    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()

    # Global Enhancements
    restored_img = cv2.detailEnhance(restored_img, sigma_s=15, sigma_r=0.2)
    restored_img = cv2.GaussianBlur(restored_img, (5, 5), 0)
    lab = cv2.cvtColor(restored_img, cv2.COLOR_BGR2Lab)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    restored_img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    return restored_img

def enhance_image(input_image_path, w_values=[0.8, 0.85, 0.9]):
    """Enhance an Image from File."""
    input_path = Path(input_image_path)
    output_path = input_path.with_name(f"{input_path.stem}.enhanced.jpg")

    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {input_image_path}")

    restored_img = _enhance_img(img, w_values=w_values)
    os.makedirs(output_path.parent, exist_ok=True)
    cv2.imwrite(str(output_path), restored_img)
    print(f"Enhanced image saved to: {output_path}")
    return str(output_path)

def enhance_image_memory(img, w_values=[0.8, 0.85, 0.9]):
    """Enhance an Image in Memory."""
    return _enhance_img(img, w_values=w_values)

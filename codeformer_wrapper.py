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

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Download and load the model
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_stage2.pth',
}

net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                       connect_list=['32', '64', '128', '256']).to(device)

ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                                model_dir='weights/CodeFormer', progress=True, file_name=None)
checkpoint = torch.load(ckpt_path, map_location=device)['params_ema']
net.load_state_dict(checkpoint)
net.eval()

# Updated function to increase upscale factor
def get_face_helper(img_shape):
    height, width = img_shape[:2]
    upscale_factor = 2  # Increase resolution consistently for better quality
    face_size = 1024 if max(height, width) > 1024 else 512  # Maintain adaptive face size

    return FaceRestoreHelper(
        upscale_factor=upscale_factor,
        face_size=face_size,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='jpg',
        use_parse=True,
        device=device
    )

# Additional sharpening step in enhance_face function
def enhance_face(face_img, w_values):
    face_tensor = img2tensor(face_img / 255., bgr2rgb=True, float32=True).to(device)
    normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    face_tensor = face_tensor.unsqueeze(0)

    restored_faces = []
    for w in w_values:
        with torch.no_grad():
            output = net(face_tensor, w=w, adain=True)[0]
            restored_faces.append(tensor2img(output, rgb2bgr=True, min_max=(-1, 1)))

    avg_face = np.mean(restored_faces, axis=0).astype('uint8')

    # Enhanced super-resolution: Sharpen and add more detail
    avg_face = cv2.bilateralFilter(avg_face, 9, 75, 75)
    avg_face = cv2.detailEnhance(avg_face, sigma_s=10, sigma_r=0.15)  # Detail enhancement
    avg_face = cv2.addWeighted(avg_face, 1.2, avg_face, -0.2, 10)  # Sharpening with contrast
    return avg_face

def seamless_blending(input_img, restored_img, mask):
    center = (input_img.shape[1] // 2, input_img.shape[0] // 2)
    return cv2.seamlessClone(restored_img, input_img, mask, center, cv2.NORMAL_CLONE)

def _enhance_img(img: np.ndarray, w_values=[0.7, 0.8, 0.9]) -> np.ndarray:
    face_helper = get_face_helper(img.shape)
    face_helper.clean_all()
    face_helper.read_image(img)
    num_faces = face_helper.get_face_landmarks_5(only_center_face=False, resize=1280, eye_dist_threshold=2)
    if num_faces == 0:
        return img

    face_helper.align_warp_face()

    for cropped_face in face_helper.cropped_faces:
        restored_face = enhance_face(cropped_face, w_values)
        face_helper.add_restored_face(restored_face)

    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()

    # Global enhancements
    lab = cv2.cvtColor(restored_img, cv2.COLOR_BGR2Lab)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    restored_img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # Noise addition for realism
    noise = np.random.normal(0, 1, restored_img.shape).astype('uint8')
    restored_img = cv2.addWeighted(restored_img, 0.98, noise, 0.02, 0)

    return restored_img

def enhance_image(input_image_path: str, w_values=[0.7, 0.8, 0.9]) -> str:
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

def enhance_image_memory(img: np.ndarray, w_values=[0.7, 0.8, 0.9]) -> np.ndarray:
    return _enhance_img(img, w_values=w_values)

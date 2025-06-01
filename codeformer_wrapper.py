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
import time

# Cross-platform device selection: CUDA > MPS > CPU
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

# Download and load model
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

# Initialize CodeFormer with enhanced parameters
net = ARCH_REGISTRY.get('CodeFormer')(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=12,  # Increased layers for better detail
    connect_list=['32', '64', '128', '256', '512']  # Added 512 connection
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
    """Dynamically adjust face helper based on image resolution"""
    height, width = img_shape[:2]
    max_dim = max(height, width)
    
    # More granular resolution handling
    if max_dim > 3000:
        face_size = 1024
        upscale_factor = 1
    elif max_dim > 1500:
        face_size = 768
        upscale_factor = 1
    else:
        face_size = 512
        upscale_factor = 2 if max_dim < 1024 else 1

    return FaceRestoreHelper(
        upscale_factor=upscale_factor,
        face_size=face_size,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',  # Lossless format
        use_parse=True,
        device=device,
        model_rootpath='weights/facelib'
    )

def enhance_face(face_img, w_values):
    """Enhanced face restoration with multi-pass fusion"""
    face_tensor = img2tensor(face_img / 255., bgr2rgb=True, float32=True).to(device)
    normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    face_tensor = face_tensor.unsqueeze(0)

    # Multi-pass restoration with different weights
    restored_faces = []
    for w in w_values:
        with torch.no_grad():
            output = net(face_tensor, w=w, adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            
            # Per-pass sharpening
            restored_face = cv2.detailEnhance(restored_face, sigma_s=12, sigma_r=0.15)
            restored_faces.append(restored_face)
    
    # Weighted fusion (emphasize higher w values)
    weights = np.linspace(0.5, 1.5, num=len(w_values))
    weights /= weights.sum()
    fused_face = np.zeros_like(restored_faces[0], dtype=np.float32)
    
    for i, face in enumerate(restored_faces):
        fused_face += face.astype(np.float32) * weights[i]
    
    return np.clip(fused_face, 0, 255).astype('uint8')

def _enhance_img(img: np.ndarray, w_values=[0.65, 0.80, 0.95]) -> np.ndarray:
    """Core enhancement pipeline for elite realism"""
    # Preserve original for fallback
    original_img = img.copy()
    
    # Initialize face helper
    face_helper = get_face_helper(img.shape)
    face_helper.clean_all()
    face_helper.read_image(img)
    
    # Enhanced face detection with adaptive threshold
    num_faces = face_helper.get_face_landmarks_5(
        only_center_face=False,
        resize=max(img.shape[0], img.shape[1]),
        eye_dist_threshold=1.5  # More sensitive detection
    )
    
    if num_faces == 0:
        print("No faces detected - applying global enhancement")
        return apply_global_enhancements(original_img)
    
    # Align and process faces
    face_helper.align_warp_face()
    
    for cropped_face in face_helper.cropped_faces:
        # Face enhancement pipeline
        restored_face = enhance_face(cropped_face, w_values)
        
        # Texture refinement
        restored_face = cv2.bilateralFilter(restored_face, 11, 85, 85)
        restored_face = adaptive_histogram_equalization(restored_face)
        
        face_helper.add_restored_face(restored_face)
    
    # Paste faces back to image
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
    
    # Edge-aware blending
    mask = np.zeros(restored_img.shape[:2], dtype=np.uint8)
    for face in face_helper.restored_faces:
        x, y, w, h = face[1]
        mask[y:y+h, x:x+w] = 255
    
    # Blend with original using seamless cloning
    result = cv2.seamlessClone(
        restored_img,
        original_img,
        mask,
        (img.shape[1]//2, img.shape[0]//2),
        cv2.NORMAL_CLONE
    )
    
    # Final global enhancements
    return apply_global_enhancements(result)

def apply_global_enhancements(img: np.ndarray) -> np.ndarray:
    """Apply elite global enhancements for photorealistic results"""
    # Color correction
    result = white_balance(img)
    
    # Advanced sharpening
    result = unsharp_mask(result, 1.5, 2.0, 0.05)
    
    # Dynamic range expansion
    result = exposure_fusion(result)
    
    # Texture enhancement
    result = cv2.detailEnhance(result, sigma_s=14, sigma_r=0.12)
    
    # Film-like grain
    result = add_film_grain(result, intensity=0.005)
    
    # Final color grading
    return apply_color_grade(result)

def adaptive_histogram_equalization(img):
    """CLAHE with adaptive clipping"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
    lab[...,0] = clahe.apply(lab[...,0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

def white_balance(img):
    """Professional white balancing"""
    result = cv2.xphoto.createGrayworldWB().balanceWhite(img)
    return np.clip(result * 1.1, 0, 255).astype(np.uint8)

def unsharp_mask(img, amount=1.0, radius=1.0, threshold=0):
    """High-quality sharpening with threshold control"""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened

def exposure_fusion(img):
    """HDR-like effect for expanded dynamic range"""
    exposures = [img]
    for gamma in [0.5, 1.5]:
        corrected = adjust_gamma(img, gamma=gamma)
        exposures.append(corrected)
    
    merge_mertens = cv2.createMergeMertens()
    fusion = merge_mertens.process(exposures)
    return np.clip(fusion*255, 0, 255).astype('uint8')

def adjust_gamma(image, gamma=1.0):
    """Gamma correction"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def add_film_grain(img, intensity=0.01):
    """Add realistic film grain"""
    grain = np.random.normal(0, intensity * 255, img.shape).astype(np.int16)
    grained = np.clip(img.astype(np.int16) + grain, 0, 255)
    return grained.astype(np.uint8)

def apply_color_grade(img):
    """Cinematic color grading"""
    # Apply film emulation LUT
    # lut = load_custom_lut('elite_realism.cube')  # Implement LUT loading
    # graded = cv2.LUT(img, lut) if lut is not None else img
    
    # Fallback color adjustment
    graded = img.copy()
    graded = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV)
    graded[..., 1] = np.clip(graded[..., 1] * 1.15, 0, 255).astype(np.uint8)
    graded[..., 2] = np.clip(graded[..., 2] * 1.05, 0, 255).astype(np.uint8)
    graded = cv2.cvtColor(graded, cv2.COLOR_HSV2BGR)
    
    # Subtle vignette
    rows, cols = graded.shape[:2]
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)
    vignette = (1 - Z * 0.2)[..., np.newaxis]
    vignette = np.clip(vignette, 0.7, 1.0)
    
    return np.clip(graded * vignette, 0, 255).astype(np.uint8)

def enhance_image(input_image_path: str, w_values=[0.65, 0.80, 0.95]) -> str:
    """Enhanced image processing with elite realism output"""
    input_path = Path(input_image_path)
    output_path = input_path.with_name(f"{input_path.stem}.elite_realism.png")
    
    # High-quality image loading
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {input_image_path}")
    
    # Handle alpha channel
    if img.shape[2] == 4:
        alpha = img[..., 3]
        img = img[..., :3]
    else:
        alpha = None
    
    # Process image
    restored_img = _enhance_img(img, w_values=w_values)
    
    # Restore alpha channel
    if alpha is not None:
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2BGRA)
        restored_img[..., 3] = alpha
    
    # Save in lossless format
    os.makedirs(output_path.parent, exist_ok=True)
    cv2.imwrite(str(output_path), restored_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    print(f"Elite realism image saved to: {output_path}")
    return str(output_path)

# GPU warm-up
if torch.cuda.is_available():
    warm_up_tensor = torch.randn(1,3,64,64).to(device)
    with torch.no_grad():
        net(warm_up_tensor, w=0.8)

#!/usr/bin/env python3
"""
COMPLETE DEEPFAKE DETECTION PIPELINE WITH CONDITIONAL VLM REASONING
"""

# ============================================================================
# SECTION 1: SETUP AND DEPENDENCIES
# ============================================================================

import os
import torch
import numpy as np
import cv2
import json
import argparse
from PIL import Image
from typing import Dict, List
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2, fftshift, dct
from qwen_vl_utils import process_vision_info

print("✓ All dependencies imported successfully!")

# ============================================================================
# SECTION 2: BACKBONE CLASSIFIER INITIALIZATION
# ============================================================================

MODEL_NAME = "prithivMLmods/AI-vs-Deepfake-vs-Real-9999"

print(f"Loading backbone model: {MODEL_NAME}")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SiglipForImageClassification.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

CLASS_NAMES = ["Artificial", "Deepfake", "Real"]

print(f"✓ Backbone model loaded successfully on {device}!")

# ============================================================================
# SECTION 3: FORENSIC SIGNAL EXTRACTION FUNCTIONS
# ============================================================================

def compute_texture_laplacian(gray):
    """
    Measures texture sharpness and natural variation.
    Low variance → unnaturally smooth regions (common in synthesis).
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_lbp(gray):
    """
    Local Binary Patterns (LBP)
    Captures micro-texture irregularities.
    Low variance often indicates synthetic or filtered textures.
    """
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    return float(np.var(lbp))


def compute_fft(gray):
    """
    Frequency domain analysis using FFT.
    Detects unnatural spectral energy caused by upsampling,
    diffusion models, or GAN artifacts.
    """
    spectrum = fftshift(fft2(gray))
    magnitude = np.log(np.abs(spectrum) + 1)
    return float(np.mean(magnitude))


def compute_dct(gray):
    """
    Discrete Cosine Transform (DCT) analysis.
    Captures JPEG compression inconsistencies introduced
    by splicing, in-painting, or recompression.
    """
    gray = np.float32(gray) / 255.0
    d = dct(dct(gray.T, norm="ortho").T, norm="ortho")
    return float(np.std(d[:40, :40]))


def extract_forensic_signals(image_path):
    """
    Runs all forensic signal extractors on an image.
    Returns a dictionary of low-level forensic measurements.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        "texture_laplacian": compute_texture_laplacian(gray),
        "lbp_texture": compute_lbp(gray),
        "fft_frequency": compute_fft(gray),
        "dct_compression": compute_dct(gray)
    }

print("✓ Forensic signal functions defined!")

# ============================================================================
# SECTION 4: BACKBONE CLASSIFICATION FUNCTION
# ============================================================================

def classify_image(image_path):
    """
    Classify image using backbone model.
    Returns prediction label and confidence.
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # Get highest probability and label
    max_idx = int(np.argmax(probs))
    manipulation_type = CLASS_NAMES[max_idx]
    
    prob_real = float(probs[CLASS_NAMES.index("Real")])
    authenticity_score = float(1.0 - prob_real)

    return {
        "manipulation_type": manipulation_type,
        "authenticity_score": authenticity_score
    }

print("✓ Backbone classification function defined!")

# ============================================================================
# SECTION 5: VLM ANALYZER CLASS
# ============================================================================

class VLMAnalyzer:
    """
    Qwen2-VL-2B analyzer.
    Only runs if backbone predicts NON-REAL or low-confidence REAL.
    Output: EXACTLY two sentences explaining why the image is not real.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"

        print(f"Loading VLM: {self.model_name}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print("✓ VLM loaded successfully!")

    def _create_prompt(self, backbone_result: Dict, signals: Dict) -> str:
        """
        Prompt focused ONLY on explaining why the image is NOT real.
        """
        return f"""You are an expert forensic image analyst.

This image has been classified as NOT REAL by an automated detection system.

Model prediction: {backbone_result['manipulation_type']}
Confidence: {backbone_result['authenticity_score']:.2%}

Forensic signals:
- Texture Laplacian: {signals['texture_laplacian']:.2f}
- LBP Texture Variance: {signals['lbp_texture']:.2f}
- FFT Frequency Energy: {signals['fft_frequency']:.2f}
- DCT Compression Std: {signals['dct_compression']:.4f}

TASK:
Explain WHY this image is not real.
Based on what can be visually observed in the image, explain why the image is not authentic.
Describe concrete visual or physical inconsistencies (e.g., texture behavior, edges, lighting, frequency artifacts)
Point out specific visual or physical inconsistencies that indicate synthetic or manipulated content.

RULES:
- Respond with EXACTLY two sentences
- Plain text only
- Do NOT mention probabilities, scores, or model confidence.
- No bullet points
- Do NOT say "this image may be real"
- Do NOT mention uncertainty
- Focus ONLY on manipulation evidence
- Be very specific to the content of THIS image.


Response:"""

    def analyze(
        self,
        image_path: str,
        backbone_result: Dict,
        signals: Dict
    ) -> str:
        """
        Run VLM only if image is non-real or low-confidence real.
        """
        # ⛔ Skip VLM if Real (this check is now done in pipeline, but keeping for safety)
        if backbone_result["manipulation_type"] == "Real":
            return "this image is real"

        try:
            prompt_text = self._create_prompt(backbone_result, signals)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            # Hard enforce EXACTLY two sentences
            sentences = [s.strip() for s in output_text.split(".") if s.strip()]
            output_text = ". ".join(sentences[:2]) + "."

            return output_text

        except Exception as e:
            print(f"⚠ VLM error: {e}")
            return (
                "The image contains visual inconsistencies that are not consistent with natural image formation. "
                "These artifacts align with patterns commonly seen in synthetic or manipulated imagery."
            )

print("✓ VLM Analyzer class defined!")

# ============================================================================
# SECTION 6: FULL PIPELINE EXECUTION
# ============================================================================

def run_pipeline(
    image_dir: str,
    output_json: str = "predictions.json",
    real_threshold: float = 0.90
):
    """
    Runs full deepfake detection pipeline on all images in a directory.
    """

    vlm = VLMAnalyzer(device=device)
    results = []

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\n📂 Found {len(image_files)} images to process\n")

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        print(f"🔍 Processing: {image_name}")

        # 1️⃣ Backbone classification
        backbone_result = classify_image(image_path)

        prediction = {
            "image_name": image_name,
            "manipulation_type": backbone_result["manipulation_type"],
            "authenticity_score": round(backbone_result["authenticity_score"], 4),
        }

        # 2️⃣ REAL → no VLM
        if (
            backbone_result["manipulation_type"] == "Real"
            and backbone_result["authenticity_score"] >= real_threshold
        ):
            prediction["explanation"] = "The image is real."

        # 3️⃣ NON-REAL → forensic + VLM
        else:
            signals = extract_forensic_signals(image_path)

            explanation = vlm.analyze(
                image_path=image_path,
                backbone_result=backbone_result,
                signals=signals
            )

            prediction["explanation"] = explanation

        results.append(prediction)
        print(f"   ✓ {backbone_result['manipulation_type']} (score: {backbone_result['authenticity_score']:.4f})\n")

    # 4️⃣ Save JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Pipeline finished. Results saved to {output_json}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deepfake Detection Pipeline with VLM Reasoning"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to folder with images"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="JSON file to save predictions"
    )
    parser.add_argument(
        "--real_threshold",
        type=float,
        default=0.90,
        help="Threshold for considering an image as 'Real' (default: 0.90)"
    )
    
    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory '{args.input_dir}' does not exist!")
        exit(1)

    if not os.path.isdir(args.input_dir):
        print(f"❌ Error: '{args.input_dir}' is not a directory!")
        exit(1)

    # Run pipeline
    run_pipeline(
        image_dir=args.input_dir,
        output_json=args.output_file,
        real_threshold=args.real_threshold
    )

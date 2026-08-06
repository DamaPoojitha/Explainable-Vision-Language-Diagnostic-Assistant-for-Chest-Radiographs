"""
End-to-end pipeline: X-ray image in -> marked-up image + readable report +
urgency tier out. This is the file the Streamlit app (and your final demo)
calls into.
"""

import argparse
import os

import torch
from PIL import Image

from dataset import eval_transform, CONDITIONS
from model import build_model
from gradcam import GradCAM, overlay_heatmap, region_description
from report_generator import generate_report, load_index
from triage import assess


def run_pipeline(
    image_path,
    checkpoint="models/chest_classifier.pt",
    report_index_path="models/report_index.pkl",
    threshold=0.5,
):
    ckpt = torch.load(checkpoint, map_location="cpu")
    conditions = ckpt.get("conditions", CONDITIONS)
    model = build_model(num_classes=len(conditions))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image = Image.open(image_path).convert("L")
    input_tensor = eval_transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor))[0]

    # Pick the highest-confidence non-"No Finding" label above threshold;
    # fall back to "No Finding" if nothing clears the bar.
    top_idx = int(torch.argmax(probs).item())
    top_condition = conditions[top_idx]
    top_confidence = probs[top_idx].item()

    if top_condition == "No Finding" or top_confidence < threshold:
        # Check if any abnormal condition still clears threshold
        for i, c in enumerate(conditions):
            if c != "No Finding" and probs[i].item() >= threshold:
                top_idx, top_condition, top_confidence = i, c, probs[i].item()
                break

    cam_engine = GradCAM(model, model.features.norm5)
    cam, _ = cam_engine.generate(input_tensor, top_idx)
    overlay = overlay_heatmap(image, cam)
    region = region_description(cam)

    triage_result = assess(top_condition, top_confidence)
    urgency_note = f"Triage: {triage_result['tier']} -- {triage_result['message']}"

    report_index = load_index(report_index_path)
    report_text = generate_report(
        top_condition, top_confidence, region,
        urgency_note=urgency_note, index=report_index,
    )

    return {
        "condition": top_condition,
        "confidence": top_confidence,
        "region": region,
        "overlay_image": overlay,
        "report": report_text,
        "triage": triage_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="models/chest_classifier.pt")
    args = parser.parse_args()

    result = run_pipeline(args.image, checkpoint=args.checkpoint)

    os.makedirs("outputs", exist_ok=True)
    result["overlay_image"].save("outputs/marked_xray.png")

    print(f"Condition: {result['condition']} ({result['confidence']:.2f})")
    print(f"Triage: {result['triage']['tier']}")
    print("\nReport:\n" + result["report"])
    print("\nMarked-up image saved to outputs/marked_xray.png")
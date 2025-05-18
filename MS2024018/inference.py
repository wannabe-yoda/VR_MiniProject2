import argparse
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from huggingface_hub import hf_hub_download

from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import get_peft_model, LoraConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True,
                        help="Absolute path to the directory containing images")
    parser.add_argument("--csv_path", required=True,
                        help="Absolute path to the CSV file with columns ['image_name','question','answer']")
    parser.add_argument("--output_csv", default="results.csv",
                        help="Absolute path to write the predictions CSV (default: ./results.csv)")
    args = parser.parse_args()

    # Resolve absolute paths (even if user provides relative by mistake)
    image_dir = os.path.abspath(args.image_dir)
    csv_path = os.path.abspath(args.csv_path)
    output_csv = os.path.abspath(args.output_csv)

    # 1. Load metadata
    df = pd.read_csv(csv_path)

    # 2. Prepare device, processor & base model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")

    # 3. Build LoRA-wrapped model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )
    model = get_peft_model(base, lora_config)

    # 4. Download & load adapter weights from HF hub
    repo_id = "wannabeyoda/BLIP_VQA_best"
    weight_file = hf_hub_download(repo_id=repo_id, filename="best_model.pt")
    adapter_state = torch.load(weight_file, map_location="cpu")
    model.load_state_dict(adapter_state, strict=False)

    model.to(device)
    model.eval()

    # 5. Inference
    answers = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if not row["image_name"] or str(row["image_name"]).startswith("."):
            answers.append("")
            continue

        img_path = os.path.join(image_dir, row["image_name"])
        question = str(row["question"])
        ans = ""
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    num_beams=3,
                    early_stopping=True
                )
            ans = processor.tokenizer.decode(
                gen_ids[0], skip_special_tokens=True
            ).strip().lower().split()[0]
        except Exception:
            pass
        answers.append(ans)

    # 6. Save results
    df["generated_answer"] = answers
    df.to_csv(output_csv, index=False)
    print(f"? Predictions written to {output_csv}")

if __name__ == "__main__":
    main()

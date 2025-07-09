#!/usr/bin/env python
"""
Score a custom attribute on RTP training data using zero-shot classification.
Creates a new dataset file compatible with fit.py for training custom classifiers.
"""

import os
import sys
import json
import argparse
from typing import List, Dict
from tqdm import tqdm

# Check if transformers is available
try:
    from transformers import pipeline
except ImportError:
    print("❌ transformers not installed. Run: pip install transformers[sentencepiece]")
    sys.exit(1)

def score_text_attribute(text: str, attribute: str, classifier) -> float:
    """
    Score text for a specific attribute using zero-shot classification.
    
    Args:
        text: Text to classify
        attribute: Target attribute (e.g., "politics", "sports", "emotion")
        classifier: HuggingFace zero-shot classification pipeline
    
    Returns:
        Score between 0-1 for the attribute
    """
    # Create binary classification: attribute vs not-attribute
    classes = [attribute, f"not {attribute}"]
    hypothesis_template = "This text is about {}"
    
    try:
        result = classifier(text, classes, hypothesis_template=hypothesis_template, multi_label=False)
        # Return score for the target attribute
        for label, score in zip(result['labels'], result['scores']):
            if label == attribute:
                return score
        return 0.0
    except Exception as e:
        print(f"Warning: Classification failed for text: {text[:50]}... Error: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Score custom attribute on RTP training data")
    parser.add_argument("--attribute", type=str, required=True, 
                       help="Target attribute to score (e.g., 'politics', 'sports', 'emotion')")
    parser.add_argument("--input_path", type=str, default="data/RTP_train.jsonl",
                       help="Input JSONL file path")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output JSONL file path (default: data/RTP_train_{attribute}.jsonl)")
    parser.add_argument("--model", type=str, default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                       help="Zero-shot classification model")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        args.output_path = f"data/RTP_train_{args.attribute.replace(' ', '_')}.jsonl"
    
    # Check input file exists
    if not os.path.exists(args.input_path):
        print(f"❌ Input file not found: {args.input_path}")
        print("💡 Download with: wget https://github.com/yidouweng/trace/releases/download/v1.0.0/RTP_train.jsonl.tar.gz -P data/ && cd data && tar -xzf RTP_train.jsonl.tar.gz")
        sys.exit(1)
    
    print(f"🔍 Scoring attribute: '{args.attribute}'")
    print(f"📄 Input: {args.input_path}")
    print(f"📄 Output: {args.output_path}")
    print(f"🤖 Model: {args.model}")
    
    # Initialize zero-shot classifier
    print("Loading zero-shot classification model...")
    try:
        classifier = pipeline("zero-shot-classification", model=args.model)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Process data
    print("Processing data...")
    processed_count = 0
    
    with open(args.input_path, 'r', encoding='utf-8') as infile, \
         open(args.output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(tqdm(infile, desc="Scoring"), 1):
            if args.max_samples and processed_count >= args.max_samples:
                break
                
            try:
                record = json.loads(line.strip())
                
                # Score prompt and continuation for the custom attribute
                prompt_text = record['prompt']['text']
                continuation_text = record['continuation']['text']
                
                prompt_score = score_text_attribute(prompt_text, args.attribute, classifier)
                continuation_score = score_text_attribute(continuation_text, args.attribute, classifier)
                
                # Create new record with custom attribute scores (same format as toxicity)
                new_record = {
                    "filename": record.get("filename", f"custom_{line_num}"),
                    "prompt": {
                        "text": prompt_text,
                        args.attribute: prompt_score  # Replace 'toxicity' with custom attribute
                    },
                    "continuation": {
                        "text": continuation_text,
                        args.attribute: continuation_score  # Replace 'toxicity' with custom attribute
                    }
                }
                
                outfile.write(json.dumps(new_record) + '\n')
                processed_count += 1
                
            except Exception as e:
                print(f"Warning: Skipping line {line_num} due to error: {e}")
                continue
    
    print(f"✅ Processed {processed_count} samples")
    print(f"📄 Output saved to: {args.output_path}")
    print()
    print("🔧 Next steps:")
    print(f"   1. Train classifier: python src/fit.py --data_path {args.output_path} --attribute {args.attribute}")
    print(f"   2. Use in generation: python src/generate.py --weights_path data/coefficients_{args.attribute}.csv")

if __name__ == "__main__":
    main()
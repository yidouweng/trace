# TRACE: HMM-Guided Text Generation

**Tractable Reasoning for Adaptable Controllable gEneration**

TRACE is a method for controllable text generation that uses Hidden Markov Models (HMMs) to guide language models away from unwanted attributes (like toxicity) while maintaining fluency and diversity.

## 🚀 Quick Start

### 1. Setup API Key

Get a [Google Perspective API key](https://developers.perspectiveapi.com/s/docs-get-started) and configure it:

**Recommended**: Edit `environment.yml` (or `environment_cpu.yml`), replace `'your_key_here'` with your actual key.

**Alternative**: Set environment variable (temporary):
```bash
export PERSPECTIVE_API_KEY="your_key_here"
```

### 2. Setup Environment

**Recommended**: Use `environment.yml` (auto-detects CUDA version):

```bash
conda env create -f environment.yml
conda activate trace
```

**If you encounter CUDA issues**: Use CPU-only environment:

```bash
conda env create -f environment_cpu.yml
conda activate trace
```

The GPU environment automatically detects and installs the correct PyTorch version for your CUDA installation (supports CUDA 11.8+ and 12.x).

### 3. Download Data and Models

Download all required data files at once:

```bash
# 1. Download pre-trained HMM model (~850MB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='gwenweng/hmm-gpt2-large', local_dir='models/hmm_gpt2-large_uncon_seq-len-32_4096_10M')
"

# 2. Download RTP training data (~6.6MB) - for custom classifier training
cd data/
wget https://github.com/yidouweng/trace/releases/download/v1.0.0/RTP_train.jsonl.tar.gz
tar -xzf RTP_train.jsonl.tar.gz

# 3. Download RTP test data (~6.6MB) - for large-scale evaluation  
wget https://github.com/yidouweng/trace/releases/download/v1.0.0/RTP_test.jsonl.tar.gz
tar -xzf RTP_test.jsonl.tar.gz
cd ..
```

**What you just downloaded:**
- **HMM Model**: Pre-trained Hidden Markov Model for toxicity control
- **RTP Train**: 100k prompts for training custom classifiers (optional)
- **RTP Test**: 10k prompts for large-scale evaluation (optional)
- **Demo prompts**: Already included in `data/prompts.jsonl` (12 examples)

### 4. Run Tutorial

🎯 **Start here**: Open and run **[tutorial.ipynb](tutorial.ipynb)** for a complete interactive walkthrough!

#### **Starting Jupyter Notebook**
```bash
# Make sure you're in the trace environment
conda activate trace

# Option A: Jupyter Lab (recommended)
jupyter lab

# Option B: Classic Jupyter Notebook  
jupyter notebook

```

#### **Important**: 
- **Always activate `trace` environment first** - the base environment lacks required packages
- **In your editor**: Select the `trace` environment as the Python interpreter for the notebook
- **Kernel issues**: If notebook shows wrong kernel, click the kernel selector (top right) and choose `trace`

The tutorial demonstrates:
- Environment setup and verification  
- Text generation with TRACE vs baseline comparison
- Toxicity, fluency, and diversity evaluation
- Analysis of where TRACE successfully reduces toxicity

## 📁 Repository Structure

```
trace/
├── tutorial.ipynb          # 🎯 START HERE - Interactive tutorial
├── src/
│   ├── generate.py         # Text generation script
│   ├── score.py            # Evaluation metrics  
│   ├── fit.py              # Train custom classifiers
│   ├── score_attribute.py  # Score custom attributes with zero-shot
│   └── ...                 # Core implementation
├── data/
│   ├── prompts.jsonl       # Demo prompts (12 examples)
│   ├── coefficients.csv    # Pre-trained toxicity classifier  
│   ├── RTP_train.jsonl     # Training data (100k prompts)
│   └── RTP_test.jsonl      # Test data (10k prompts)
├── models/                 # Pre-trained HMM model
├── environment.yml         # GPU environment
└── environment_cpu.yml     # CPU environment
```

## 🔬 Advanced Usage

### Custom Generation
```bash
python src/generate.py \
    --hmm_model_path models/hmm_gpt2-large_uncon_seq-len-32_4096_10M \
    --prompts_path data/prompts.jsonl \
    --a 1.0 --max_len 20 --num_generations 3
```

### Large-Scale Evaluation

Now that you have the RTP test dataset (10k prompts), you can run comprehensive evaluation:

```bash
# Generate text for all 10k test prompts
python src/generate.py --prompts_path data/RTP_test.jsonl

# Score the generated text for toxicity, fluency, and diversity
python src/score.py
```

This will take significantly longer than the 12-prompt demo, but provides robust statistical evaluation.

### Train Custom Classifiers

TRACE can control any attribute, not just toxicity! With the RTP training data (100k prompts) you downloaded, you can train classifiers for any attribute:

#### **Option 1: Score RTP Data for Custom Attribute**
```bash
# Example: Train a "politics" classifier
# 1. Score training data for your attribute (just provide keyword!)
python src/score_attribute.py --attribute politics

# 2. Train classifier  
python src/fit.py --data_path data/RTP_train_politics.jsonl --attribute politics

# 3. Use in generation
python src/generate.py --weights_path data/coefficients_politics.csv --a 1.0
```

**Other example attributes**: `sports`, `emotion`, `formality`, `sentiment`, `entertainment`

#### **Option 2: Use Your Own Dataset**
```bash
# Prepare your data in the same format as RTP_train.jsonl:
# {"prompt": {"text": "...", "your_attribute": 0.8}, "continuation": {"text": "...", "your_attribute": 0.2}}

python src/fit.py --data_path your_custom_data.jsonl --attribute your_attribute
```

## 🛠️ Troubleshooting

### **Environment Setup Issues**

**CUDA/PyTorch Import Errors**
```bash
# Error: "undefined symbol: cudaLaunchKernelExC" or PyTorch import fails
# This indicates CUDA version mismatch
```

**Solution**: The environment automatically detects and installs the correct CUDA version. If you encounter CUDA issues:

1. **Check your CUDA version**:
   ```bash
   nvidia-smi  # Look for "CUDA Version: X.X" in the output
   ```

2. **Recreate environment** (this will auto-detect your CUDA version):
   ```bash
   conda deactivate
   conda env remove -n trace -y
   conda env create -f environment.yml
   conda activate trace
   ```

3. **Test installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

4. **Still having issues?** Use CPU-only environment:
   ```bash
   conda env create -f environment_cpu.yml
   conda activate trace
   ```

**Environment Detection**
- **CUDA 11.8+**: Use `environment.yml` (recommended)
- **CUDA 12.x**: Use `environment.yml` (auto-detects)
- **No CUDA/CPU only**: Use `environment_cpu.yml`
- **Unsure?**: Try `environment.yml` first, fallback to `environment_cpu.yml`

### **Jupyter Notebook Issues**

**Can't open notebook in VS Code/Cursor?**
1. **Activate environment**: `conda activate trace`
2. **Start Jupyter server**: `jupyter lab --no-browser --port=8888`
3. **In editor**: Select `trace` Python interpreter
4. **Connect**: Point editor to `http://localhost:8888`

**Wrong kernel/environment in notebook?**
- Click the kernel selector (top-right of notebook)
- Choose `Python 3 (ipykernel)` from `trace` environment
- If not listed: `conda activate trace && python -m ipykernel install --user --name=trace`

**"Module not found" errors?**
- Check you're in `trace` environment: `echo $CONDA_DEFAULT_ENV`
- If showing `base`: `conda activate trace` then restart Jupyter

### **Other Issues**

**Having more issues?** Check our comprehensive **[FAQ.md](FAQ.md)** for solutions to:
- Environment setup problems
- Scoring issues (0.0/NA results)
- CUDA/memory errors  
- API key configuration
- Performance optimization

## 📈 Expected Results

With default settings, TRACE typically achieves:
- **70%+ toxicity reduction** vs baseline
- **Minimal fluency impact** (<10% perplexity change)
- **Maintained diversity** (>85% distinct-2)

## 📜 Citation

```bibtex
@inproceedings{yidou-weng2025trace,
  title={TRACE Back from the Future: A Probabilistic Reasoning Approach to Controllable Language Generation},
  author={Weng-Yidou, Gwen and Wang, Benjie and Van den Broeck, Guy},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License.
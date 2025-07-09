# TRACE - Frequently Asked Questions

## 🛠️ Installation & Setup

### Q: Which environment file should I use?

- **`environment.yml`**: For systems with NVIDIA GPUs and CUDA support (recommended)
- **`environment_cpu.yml`**: For systems without GPUs or CUDA (CPU-only)

### Q: Environment creation fails with package conflicts

Try these solutions in order:

1. **Update conda first**:
   ```bash
   conda update conda
   ```

2. **Use mamba for faster solving** (if available):
   ```bash
   mamba env create -f environment.yml
   ```

3. **Try the CPU environment** even if you have a GPU:
   ```bash
   conda env create -f environment_cpu.yml
   ```

### Q: How do I set up the Perspective API key?

1. **Get a key** from [Google Perspective API](https://developers.perspectiveapi.com/s/docs-get-started)

2. **Method 1 - Edit environment file (Recommended)**:
   - Edit `environment.yml` or `environment_cpu.yml`
   - Replace `"your_api_key_here"` with your actual key
   - Recreate the environment: `conda env create -f environment.yml --force`

3. **Method 2 - Set environment variable**:
   ```bash
   export PERSPECTIVE_API_KEY="your_actual_key_here"
   ```

4. **Method 3 - Set in notebook**:
   ```python
   import os
   os.environ['PERSPECTIVE_API_KEY'] = "your_actual_key_here"
   ```

## 🐛 Scoring Issues

### Q: All toxicity scores are 0.0 and fluency scores are "NA"

**Most common cause**: Wrong conda environment

**Solution**:
```bash
# Check current environment
echo $CONDA_DEFAULT_ENV

# If not 'trace', activate it:
conda activate trace

# Restart Jupyter if using notebooks:
jupyter lab
```

**Other causes to check**:
- Missing PyTorch: `python -c "import torch; print('✅ PyTorch works')"`
- Missing Transformers: `python -c "import transformers; print('✅ Transformers works')"`
- Invalid API key: Check your Perspective API key is valid

### Q: "ModuleNotFoundError: No module named 'torch'"

You're running in the wrong conda environment or PyTorch isn't installed.

**Solution**:
```bash
conda activate trace
python -c "import torch"  # Test if PyTorch is available
```

If PyTorch is missing, recreate the environment:
```bash
conda env remove -n trace
conda env create -f environment.yml
```

### Q: API key errors or rate limiting

**Invalid key**: 
- Verify your key at the [Perspective API console](https://console.cloud.google.com/apis/credentials)
- Make sure the API is enabled for your project

**Rate limiting**:
- The free tier has rate limits
- Reduce batch size: `python src/score.py --batch_size 1`
- Add delays between requests (implemented automatically)

## 🚀 Performance Issues

### Q: CUDA out of memory errors

**Reduce batch sizes**:
```bash
python src/score.py --batch_size 1
python src/generate.py --generation_batch_size 1 --prompt_batch_size 1
```

**Use CPU-only mode**:
```bash
python src/score.py --device cpu
```

### Q: Scoring is very slow

**Expected behavior**: 
- Fluency scoring loads GPT2-XL (1.5B parameters) 
- Toxicity scoring makes API calls
- ~2-3 seconds per prompt is normal

**Speed up options**:
- Increase batch size: `--batch_size 10` (if memory allows)
- Use GPU: Make sure CUDA is working

## 📁 File Issues

### Q: "FileNotFoundError" for HMM model or data files

**Missing HMM model**:
```bash
# Download from HuggingFace Hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='gwenweng/hmm-gpt2-large', local_dir='models/hmm_gpt2-large_uncon_seq-len-32_4096_10M')
"
```

**Missing data files**:
- `data/prompts.jsonl` should be included in the repository
- `data/coefficients.csv` should be included in the repository
- For full evaluation, download `RTP_test.jsonl` separately

### Q: Permission denied errors

**On shared systems**:
```bash
# Use --user flag for pip installs
pip install --user package_name

# Or create environment in your home directory
conda env create -f environment.yml --prefix ~/envs/trace
conda activate ~/envs/trace
```

## 🧪 Generation Issues

### Q: Generation produces empty or weird outputs

**Check inputs**:
- Verify `data/prompts.jsonl` format: `{"prompt": {"text": "your prompt here"}}`
- Make sure prompts aren't too long (>512 tokens)

**Check parameters**:
- Default `--a 1.0` is usually good
- Try `--a 0.5` for less guidance or `--a 2.0` for more
- Increase `--max_len` if outputs are too short

### Q: "RuntimeError" during generation

**Memory issues**: Reduce batch sizes
**Model loading issues**: Check HMM model path exists
**CUDA issues**: Try `--device cpu`

## 🔧 Advanced Troubleshooting

### Q: Import errors with MKL threading

This is handled automatically by the environment files, but if you see MKL errors:

```bash
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
```

### Q: How to use different base models?

**For generation**:
```bash
python src/generate.py --model_path gpt2-medium  # Use different base model
```

**For scoring**:
```bash
python src/score.py --perp_model gpt2-large  # Use different fluency model
```

### Q: How to control attributes other than toxicity?

**Step 1: Score training data for your attribute**:
```bash
python src/score_attribute.py --attribute politics
```

**Step 2: Train classifier**:
```bash
python src/fit.py --data_path data/RTP_train_politics.jsonl --attribute politics
```

**Step 3: Use in generation**:
```bash
python src/generate.py --weights_path data/coefficients_politics.csv
```

**Common attributes**: `politics`, `sports`, `emotion`, `formality`, `sentiment`, `entertainment`

### Q: Getting different results than expected?

**Reproducibility**:
- Set seed: `--seed 42`
- Use same model versions
- Check hardware differences (GPU vs CPU can give slightly different results)

**Parameter sensitivity**:
- TRACE is sensitive to `--a` parameter
- HMM quality affects results
- Toxicity classifier affects guidance

## 📊 Tutorial Notebook Issues

### Q: Jupyter notebook kernel crashes

**Memory issues**: 
- Restart kernel and clear outputs
- Close other notebooks
- Use smaller models or batch sizes

**Environment issues**:
- Make sure Jupyter is running in the 'trace' environment
- Install jupyter in the environment: `conda install jupyterlab`

### Q: Plotting/visualization errors

**Missing matplotlib**:
```bash
conda activate trace
conda install matplotlib seaborn
```

**Display issues**:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

## 🆘 Getting More Help

If these solutions don't work:

1. **Check the GitHub Issues**: [github.com/yidouweng/trace/issues](https://github.com/yidouweng/trace/issues)
2. **Create a new issue** with:
   - Your operating system
   - Python/conda versions
   - Full error message
   - Steps to reproduce
3. **Include environment info**:
   ```bash
   conda list > environment_info.txt
   python --version
   nvidia-smi  # If using GPU
   ```

## 📚 Additional Resources

- **Paper**: TRACE Back from the Future: A Probabilistic Reasoning Approach to Controllable Language Generation (ICML 2025)
- **HuggingFace Model**: [gwenweng/hmm-gpt2-large](https://huggingface.co/gwenweng/hmm-gpt2-large)
- **GitHub Repository**: [yidouweng/trace](https://github.com/yidouweng/trace)
- **Perspective API Docs**: [developers.perspectiveapi.com](https://developers.perspectiveapi.com/)
- **PyTorch Installation**: [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
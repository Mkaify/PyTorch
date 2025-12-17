# Security Guidelines for PyTorch Repository

This document provides security best practices for contributing to and maintaining this PyTorch repository.

## 🔐 API Keys and Secrets Management

### Never Commit Secrets
**NEVER** commit the following to the repository:
- API keys (OpenAI, Anthropic, Hugging Face, etc.)
- Authentication tokens
- Passwords
- Private keys or certificates
- Database credentials
- Service account files
- Any other sensitive information

### How to Handle Secrets Properly

#### 1. Use Environment Variables
```python
import os

# ✅ GOOD - Load from environment variable
api_key = os.getenv('OPENAI_API_KEY')
hf_token = os.getenv('HUGGINGFACE_TOKEN')

# ❌ BAD - Hardcoded secret
api_key = "sk-1234567890abcdefghijklmnop"  # NEVER DO THIS!
```

#### 2. Use .env Files (Already in .gitignore)
Create a `.env` file in your local repository (it's ignored by git):
```bash
# .env file (local only, not committed)
OPENAI_API_KEY=sk-your-actual-key-here
HUGGINGFACE_TOKEN=hf_your-token-here
WANDB_API_KEY=your-wandb-key-here
```

Load it in your Python code:
```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file
api_key = os.getenv('OPENAI_API_KEY')
```

#### 3. Use Configuration Templates
Instead of committing config files with secrets, commit templates:
```yaml
# config.template.yml (committed)
api:
  openai_key: YOUR_OPENAI_KEY_HERE
  huggingface_token: YOUR_HF_TOKEN_HERE
```

Users copy this to `config.yml` (which is in .gitignore) and add their real keys.

## 🔍 What to Check Before Committing

### Pre-Commit Checklist
- [ ] No API keys in code files
- [ ] No hardcoded passwords
- [ ] No authentication tokens
- [ ] `.env` files are not staged
- [ ] Large model files are not committed (use Git LFS or download at runtime)
- [ ] No personal data or sensitive information

### Files to Review Carefully
- Python files that interact with external APIs
- Configuration files (YAML, JSON, INI)
- Jupyter notebooks
- Requirements files
- Documentation with examples

## 🛡️ Using External APIs Safely

### Hugging Face
```python
# ✅ GOOD - Public models (no token needed for inference)
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")

# ✅ GOOD - Using token from environment for private models
from huggingface_hub import login
import os
login(token=os.getenv('HUGGINGFACE_TOKEN'))
```

### OpenAI
```python
# ✅ GOOD
import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ❌ BAD
from openai import OpenAI
client = OpenAI(api_key="sk-proj-...")  # NEVER!
```

### Weights & Biases (wandb)
```python
# ✅ GOOD
import os
import wandb
wandb.login(key=os.getenv('WANDB_API_KEY'))

# ❌ BAD
wandb.login(key="abc123...")  # NEVER!
```

## 📋 Common Secret Patterns to Avoid

### OpenAI
- `sk-...` (API keys)
- `sk-proj-...` (Project API keys)

### GitHub
- `ghp_...` (Personal access tokens)
- `gho_...` (OAuth tokens)
- `github_pat_...` (Fine-grained tokens)

### AWS
- `AKIA...` (Access key IDs)
- Secret access keys (40 alphanumeric characters)

### Google Cloud
- `AIza...` (API keys)
- Service account JSON files

### Hugging Face
- `hf_...` (Access tokens)

## 🚨 What to Do If You Accidentally Commit a Secret

### If You Haven't Pushed Yet
1. Remove the secret from your code
2. Use `git reset` to undo the commit
3. Add the file to `.gitignore` if needed
4. Commit the corrected version

### If You've Already Pushed
1. **IMMEDIATELY REVOKE THE KEY/TOKEN** at the service provider
2. Remove the secret from your code
3. Consider the key compromised - never reuse it
4. Generate a new key/token
5. Commit the fix
6. Optionally: Rewrite git history (use with caution)

### Services to Revoke Keys
- OpenAI: https://platform.openai.com/api-keys
- GitHub: https://github.com/settings/tokens
- Hugging Face: https://huggingface.co/settings/tokens
- Weights & Biases: https://wandb.ai/authorize

## 🧰 Tools and Resources

### Recommended Tools
- **python-dotenv**: Load environment variables from `.env` files
- **git-secrets**: Prevent committing secrets (AWS tool)
- **detect-secrets**: Scan repositories for secrets
- **GitHub Secret Scanning**: Automatic detection (enable in repository settings)

### Installation
```bash
pip install python-dotenv
```

### GitHub Security Features
Enable these in repository settings:
- Secret scanning alerts
- Push protection
- Dependabot security updates
- Code scanning (CodeQL)

## 📚 Additional Resources

- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [OWASP Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Python Dotenv Documentation](https://github.com/theskumar/python-dotenv)

## ✅ Quick Reference

### Good Practices
```python
# ✅ Environment variables
api_key = os.getenv('API_KEY')

# ✅ Config files in .gitignore
with open('config.yml') as f:  # config.yml is in .gitignore
    config = yaml.safe_load(f)

# ✅ Public models (no authentication needed)
model = AutoModel.from_pretrained("bert-base-uncased")
```

### Bad Practices
```python
# ❌ Hardcoded secrets
api_key = "sk-1234567890"

# ❌ Secrets in comments
# My API key is: sk-proj-abcdefg

# ❌ Secrets in print statements
print(f"Using API key: {api_key}")  # May appear in logs
```

---

**Remember:** When in doubt, don't commit it! Use environment variables and the `.env` file instead.

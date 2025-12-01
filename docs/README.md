# ⚙️ Set up and initialization (wit pip install)
1. **Create a virtual environment**
```bash
   python -m venv myenv
```

2. **Activate the virtual environment**

   - On macOS/Linux:
```bash
     source myenv/bin/activate
```
   
   - On Windows:
```bash
     myenv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install mkdocs-material
```

### Running Locally

Start the development server:
```bash
mkdocs serve
```

The site will be available at `http://127.0.0.1:8000/`


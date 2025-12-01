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

The site will be available at `http://127.0.0.1:8000/` and the website that is deployed: https://yourusername.github.io/repository-name/


# ⚙️ How to deploy the website

1. Go to your repository -> Settings -> Pages
2. Ensure the page set to deploy from the gh-pages branch.
3. Your site will be available at: https://yourusername.github.io/repository-name/


# ⚙️ How to update the website

1. Change contents of .md files you would like to change
2. In the directory LLMRouter/docs, run command
```
mkdocs serve
```
3. Then you can see the updated contents at: https://yourusername.github.io/repository-name/

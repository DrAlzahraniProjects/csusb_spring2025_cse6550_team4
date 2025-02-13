## Prerequisites

Before you begin, ensure you have the following:

1. **Git**: [Install Git](https://git-scm.com/) from its official website.
2. **Docker**: [Install Docker](https://www.docker.com) from its official website.
3. **Linux/MacOS**: No extra setup needed.
4. **Windows**: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker's WSL integration by following [this guide](https://docs.docker.com/desktop/windows/wsl/).

---

### Step 1: Clone the Repository

Clone the GitHub repository to your local machine:

```bash
git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team4
```

### Step 2: Navigate to the Repository

Change to the cloned repository directory:

```bash
cd csusb_spring2025_cse6550_team4
```

### Step 3: Pull the Latest Version

Update the repository to the latest version:

```bash
git pull origin main

```

### Step 4: Build the Docker Container

Run the setup script to build and start the Docker container:

```bash
docker build -t csusb_spring2025_cse6550_team4 .
```

### Step 5: Run the Docker Container

You can run a Chatbot in either Streamlit or Jupyter

For Streamlit:

```bash
docker run -p 8501:8501 csusb_spring2025_cse6550_team4

```

For Jupyter:

```bash
docker run -p 8888:8888 csusb_spring2025_cse6550_team4 jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

```

### Step 6: Access the Chatbot

For Streamlit:

- Once the container starts, Open browser at http://localhost:8501.

For Jupyter:

- Once the container starts, the terminal will display a URL (e.g., `http://127.0.0.1`) with a token.
- Copy and paste this URL into your browser to access the Jupyter Notebook interface.

### Step 7: Run the program

For Streamlit:

- Chatbot is automatically displayed.

Open browser at http://localhost:8501

For Jupyter:

1. In Jupyter, navigate to `notebook.ipynb`.
2. Open the notebook.
3. Select **Run All Cells** to execute the code.

---

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
git clone --branch docker --single-branch https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team4.git
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

### Step 4: Set Build Script

Run the setup script to build and start the Docker container:

```bash
chmod +x docker-launch.sh
```

### Step 5: Run Build Script

```bash
./docker-launch.sh
```

### Step 6: Access the Chatbot

For Streamlit:

- Once the container starts, Open browser at http://localhost:2504.

For Jupyter:

- http://localhost:2514/

### Step 7: Run the program

For Streamlit:

- Chatbot is automatically displayed.

For Jupyter:

1. In Jupyter, navigate to `notebook.ipynb`.
2. Open the notebook.
3. Select **Run All Cells** to execute the code.

---

### Hosted on CSE department web server

For Streamlit:

Open browser at https://sec.cse.csusb.edu/team14

For Jupyter:

Open browser at https://sec.cse.csusb.edu/team14/jupyter

## Prerequisites

Before you begin, ensure you have the following:

1. **Git**: [Install Git](https://git-scm.com/) from its official website.
2. **Docker**: [Install Docker](https://www.docker.com) from its official website.
3. **Linux/MacOS**: No extra setup needed.
4. **Windows**: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker's WSL integration by following [this guide](https://docs.docker.com/desktop/windows/wsl/).

---

### Step 1: Remove the existing code directory completely

Because the Git repository can't been updated correctly, need to remove the directory first.

```bash
rm -rf csusb_spring2025_cse6550_team4 csusb_spring2025_cse6550_team4-new
```

### Step 2: Clone the Repository

Clone the GitHub repository to your local machine:

```bash
git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team4.git
```

### Step 3: Navigate to the Repository

Change to the cloned repository directory:

```bash
cd csusb_spring2025_cse6550_team4
```

### Step 4: Pull the Latest Version

Update the repository to the latest version:

```bash
git pull origin main
```

### Step 5: Set Build Script

Run the setup script to build and start the Docker container:

```bash
chmod +x docker-launch.sh
```

### Step 6: Run Build Script (enter your Groq API Key when prompted):

```bash
./docker-launch.sh
```

### Step 7: Access the Chatbot

For Streamlit:

- Once the container starts, Open browser at http://localhost:2504/team4s25.

### Step 8: Enable execute permissions for the Docker cleanup script:

```bash
chmod +x docker-cleanup.sh
```

### Step 9: Run the script to stop and remove the Docker image and container :

```bash
./docker-cleanup.sh
```

---

### Hosted on CSE department web server

For Streamlit:

Open browser at  https://sec.cse.csusb.edu/team4s25


## Google Colab Notebook  

We have integrated a Google Colab notebook for easy access and execution.

[Open in Colab](https://colab.research.google.com/drive/1c9lR1g3JFY5eTjBAA-KURuW-_DH83ijp?usp=sharing#scrollTo=WdYysvnzW4Do)  


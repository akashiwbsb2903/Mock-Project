# Mock-Project

## Project Overview

Protein Scope is a comprehensive nutrition insights platform that leverages data from various sources (blogs, journals, Reddit, etc.) to provide analytics, trends, and chatbot-based insights. The project is modular, with scrapers, data processing, and a web application for interactive exploration.

## Project Structure

```
Mock-Project-main 15 copy/
├── fix_combined_json.py
├── test_scope.py
├── test_scope_multi.py
├── nutrition_insights/
│   ├── __init__.py
│   ├── llm_connection.py
│   ├── merge_scrapper.py
│   ├── README.md
│   ├── requirements.txt
│   ├── run_all.py
│   ├── data/
│   ├── phase3/
│   │   ├── app.py
│   │   ├── run_app.py
│   │   ├── requirements.txt
│   │   ├── assets/
│   │   ├── components/
│   │   ├── config/
│   │   ├── services/
│   │   └── utils/
│   ├── rag/
│   ├── scrappers/
│   └── scripts/
└── ...
```

- **nutrition_insights/**: Main package with all core modules and data.
  - **data/**: Contains datasets and processed files.
  - **phase3/**: Web application (Streamlit or similar) and its components.
  - **rag/**: Retrieval-Augmented Generation (RAG) utilities.
  - **scrappers/**: Scripts to scrape data from blogs, journals, and Reddit.
  - **scripts/**: Data processing scripts.

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/akashiwbsb2903/Mock-Project.git
cd Mock-Project-main\ 15\ copy
```

### 2. Set Up Python Environment
It is recommended to use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required packages:
```bash
pip install -r nutrition_insights/requirements.txt
pip install -r nutrition_insights/phase3/requirements.txt
```

### 4. Run the Application

To start the web app (using Streamlit):
```bash
cd nutrition_insights/phase3
streamlit run app.py
```

Or run all scripts as needed from the root or respective folders.



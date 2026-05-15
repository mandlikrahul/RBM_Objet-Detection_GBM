# 1. PROJECT TITLE
    Hybrid multi-head transformer enabled Attention based Semantic CNN-GBM for object detection  

# 2. HARDWARE REQUIREMENTS
    OS-Windows 11
    RAM-16GB
    ROM-More than 100 GB
    CPU-1.7 Ghz

# 3. SOFTWARE REQUIREMENTS
    Software name: Python : Version: 3.9.13 (Download link:
    https://www.python.org/downloads/windows/)
    Software name: pycharm : Version: 2025.2.3 (Download link: Pycharm 2025.2.3-Community
    Edition (https://www.jetbrains.com/pycharm/download/#section=windows)



# 4. Dataset Links :

    1. Coco dataset : https://cocodataset.org/#home 
    2. Open Image V7 : https://storage.googleapis.com/openimages/web/download_v7.html 
    

# 5. Installation:
```bash
     1. Clone the repository
    git clone https://github.com/<your-username>/RBM_Objet-Detection_GBM.git
    cd RBM_Objet-Detection_GBM

     2. Create a virtual environment (recommended)
    python -m venv venv
    venv\Scripts\activate        # Windows
    # source venv/bin/activate   # Linux/macOS

    3. Install dependencies
    pip install -r requirements.txt
```
# 6. Usage:

     Run Full Pipeline (Training + Evaluation)

    ```bash
    python main.py
    ```

You will be prompted: **"Do you need Complete Execution?"**
    - **Yes** — runs data loading, training, evaluation, and plotting for both datasets
    - **No** — skips training and only plots previously saved results

    ### Run GUI Demo

    ```bash
    python GUI.py
    ```

    Steps in the GUI:
    1. Click Load DB1 (COCO) or **Load DB2** (Open Images) to load a pre-trained model
    2. Click Load Image to select a test image from the Dataset folder
    3. Click Preprocessing to apply CLAHE-based enhancement
    4. Click Model Prediction to run inference and display detected bounding boxes



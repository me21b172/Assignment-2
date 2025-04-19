# **CNN Training and Fine-tuning Assignment**

**Name:** Anuj Jagannath Said

**Roll Number:** ME21B172

**Assignment:** Learn how to use CNNs - train from scratch and fine-tune a pre-trained model

## **Link for wandb Report**
Following is the link for wandb report
```
  https://api.wandb.ai/links/me21b172-indian-institute-of-technology-madras/borhi11r
```

## **Task Description**

This assignment consists of two main tasks:

1. **Task A**: Implement and train a CNN model from scratch for image classification on the iNaturalist dataset
2. **Task B**: Fine-tune a pre-trained ResNet50 model for image classification

## **Setup Instructions**

### **Prerequisites**

- Python 3.6+
- CUDA-enabled GPU (recommended)
- WandB account (for hyperparameter tuning visualization)

### **Installation**

1. Clone the repository:
    
   
    
    ```
    git clone <repository-url>
    cd <repository-folder>
    ```
    
2. Create and activate a virtual environment:
    
    
    
    ```
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate    # Windows
    ```
    
3. Install dependencies:
    
    
    
    ```
    pip install -r requirements.txt
    ```
    
4. Set up WandB (for Task A):CopyDownload
    
    
    
    ```
    wandb login
    ```
    
5. Prepare the dataset:
    - Download the iNaturalist dataset
    - Organize it in the following structure:
        
        ```
        inaturalist_12K/
        ├── train/
        │   ├── class1/
        │   ├── class2/
        │   └── ...
        └── val/
            ├── class1/
            ├── class2/
            └── ...
        ```
        
    - Update the **`data_dir`** path in **`config.txt`** to point to your dataset location

## **How to Run**

### **Task A: Train CNN from Scratch**

**File:** **`wandb_runner.py`**

This script handles both hyperparameter tuning and final model training.

### **Options:**

1. **Hyperparameter Tuning** (15 trials):
    

    
    ```
    python wandb_runner.py --tune True
    ```
    
2. **Train Best Model** (25 epochs with optimized parameters):
    
    
    
    ```
    python wandb_runner.py --tune True
    ```
    
3. **Standard Training** (without tuning):
    
    
    
    ```
    python wandb_runner.py --epochs 10
    ```
    

### **Arguments:**

- **`-tune`**: Boolean flag to enable hyperparameter tuning (default: False)
- **`-epochs`**: Number of training epochs (default: 5, used only when not tuning)

### **Task B: Fine-tune Pre-trained Model**

**File:** **`model.py`**

This script fine-tunes a pre-trained ResNet50 model.

### **Run Command:**


```
python model.py --epochs 7 --batch_size 64
```

### **Arguments:**

- **`-batch_size`**: Batch size for training (default: 64)
- **`-epochs`**: Number of training epochs (default: 7)
- **`-wandb`**: Boolean flag to enable WandB logging (default: False)

## **Configuration**

The **`config.txt`** file contains:

- **`num_classes`**: Number of output classes (set to match your dataset)
- **`data_dir`**: Path to your dataset directory

Example **`config.txt`**:

```
{
    'num_classes': '10',
    'data_dir': './inaturalist_12K'
}
```

## **Additional Notes**

1. For Task A, the model architecture parameters can be modified in **`wandb_runner.py`**
2. WandB integration provides visualization of:
    - Training/validation metrics
    - Hyperparameter tuning results
    - Sample predictions
3. GPU is strongly recommended for faster training
4. The code automatically handles device selection (CPU/GPU)

## **Results**

After training:

- Model performance metrics will be printed in console
- For WandB-enabled runs, results will be available on your WandB dashboard
- Sample predictions will be logged for visualization

Also mention the space for wandb report where I can paste my wandb report link.
give me entire above text in markdown format

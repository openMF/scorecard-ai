
# Scorecard-AI 🧠📊

[![Hosted on HuggingFace](https://img.shields.io/badge/Hosted%20On-HuggingFace-blue)](https://parthkl-scorecard.hf.space/)
  
**Scorecard-AI** is a powerful tool designed for predicting **Interest Repaid Derived** and **Nominal Interest Rates** using machine learning models. This tool allows users to interactively select features, input their values, and choose the model to make predictions. If some feature values are not available, the tool replaces them with the average values or allows users to choose from predefined sets of values.

## Features 🚀

- **User-friendly interface**: Select features, input values, and choose models for prediction.
- **Model flexibility**: Multiple model options to choose from, ensuring high customization.
- **Smart defaults**: Missing feature values are replaced with average values automatically, saving time for users.
- **Predefined values**: Users can also select from preset feature values.
  
## Project Structure 🗂️

```
Web/
│
├── .gitattributes
├── README.md
├── dataset_link
├── interest_repaid_derived.py          # Original code for Interest Repaid Derived prediction
├── intrest_repaid_frontend.ipynb       # Modified Jupyter Notebook for the frontend
├── nominal_frontend.ipynb              # Modified Jupyter Notebook for Nominal Interest Rate frontend
└── nominal_interest_rate.py            # Original code for Nominal Interest Rate prediction
```

### Web Folder 🌐

The `Web` folder contains the **Django** application used to host the Scorecard-AI tool on **HuggingFace Docker Spaces**. The frontend is built using **HTML** and **CSS**, providing a seamless user experience for interacting with the machine learning models.

Check out the live demo here: [Scorecard-AI Live](https://parthkl-scorecard.hf.space/) ✨

### Notebooks and Scripts 📒

- **`interest_repaid_derived.py`**: Original Python script for predicting Interest Repaid Derived.
- **`nominal_interest_rate.py`**: Original Python script for predicting Nominal Interest Rate.
- **`intrest_repaid_frontend.ipynb`** & **`nominal_frontend.ipynb`**: These Jupyter notebooks are modified versions of the original Python scripts, adapted for the frontend application.

## Usage 🛠️

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/scorecard-ai.git
    ```
    
2. **Install Dependencies**:
    Install the required Python packages by running:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Locally**:
    You can run the Django application locally by navigating to the `Web` folder and running the following:
    ```bash
    python manage.py runserver
    ```

4. **Predict Interest Rates**:
    - Select features via the frontend.
    - Choose the model you'd like to use for prediction.
    - Input values or allow the system to use average values where data is missing.
  
## How it Works 🧠

1. **Feature Selection**: Users are prompted to select the relevant features from the dataset for the prediction task.
2. **Input Values**: Users input the values for the selected features. If any values are missing, the system automatically fills them with the average values from the dataset.
3. **Model Selection**: The user selects the model they want to use for prediction. Each model is pre-trained and fine-tuned for either **Interest Repaid Derived** or **Nominal Interest Rate** prediction.
4. **Results**: The predicted value is displayed in the frontend after processing.

## Dependencies 📦

- **Django**: Backend web framework.
- **Jupyter Notebook**: Used for model development and testing.
- **Python**: The primary language for the application.
- **HuggingFace Docker Spaces**: The application is hosted using HuggingFace's Docker spaces.

## Future Enhancements 🌱

- Expand the range of models available for prediction.
- Add more pre-defined feature values for different user scenarios.
- Improve UI/UX for an even smoother experience.
  
## Contributing 🤝

Contributions are welcome! Feel free to open an issue or submit a pull request. Be sure to follow the contribution guidelines.

---

### References 🔗
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/spaces)
- [Django Official Documentation](https://docs.djangoproject.com/)
- [Python Official Documentation](https://www.python.org/doc/)

---

Made with ❤️ by Parth Kaushal

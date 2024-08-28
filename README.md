# Data Analysis Project Example

This project shows how to use the Python programming language and its associated machine learning libraries for data analysis, including data loading, preprocessing, analysis visualization, and model building.

## Installation

Ensure you have Python 3 and pip installed. Then, run the following command to install the necessary dependencies:

```sh
pip install scikit-learn==1.2.2 pandas==2.1.1 numpy==1.26.4 matplotlib==3.8.4 jupyter==1.0.0
```

## Project Description

This project aims to analyze a dataset to extract valuable insights and visualize the results. The dataset contains multiple attributes, which we will explore and analyze using the following libraries and versions:
- Scikit-learn（version 1.2.2）
- Pandas（version 2.1.1）
- NumPy（version 1.26.4）
- Matplotlib（version 3.8.4）
- Jupyter Notebook（version 1.0.0）

## Data

The dataset used is `Appendix_A_Data.xlsx`, which includes the following columns:
- $COD_{(Water Inlet)}$: Description of Column1
- $NH_{4}^{+}-N_{(water inlet)}$: Description of Column2
- $COD_{(Effluent)}$: Description of Column3
- $NH_{4}^{+}-N_{(Effluent)}$`: Description of Column4
- $C$: Description of Column5
- $pH$: Description of Column6
- $DO$: Description of Column7
- $T$: Description of Column8
- $ORP$: Description of Column9
- $MLSS$: Description of Column10
- $MLVSS$: Description of Column11
- $V_{30}$: Description of Column12
- $RI_{(Solution)}$: Description of Column13
- $RI_{(globule)}$: Description of Column14
- $Reflux ratio$: Description of Column15
- $Flocculation effect$: Description of Column16

## File Structure

- `Appendix_B.ipynb`: Jupyter Notebook containing the data analysis code.
- `Appendix_A_Data.xlsx`: The dataset used for analysis.
- `README.md`: This file.

## Usage

1.  Clone the repository:
    ```sh
    git clone https://github.com/yourusername/data-analysis-project.git
    cd data-analysis-project
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Jupyter Notebook:
    ```sh
    jupyter notebook data_analysis.ipynb
    ```

## Code Example

Here's a brief example of the analysis code:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Pulling up data
import pandas as pd
df = pd.read_excel('Appendix_A_Data.xlsx')
df

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# Create a dataset with text features
#Build a dataset containing textual features
data = df.iloc[:,15]

# Converting Series Objects to NumPy Arrays
X = data.to_numpy()
# Converting data to a string type
X = X.astype(str)
# Converting data to a two-dimensional array
X = X.reshape(-1, 1)
# Creating a OneHotEncoder object
encoder = OneHotEncoder()
# Create a OneHotEncoder object
X_encoded = encoder.fit_transform(X).toarray()
# Create a DataFrame to store the uniquely hot encoded data
encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['feature']))
# Printing encoded data
print(X_encoded)

# Split the dataset
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.3)

# Standardized features
scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Define the model
models = {
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Neural Network": MLPRegressor(random_state=42, max_iter=1000),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(random_state=42)
}

# Train the model and make predictions
predictions = {}
for name, model in models.items():
    model.fit(Xtrain, Ytrain)
    predictions[name] = model.predict(Xtest)
```
## Visualization of the follow-up

## Results
The analysis revealed the following key insights:

Early warning classification model
- $Normal conditions$
- $Temperature abnormality$
- $Sludge concentration abnormality$
- $Heavy metal concentration abnormality$
- $Salinity abnormality$
- $pH abnormality$
- $Dissolved oxygen abnormality$

Predictive regression model
- $COD_{(Effluent)}$
- $NH_{4}^{+}-N_{(Effluent)}$
## Contribution

If you wish to contribute to this project, please fork the repository and submit a pull request. For significant changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT license.
```

### Detailed explanation of the steps

**Project Overview**:
   - Provide a short description of the project's background and objectives, so that the reader quickly understands the purpose and importance of the project.

**Installation Instructions**:
   - List all the dependencies required by the project and provide specific commands to install those dependencies to ensure that users can install and run the project smoothly.

**Project Description**:
   - Describe the goals of the project and the libraries used and their versions to help users understand the technical background of the project.

**Data Description**:
   - Describe the dataset in detail, including the source of the dataset and the meaning of each column, to help users understand the structure and content of the data.

**Project File Structure**:
   - Make a detailed list of the main files and folders of the project, explain the purpose and function of each file and folder, and help users understand the organizational structure of the project.

**Instructions for use**:
   - Provide clear steps on how to clone the project repository, install dependencies, run the project code, etc., to ensure that users can run the project smoothly.

**Code Sample**
   - Provide key code examples, show the core functions and usage methods of the project, and help users understand the implementation and operation of the project code.

**Results and Analysis**:
   - Summarize the main findings and results of the analysis process, and display the corresponding charts and visualizations to help users understand the analysis process and conclusions.

**Contribution Guidelines**:
   - Provide detailed contribution guidelines on how to contribute to the project, including forks, pull requests, and other processes, and encourage users to participate in the development of the project.

**License**
   - Declare the license type of the project, and clarify the use and distribution rights of the project.

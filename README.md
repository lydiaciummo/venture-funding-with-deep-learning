# Venture Funding With Deep Learning
This project uses TensorFlow and Keras to create a binary classification model that determines whether a startup should receive funding from a venture capital firm. I tested 3 different neural network models to see which one performs the best. For this project, I was asked to address the following scenario:

>You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked you to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup. The business team has given you a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful.

---

## Process

### Preparing the Data
To prepare the data to be used on the neural network model, I used `OneHotEncoder` to encode the categorical variables and then scaled the data using `StandardScaler`. The following code shows the process for encoding the categorical variables:

```
# Create a list of categorical variables 
categorical_variables = list(applicant_data_df.dtypes[applicant_data_df.dtypes == 'object'].index)

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Encode the categorcal variables using OneHotEncoder
encoded_data = enc.fit_transform(applicant_data_df[categorical_variables])

# Create a DataFrame of with the encoded variables
encoded_df = pd.DataFrame(
    encoded_data,
    columns=enc.get_feature_names_out(categorical_variables)
)
```

After encoding the categorical variables, I used `pd.concat` to concatenate them into a DataFrame with the numerical variables. From this dataframe, I separated the features columns from the target column and used `train_test_split` to split the features and target sets into testing and training sets:

```
# Define the target set y using the IS_SUCCESSFUL column
y = encoded_df['IS_SUCCESSFUL']

# Define features set X by selecting all columns but IS_SUCCESSFUL
X = encoded_df.drop(columns='IS_SUCCESSFUL')

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

For the last step of preparing the data, I used `StandardScaler` to scale the training and testing datasets:

```
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the features training dataset
X_scaler = scaler.fit(X_train)

# Fit the scaler to the features training dataset
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

### Compile 3 Different Binary Classification Models Using Neural Networks
The goal of this section of the project was to test neural networks with different attributes to see if one would perform better than the others. My first model contained 116 input features, 1 output node (successful or not successful), and two hidden layers, the first layer containing 58 nodes and the second layer containing 29 nodes. For the hidden layers I used the Rectified Linear Unit (ReLU) activation function, and the Sigmoid activation function for the output layer.

```
# Define the the number of inputs (features) to the model
number_input_features = 116

# Define the number of neurons in the output layer
number_output_neurons = 1

# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1 =  round((number_input_features + number_output_neurons) / 2)

# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2 =  hidden_nodes_layer1 / 2

# Create the Sequential model instance
nn = Sequential()

# Add the first hidden layer
nn.add(Dense(
    units=hidden_nodes_layer1,
    input_dim=number_input_features,
    activation='relu'
))

# Add the second hidden layer
nn.add(Dense(
    units=hidden_nodes_layer2,
    activation='relu'
))

# Add the output layer to the model specifying the number of output neurons and activation function
nn.add(Dense(
    units=number_output_neurons,
    activation='sigmoid'
))
```

To compile the model, I used the Binary Crossentropy loss function, the Adam optimizer, and the Accuracy metric. I then fitted the model to the scaled training data with 50 epochs.

```
# Compile the Sequential model
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model using 50 epochs and the training data
model_1 = nn.fit(X_train_scaled, y_train, epochs=50, verbose=0)
```

For the other two models I repeated the same process with some modifications. For the first alternative model, I removed the 'STATUS' and 'SPECIAL_CONSIDERATIONS' features to see if that would make any difference, but kept everything else the same.

For the second alternative model, I used only 1 hidden layer instead of 2, because I have read that, for most binary classification models, 1 layer should be sufficient.

Evaluating the results of the models, I saw very little difference in the loss and accuracy. The results are as follows:

```
Original Model Results
268/268 - 0s - loss: 0.5564 - accuracy: 0.7305 - 170ms/epoch - 636us/step
Loss: 0.556383490562439, Accuracy: 0.7304956316947937

Alternative Model 1 Results
268/268 - 0s - loss: 0.5532 - accuracy: 0.7304 - 244ms/epoch - 911us/step
Loss: 0.5532251596450806, Accuracy: 0.7303789854049683

Alternative Model 2 Results
268/268 - 0s - loss: 0.5561 - accuracy: 0.7315 - 230ms/epoch - 858us/step
Loss: 0.5561337471008301, Accuracy: 0.7315452098846436
```

If I were testing these models in a real-world setting, I would continue testing different methods to see if I could get better accuracy. One possible solution would be to use Principal Component Analysis (PCA) to reduce the number of input features.

---

## Technologies
* Python 3.9
* Python libraries: Pandas, Pathlib, TensorFlow, Scikit-learn
* Jupyter Lab and Jupyter Notebooks

---

## Contributors

Lydia Ciummo - lydiaciummo@hotmail.com

---

## License

GNU General Public License v3.0
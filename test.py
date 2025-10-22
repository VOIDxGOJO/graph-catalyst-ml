import tdc
from tdc.single_label import D4

# Load the dataset, for example, "D4" dataset for regression
dataset = D4()
train_data = dataset.get_train()
test_data = dataset.get_test()

# Print the first few rows of the training data
print(train_data.head())

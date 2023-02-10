In this directory you’ll find the following files:


* train.csv - 
This is the training data, containing a time series of different features regarding the SKUs at the different stores. These include store_id, sku_category and tot_promoted - which represents the total number of SKUs being promoted in that store for that date. In addition, there’s a column describing the target variable - sales.
* test.csv - 
This is the test data, for the dates of which you are required to predict the sales. It has the same features as the training data of course.
* stores.csv - 
Additional file including metadata regarding the different stores, such as geographical location, association with a specific group (type) of stores, etc.